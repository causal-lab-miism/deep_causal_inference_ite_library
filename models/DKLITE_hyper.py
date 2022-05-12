from models.CausalModel import *
import tensorflow as tf
import numpy as np
import os
import warnings
import logging
from tensorflow.keras.losses import Loss
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from utils.layers import FullyConnected
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt
from utils.callback import callbacks
from tensorflow.keras.callbacks import EarlyStopping
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
logging.getLogger('tensorflow').disabled = True


class HyperDklite(kt.HyperModel, CausalModel):
    def __init__(self, params):
        super().__init__()
        self.params = params

    def build(self, hp):
        model = DKLITELearner(name='DKLITE', params=self.params, hp=hp, dim_z=self.params['dim_z'])
        dklite_loss = DKLITELoss(params=self.params, dim_z=self.params['dim_z'])
        model.compile(optimizer=Adam(learning_rate=self.params['lr']),
                      loss=dklite_loss,
                      metrics=dklite_loss.regression_loss)
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=self.params['tuner_batch_size'],
            **kwargs,
        )


class DKLITELearner(Model):
    def __init__(self, params, hp, dim_z, name="DKLITE", **kwargs):
        super(DKLITELearner, self).__init__(name=name, **kwargs)
        self.params = params
        self.dim_z = dim_z
        self.n_fc_encoder = hp.Int('n_fc_encoder', min_value=2, max_value=10, step=1)
        self.hidden_phi_encoder = hp.Int('hidden_phi_encoder', min_value=16, max_value=512, step=16)

        self.encoder = FullyConnected(n_fc=self.n_fc_encoder, hidden_phi=self.hidden_phi_encoder,
                                      final_activation=self.params['activation'], out_size=self.dim_z,
                                      kernel_init=self.params['kernel_init'],
                                      kernel_reg=regularizers.l2(self.params['reg_l2']), name='encoder')
        self.n_fc_decoder = hp.Int('n_fc_decoder', min_value=2, max_value=10, step=1)
        self.hidden_phi_decoder = hp.Int('hidden_phi_decoder', min_value=16, max_value=512, step=16)
        self.decoder = FullyConnected(n_fc=self.n_fc_decoder, hidden_phi=self.hidden_phi_decoder,
                                      final_activation=self.params['activation'], out_size=params['x_size'],
                                      kernel_init=self.params['kernel_init'],
                                      kernel_reg=regularizers.l2(self.params['reg_l2']), name='decoder')

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return tf.concat([encoded, decoded], axis=1)


def gp_nn(y_f, z_f):
    beta = tf.ones([1, 1], tf.float32)
    lam = 1000 * tf.ones([1, 1], tf.float32)
    r = beta / lam
    dd = tf.shape(z_f)[1]
    phi_phi = tf.matmul(tf.transpose(z_f), z_f)
    ker = r * phi_phi + tf.eye(tf.shape(z_f)[1], dtype=tf.float32)
    l_matrix = tf.linalg.cholesky(ker)
    l_inv_reduce = tf.linalg.triangular_solve(l_matrix, rhs=tf.eye(dd, dtype=tf.float32))
    l_y = tf.matmul(l_inv_reduce, tf.matmul(tf.transpose(z_f), tf.expand_dims(tf.cast(y_f, dtype=tf.float32), axis=1)))
    ker_inv = tf.matmul(tf.transpose(l_inv_reduce), l_inv_reduce) / lam
    mean = r * tf.matmul(tf.transpose(l_inv_reduce), l_y)
    term1 = - tf.reduce_mean(tf.square(l_y))
    # term2 = tf.log(tf.linalg.diag_part(l_matrix)) / ((1-index)*tf.reduce_sum(1 - self.T) + (index)* tf.reduce_sum(self.T))
    ml_primal = term1  # +  term2

    return ker_inv, mean, ml_primal


class DKLITELoss(Loss):
    def __init__(self, params, dim_z, name="dklite_loss"):
        super().__init__(name=name)
        self.params = params
        self.dim_z = dim_z
        self.reg_var = params['reg_var']
        self.reg_rec = params['reg_rec']

    # compute loss
    def call(self, concat_true, concat_pred):
        return self.prediction_loss(concat_true, concat_pred)

    def regression_loss(self, concat_true, concat_pred):
        # source: https://github.com/kochbj/Deep-Learning-for-Causal-Inference
        z = concat_pred[:, :self.dim_z]
        x_dec = concat_pred[:, self.dim_z:]

        y = concat_true[:, 0]  # get individual vectors
        t = concat_true[:, 1]

        z_0 = tf.gather(z, tf.where(t < 0.5)[:, 0])
        y_0 = tf.gather(y, tf.where(t < 0.5)[:, 0])
        z_1 = tf.gather(z, tf.where(t > 0.5)[:, 0])
        y_1 = tf.gather(y, tf.where(t > 0.5)[:, 0])

        mean_0 = tf.reduce_mean(y_0)
        mean_1 = tf.reduce_mean(y_1)

        y_0 = tf.subtract(y_0, mean_0)
        y_1 = tf.subtract(y_1, mean_1)

        _, mean_0_gp, _ = gp_nn(y_0, z_0)
        _, mean_1_gp, _ = gp_nn(y_1, z_1)

        y0_pred = tf.matmul(z_0, mean_0_gp) + mean_0
        y1_pred = tf.matmul(z_1, mean_1_gp) + mean_1

        # print(np.mean(np.square(Y[T==0] - Y_hat_test[:, 0][np.squeeze(T==0)])) + np.mean(np.square(Y[T==1] - Y_hat_test[:, 1][np.squeeze(T==1)])))
        # y0_pred = tf.gather(y0_pred, tf.where(t < 0.5)[:, 0])
        # y1_pred = tf.gather(y1_pred, tf.where(t > 0.5)[:, 0])
        # loss0 = tf.reduce_mean(tf.square(y_0 - y0_pred))
        # loss1 = tf.reduce_mean(tf.square(y_1 - y1_pred))

        loss0 = tf.reduce_mean((1. - t) * tf.square(y - y0_pred))
        loss1 = tf.reduce_mean(t * tf.square(y - y1_pred))


        return loss0 + loss1

    def prediction_loss(self, concat_true, concat_pred):
        # source: https://github.com/kochbj/Deep-Learning-for-Causal-Inference
        z = concat_pred[:, :self.dim_z]
        x_dec = concat_pred[:, self.dim_z:]

        y = concat_true[:, 0]  # get individual vectors
        t = concat_true[:, 1]
        x = concat_true[:, 2:]

        loss_1 = tf.reduce_mean(tf.square(x - x_dec), axis=1)

        z_0 = tf.gather(z, tf.where(t < 0.5)[:, 0])
        y_0 = tf.gather(y, tf.where(t < 0.5)[:, 0])
        z_1 = tf.gather(z, tf.where(t > 0.5)[:, 0])
        y_1 = tf.gather(y, tf.where(t > 0.5)[:, 0])

        mean_0 = tf.reduce_mean(y_0)
        mean_1 = tf.reduce_mean(y_1)

        y_0 = (y_0 - mean_0)
        y_1 = (y_1 - mean_1)

        ker_inv_0_gp, mean_0_gp, ml_primal_0_gp = gp_nn(y_0, z_0)
        ker_inv_1_gp, mean_1_gp, ml_primal_1_gp = gp_nn(y_1, z_1)

        var_0 = tf.reduce_mean(
            tf.linalg.diag_part(tf.matmul(z_1, tf.matmul(ker_inv_0_gp, tf.transpose(z_1)))))
        var_1 = tf.reduce_mean(
            tf.linalg.diag_part(tf.matmul(z_0, tf.matmul(ker_inv_1_gp, tf.transpose(z_0)))))

        prediction_loss = ml_primal_0_gp + ml_primal_1_gp + self.reg_var * (
                var_0 + var_1) + self.reg_rec * loss_1

        return prediction_loss


class DKLITE(CausalModel):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.folder_ind = None

    def fit_model(self, x, y, t, seed):
        directory_name = 'params/' + self.params['dataset_name']
        if self.dataset_name == 'acic':
            directory_name = directory_name + f'/{self.params["model_name"]}'
            project_name = str(self.folder_ind)
        else:
            project_name = self.params["model_name"]

        setSeed(seed)
        ytx = np.concatenate([y, t, x], 1)

        tuner = kt.RandomSearch(
            HyperDklite(params=self.params),
            objective=kt.Objective("val_regression_loss", direction="min"),
            directory=directory_name,
            project_name=project_name,
            max_trials=self.params['max_trials'],
            overwrite=False)

        stop_early = [EarlyStopping(monitor='val_regression_loss', patience=5)]
        tuner.search(x, ytx, epochs=50, validation_split=0.2, callbacks=[stop_early], verbose=1)

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        model = tuner.hypermodel.build(best_hps)

        # print(f"""The hyperparameter search is complete. the optimal hyperparameters are
        #       layer is n_fc_encoder={best_hps.get('n_fc_encoder')} hidden_phi_encoder = {best_hps.get('hidden_phi_encoder')}
        #       n_fc_decoder = {best_hps.get('n_fc_decoder')} hidden_phi_decoder = {best_hps.get('hidden_phi_decoder')}""")

        model.fit(x=x, y=ytx,
                  callbacks=callbacks('loss'),
                  validation_split=0.0,
                  epochs=self.params['epochs'],
                  batch_size=self.params['batch_size'],
                  verbose=self.params['verbose'])

        return model.encoder

    @staticmethod
    def compute_pehe(T_true, T_test):
        return np.sqrt(np.mean((T_true.reshape((-1, 1)) - T_test.reshape((-1, 1))) ** 2))

    def find_mean_gp(self, data, encoder):
        x = data['x']
        if self.params['binary']:
            y = data['y']
        else:
            y = data['ys']
        t = data['t']
        z = encoder(x)
        z_0 = tf.gather(z, tf.where(t < 0.5)[:, 0])
        y_0 = tf.gather(y, tf.where(t < 0.5)[:, 0])
        z_1 = tf.gather(z, tf.where(t > 0.5)[:, 0])
        y_1 = tf.gather(y, tf.where(t > 0.5)[:, 0])

        y_0 = tf.cast(y_0, tf.float32)
        y_1 = tf.cast(y_1, tf.float32)

        mean_0 = tf.reduce_mean(y_0)
        mean_1 = tf.reduce_mean(y_1)

        y_0 = tf.subtract(y_0, mean_0)
        y_1 = tf.subtract(y_1, mean_1)

        y_0 = tf.squeeze(y_0)
        y_1 = tf.squeeze(y_1)

        _, mean_0_gp, _ = gp_nn(y_0, z_0)
        _, mean_1_gp, _ = gp_nn(y_1, z_1)

        return mean_0, mean_1, mean_0_gp, mean_1_gp

    def evaluate(self, data_test, data_train, encoder):
        z_test = encoder(data_test['x'])
        mean_0, mean_1, mean_0_gp, mean_1_gp = self.find_mean_gp(data_train, encoder)

        pred_tr_0 = tf.matmul(z_test, mean_0_gp) + tf.cast(mean_0, tf.float32)
        pred_tr_1 = tf.matmul(z_test, mean_1_gp) + tf.cast(mean_1, tf.float32)
        y_test_pred = tf.concat([pred_tr_0, pred_tr_1], axis=1)
        return y_test_pred

    def train_and_evaluate(self, metric_list, **kwargs):
        data_train, data_test = self.load_data(**kwargs)

        if self.params['binary']:
            model = self.fit_model(data_train['x'], data_train['y'], data_train['t'], seed=0)
        else:
            model = self.fit_model(data_train['x'], data_train['ys'], data_train['t'], seed=0)

        concat_pred = self.evaluate(data_test, data_train, model)

        y0_pred, y1_pred = concat_pred[:, 0], concat_pred[:, 1]
        y0_pred = tf.expand_dims(y0_pred, axis=1)
        y1_pred = tf.expand_dims(y1_pred, axis=1)

        if self.params['dataset_name'] == 'jobs':
            policy_value, policy_risk, policy_curve = self.find_policy_risk(y0_pred, y1_pred, data_test)
            print(kwargs.get('count'), 'Policy Risk Test = ', policy_risk)
            metric_list.append(policy_risk)
        else:
            pehe = self.find_pehe(y0_pred, y1_pred, data_test)
            if self.params['dataset_name'] == 'acic':
                print(kwargs.get('folder_ind'), kwargs.get('file_ind'), 'Pehe Test = ', pehe)
            else:
                print(kwargs.get('count'), 'Pehe Test = ', pehe)
            metric_list.append(pehe)
