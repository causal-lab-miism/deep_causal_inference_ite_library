import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.metrics import binary_accuracy
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import SGD
from models.CausalModel import *
from utils.layers import FullyConnected
import keras_tuner as kt
from utils.callback import callbacks
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN
from utils.set_seed import setSeed


class HyperDragonNet(kt.HyperModel, CausalModel):
    def __init__(self, params):
        super().__init__()
        self.params = params

    @staticmethod
    def binary_classification_loss(concat_true, concat_pred):
        t_true = concat_true[:, 1]
        t_pred = concat_pred[:, 2]
        t_pred = (t_pred + 0.001) / 1.002
        losst = tf.reduce_sum(K.binary_crossentropy(t_true, t_pred))

        return losst

    def dragonnet_loss_binarycross(self, concat_true, concat_pred):
        return self.regression_loss(concat_true, concat_pred) + self.binary_classification_loss(concat_true,
                                                                                                concat_pred)

    @staticmethod
    def treatment_accuracy(concat_true, concat_pred):
        t_true = concat_true[:, 1]
        t_pred = concat_pred[:, 2]
        return binary_accuracy(t_true, t_pred)

    @staticmethod
    def track_epsilon(concat_true, concat_pred):
        epsilons = concat_pred[:, 3]
        return tf.abs(tf.reduce_mean(epsilons))

    @staticmethod
    def make_tarreg_loss(ratio=1., dragonnet_loss=dragonnet_loss_binarycross):
        def tarreg_ATE_unbounded_domain_loss(concat_true, concat_pred):
            vanilla_loss = dragonnet_loss(concat_true, concat_pred)

            y_true = concat_true[:, 0]
            t_true = concat_true[:, 1]

            y0_pred = concat_pred[:, 0]
            y1_pred = concat_pred[:, 1]
            t_pred = concat_pred[:, 2]

            epsilons = concat_pred[:, 3]
            # t_pred = (t_pred + 0.01) / 1.02
            t_pred = tf.clip_by_value(t_pred, 0.01, 0.99, name='t_pred')

            y_pred = t_true * y1_pred + (1 - t_true) * y0_pred

            h = t_true / t_pred - (1 - t_true) / (1 - t_pred)

            y_pert = y_pred + epsilons * h
            targeted_regularization = tf.reduce_sum(tf.square(y_true - y_pert))

            # final
            loss = vanilla_loss + ratio * targeted_regularization
            return loss

        return tarreg_ATE_unbounded_domain_loss

    def build(self, hp):
        model = DragonNetModel(name='dragonnet', params=self.params, hp=hp)
        targeted_regularization = False
        knob_loss = self.dragonnet_loss_binarycross
        ratio = 1.

        metrics = [self.regression_loss, self.binary_classification_loss, self.treatment_accuracy, self.track_epsilon]

        if targeted_regularization:
            loss = self.make_tarreg_loss(ratio=ratio, dragonnet_loss=knob_loss)
        else:
            loss = knob_loss
        model.compile(optimizer=SGD(learning_rate=self.params['lr'], momentum=0.9),
                      loss=loss, metrics=metrics)

        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=self.params['batch_size'],
            **kwargs,
        )


class EpsilonLayer(Layer):
    def __init__(self):
        super(EpsilonLayer, self).__init__()

    def build(self, input_shape):
        self.epsilon = self.add_weight(name='epsilon',
                                       shape=[1, 1],
                                       initializer='RandomNormal',
                                       #  initializer='ones',
                                       trainable=True)
        super(EpsilonLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return self.epsilon * tf.ones_like(inputs)[:, 0:1]


class DragonNetModel(Model):
    def __init__(self, name, params, hp, **kwargs):
        super(DragonNetModel, self).__init__(name=name, **kwargs)
        self.params = params
        self.hp_fc = hp.Int('n_fc', min_value=2, max_value=10, step=1)
        self.hp_hidden_phi = hp.Int('hidden_phi', min_value=16, max_value=512, step=16)
        self.fc = FullyConnected(n_fc=self.hp_fc, hidden_phi=self.hp_hidden_phi, final_activation='elu',
                                 out_size=self.hp_hidden_phi, kernel_init=params['kernel_init'], kernel_reg=None,
                                 name='fc')

        self.hp_fc_y0 = hp.Int('n_fc_y0', min_value=2, max_value=10, step=1)
        self.hp_hidden_phi_y0 = hp.Int('hidden_y0', min_value=16, max_value=512, step=16)
        self.pred_y0 = FullyConnected(n_fc=self.hp_fc_y0, hidden_phi=self.hp_hidden_phi_y0,
                                      final_activation=params['activation'], out_size=1,
                                      kernel_init=params['kernel_init'],
                                      kernel_reg=regularizers.l2(params['reg_l2']), name='y0')

        self.hp_fc_y1 = hp.Int('n_fc_y1', min_value=2, max_value=10, step=1)
        self.hp_hidden_phi_y1 = hp.Int('hidden_y1', min_value=16, max_value=512, step=16)
        self.pred_y1 = FullyConnected(n_fc=self.hp_fc_y1, hidden_phi=self.hp_hidden_phi_y1,
                                      final_activation=params['activation'], out_size=1,
                                      kernel_init=params['kernel_init'],
                                      kernel_reg=regularizers.l2(params['reg_l2']), name='y1')

        self.hp_fc_t = hp.Int('n_fc_t', min_value=2, max_value=10, step=1)
        self.hp_hidden_phi_t = hp.Int('hidden_t', min_value=16, max_value=512, step=16)
        self.pred_t = FullyConnected(n_fc=self.hp_fc_t, hidden_phi=self.hp_hidden_phi_t,
                                     final_activation='sigmoid', out_size=1,
                                     kernel_init=params['kernel_init'],
                                     kernel_reg=regularizers.l2(params['reg_l2']), name='t')

        self.dl = EpsilonLayer()

    def call(self, inputs):
        phi = self.fc(inputs)
        y0_pred = self.pred_y0(phi)
        y1_pred = self.pred_y1(phi)
        t_pred = self.pred_t(phi)

        epsilons = self.dl(t_pred, name='epsilon')
        concat_pred = tf.concat([y0_pred, y1_pred, t_pred, epsilons], axis=-1)
        return concat_pred


class DragonNet(CausalModel):
    def __init__(self, params):
        super().__init__(params)
        self.params = params

    def fit_model(self, x, y, t, seed, count):
        directory_name = 'params/' + self.params['dataset_name']
        setSeed(seed)
        t = tf.cast(t, dtype=tf.float32)

        if self.dataset_name == 'acic':
            directory_name = directory_name + f'/{self.params["model_name"]}'
            project_name = str(self.folder_ind)
        else:
            project_name = self.params["model_name"]

        tuner = kt.RandomSearch(
            HyperDragonNet(self.params),
            objective=kt.Objective("val_regression_loss", direction="min"),
            directory=directory_name,
            tuner_id='1',
            overwrite=False,
            project_name=project_name,
            max_trials=self.params['max_trials'],
            seed=0)

        yt = tf.concat([y, t], axis=1)
        stop_early = [TerminateOnNaN(), EarlyStopping(monitor='val_regression_loss', patience=5)]
        tuner.search(x, yt, epochs=50, validation_split=0.2, callbacks=[stop_early], verbose=1)

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        if self.params['dafaults']:
            best_hps.values = {'n_fc': self.params['n_fc'], 'hidden_phi': self.params['hidden_phi'],
                               'n_fc_y0': self.params['n_fc_y0'], 'n_fc_y1': self.params['n_fc_y1'],
                               'hidden_y1': self.params['hidden_y1'], 'hidden_y0': self.params['hidden_y0'],
                               'n_fc_t': self.params['n_fc_t'], 'hidden_t': self.params['hidden_t']}

        model = tuner.hypermodel.build(best_hps)

        model.fit(x, yt,
                  callbacks=callbacks('regression_loss'),
                  validation_split=0.0,
                  epochs=self.params['epochs'],
                  batch_size=self.params['batch_size'],
                  verbose=self.params['verbose'])
        if count == 0:
            print(f"""The hyperparameter search is complete. the optimal hyperparameters are
                    layer is n_fc={best_hps.get('n_fc')} hidden_phi = {best_hps.get('hidden_phi')}
                    hidden_y1 = {best_hps.get('hidden_y1')} n_fc_y1 = {best_hps.get('n_fc_y1')}
                    hidden_y0 = {best_hps.get('hidden_y0')}  n_fc_y0 = {best_hps.get('n_fc_y0')},
                    n_fc_t={best_hps.get('n_fc_t')}  hidden_t = {best_hps.get('hidden_t')}""")
            print(model.summary())

        return model

    @staticmethod
    def evaluate(x_test, model):
        return model.predict(x_test)

    def train_and_evaluate(self, metric_list, **kwargs):
        data_train, data_test = self.load_data(**kwargs)

        self.folder_ind = kwargs.get('folder_ind')
        self.sub_dataset = kwargs.get('count')
        count = kwargs.get('count')

        if self.params['binary']:
            model = self.fit_model(data_train['x'], data_train['y'], data_train['t'], count=count, seed=0)
        else:
            model = self.fit_model(data_train['x'], data_train['ys'], data_train['t'], count=count, seed=0)

        concat_pred = self.evaluate(data_test['x'], model)

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
