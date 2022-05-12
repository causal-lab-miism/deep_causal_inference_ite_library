import logging
import warnings
from models.CausalModel import *
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from utils.layers import FullyConnected
from tensorflow.keras import regularizers
from utils.callback import callbacks
from utils.set_seed import setSeed
import keras_tuner as kt
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
logging.getLogger('tensorflow').disabled = True
tfd = tfp.distributions
tfpl = tfp.layers
tfk = tf.keras
tfkl = tf.keras.layers


class HyperTEDVAE(kt.HyperModel, CausalModel):
    def __init__(self, params):
        super().__init__()
        self.params = params

    def build(self, hp):
        model = TEDVAEModel(name='tedvae', params=self.params, hp=hp)
        lr = hp.Choice("lr", [1e-4, 1e-3])
        model.compile(optimizer=Adam(learning_rate=lr))

        return model

    # def fit(self, hp, model, *args, **kwargs):
    #     return model.fit(
    #         *args,
    #         batch_size=hp.Choice('batch_size', [32, 64, 128, 256, 512, 1024]),
    #         **kwargs,
    #     )


class BernoulliNet(Model):
    def __init__(self, units, out_size, num_layers, params, name, **kwargs):
        super(BernoulliNet, self).__init__(name=name, **kwargs)
        self.fully_connected = FullyConnected(n_fc=num_layers, hidden_phi=units, out_size=out_size,
                                              final_activation=None, kernel_init=params['kernel_init'],
                                              kernel_reg=regularizers.l2(params['reg_l2']), name=name)
        self.bern_dist = tfp.layers.DistributionLambda(lambda t: tfd.Bernoulli(dtype=tf.float32,
                                                                               logits=tf.clip_by_value(t,
                                                                                                       clip_value_min=0,
                                                                                                       clip_value_max=1)))

    def call(self, input):
        z = self.fully_connected(input)
        out = self.bern_dist(z)
        return out


class GaussianNet(Model):
    def __init__(self, units, out_size, num_layers, params, name, **kwargs):
        super(GaussianNet, self).__init__(name=name, **kwargs)
        self.fully_connected = FullyConnected(n_fc=num_layers, hidden_phi=units, out_size=2*out_size,
                                              final_activation=None, kernel_init=params['kernel_init'],
                                              kernel_reg=regularizers.l2(params['reg_l2']), name=name)
        self.gaus_dist = tfp.layers.DistributionLambda(lambda t: tfp.distributions.MultivariateNormalDiag(
            loc=tf.clip_by_value(t[..., :out_size], clip_value_min=-1e2, clip_value_max=1e2),
            scale_diag=tf.clip_by_value(1e-3 + tf.math.softplus(t[..., out_size:]), clip_value_min=0,
                                        clip_value_max=1e2)))

    def call(self, input):
        z = self.fully_connected(input)
        out = self.gaus_dist(z)
        return out


class GaussianNet_KL(Model):
    def __init__(self, units, out_size, num_layers, params, name, kl_weight=1.0, **kwargs):
        super(GaussianNet_KL, self).__init__(name=name,  **kwargs)
        self.fully_connected = FullyConnected(n_fc=num_layers, hidden_phi=units, out_size=2*out_size,
                                              final_activation=None, kernel_init=None,
                                              kernel_reg=regularizers.l2(params['reg_l2']), name=name)
        self.prior = tfd.Independent(tfd.MultivariateNormalDiag(loc=tf.zeros(out_size), scale_diag=tf.ones(out_size)))
        self.gaus_dist = tfp.layers.DistributionLambda(lambda t: tfp.distributions.MultivariateNormalDiag(
            loc=tf.clip_by_value(t[..., :out_size], clip_value_min=-1e2, clip_value_max=1e2),
            scale_diag=tf.clip_by_value(1e-3 + tf.math.softplus(t[..., out_size:]), clip_value_min=0,
                                        clip_value_max=1e2)),
                                                       activity_regularizer=tfpl.KLDivergenceRegularizer(
                                                           self.prior, weight=kl_weight),)

    def call(self, input):
        z = self.fully_connected(input)
        out = self.gaus_dist(z)
        return out


class TEDVAEModel(Model):
    def __init__(self, name, params, hp, **kwargs):
        super(TEDVAEModel, self).__init__(name=name, **kwargs)

        self.params = params

        self.kl_weight = 1.0/params['batch_size']

        self.hp_fc_enc_x = hp.Int('hp_fc_enc_x', min_value=2, max_value=10, step=1)
        self.hp_hidden_phi_enc_x = hp.Int('hp_hidden_phi_enc_x', min_value=16, max_value=512, step=16)
        self.hp_fc_dec_x = hp.Int('hp_fc_dec_x', min_value=2, max_value=10, step=1)
        self.hp_hidden_phi_dec_x = hp.Int('hp_hidden_phi_dec_x', min_value=16, max_value=512, step=16)

        self.encoder_x = GaussianNet_KL(units=self.hp_hidden_phi_enc_x, out_size=params['latent_dim_z'],
                                        num_layers=self.hp_fc_enc_x, params=params, name='x_encoder')

        self.encoder_t = GaussianNet_KL(units=self.hp_hidden_phi_enc_x, out_size=params['latent_dim_zt'],
                                        num_layers=self.hp_fc_enc_x, params=params, name='t_encoder')

        self.encoder_y = GaussianNet_KL(units=self.hp_hidden_phi_enc_x, out_size=params['latent_dim_zy'],
                                        num_layers=self.hp_fc_enc_x, params=params, name='y_encoder')

        self.decoder_x_bin = BernoulliNet(units=self.hp_hidden_phi_dec_x, out_size=params['num_bin'],
                                          num_layers=self.hp_fc_dec_x, params=params, name='decoder_x_bin')

        self.decoder_x_cont = GaussianNet(units=self.hp_hidden_phi_dec_x, out_size=params['num_cont'],
                                          num_layers=self.hp_fc_dec_x, params=params, name='decoder_x_cont')

        self.decoder_t = BernoulliNet(units=self.hp_hidden_phi_dec_x, out_size=1, num_layers=2, params=params,
                                      name='decoder_t')
        self.decoder_y0 = GaussianNet(units=self.hp_hidden_phi_dec_x, out_size=1, num_layers=self.hp_fc_dec_x,
                                      params=params, name='decoder_y0')
        self.decoder_y1 = GaussianNet(units=self.hp_hidden_phi_dec_x, out_size=1, num_layers=self.hp_fc_dec_x,
                                      params=params, name='decoder_y1')

        self.alpha_t = 50
        self.alpha_y = 100

    def compile(self, optimizer):
        super(TEDVAEModel, self).compile()
        self.optimizer = optimizer
        self.loss_metric = tf.keras.metrics.Mean(name="loss_metric")

    @property
    def metrics(self):
        return [self.loss_metric]

    def test_step(self, data):
        y = tf.expand_dims(tf.cast(data[:, 0], dtype=tf.float32), axis=1)  # get individual vectors
        t = tf.expand_dims(data[:, 1], axis=1)
        x = tf.cast(data[:, 2:], dtype=tf.float32)
        x_cont = x[:, :self.params['num_cont']]
        x_bin = x[:, self.params['num_cont']:]

        z = self.encoder_x(x)
        zt = self.encoder_t(x)
        zy = self.encoder_y(x)

        zty_concat = tf.concat([z, zt, zy], axis=-1)
        zt_concat = tf.concat([z, zt], axis=-1)
        zy_concat = tf.concat([z, zy], axis=-1)

        x_bin_pred = self.decoder_x_bin(zty_concat)
        x_cont_pred = self.decoder_x_cont(zty_concat)

        t_pred = self.decoder_t(zt_concat)

        y0_pred = self.decoder_y0(zy_concat)
        y1_pred = self.decoder_y1(zy_concat)

        t = tf.cast(t, tf.float32)
        loc_y = (1 - t) * y0_pred.mean() + t * y1_pred.mean()
        scale_y = (1 - t) * y0_pred.variance() + t * y1_pred.variance()

        y_pred = tfd.Independent(tfp.distributions.Normal(loc=loc_y, scale=scale_y))

        # loss_x_binary = tf.reduce_mean(-tf.nn.sigmoid_cross_entropy_with_logits(labels=x_bin, logits=x_bin_pred))
        loss_x_binary = tf.reduce_mean(-x_bin_pred.log_prob(x_bin))
        loss_x_cont = tf.reduce_mean(-x_cont_pred.log_prob(x_cont))

        loss_t_pred = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=t, logits=t_pred))
        loss_y_pred = tf.reduce_mean(tf.square(y - y_pred.mean()))

        loss = loss_x_binary + loss_x_cont + self.alpha_t*loss_t_pred + self.alpha_y*loss_y_pred

        return {
            "loss": loss,
        }

    def train_step(self, data):
        y = tf.expand_dims(tf.cast(data[:, 0], dtype=tf.float32), axis=1)  # get individual vectors
        t = tf.expand_dims(data[:, 1], axis=1)
        x = tf.cast(data[:, 2:], dtype=tf.float32)
        x_cont = x[:, :self.params['num_cont']]
        x_bin = x[:, self.params['num_cont']:]

        with tf.GradientTape() as tape:

            """Encoder"""

            z = self.encoder_x(x)
            zt = self.encoder_t(x)
            zy = self.encoder_y(x)

            """Decoder"""

            zty_concat = tf.concat([z, zt, zy], axis=-1)
            zt_concat = tf.concat([z, zt], axis=-1)
            zy_concat = tf.concat([z, zy], axis=-1)

            x_bin_pred = self.decoder_x_bin(zty_concat)
            x_cont_pred = self.decoder_x_cont(zty_concat)

            t_pred = self.decoder_t(zt_concat)

            y0_pred = self.decoder_y0(zy_concat)
            y1_pred = self.decoder_y1(zy_concat)

            t = tf.cast(t, tf.float32)
            loc_y = (1-t)*y0_pred.mean() + t*y1_pred.mean()
            scale_y = (1-t)*y0_pred.variance() + t*y1_pred.variance()

            y_pred = tfd.Independent(tfp.distributions.Normal(loc=loc_y, scale=scale_y))

            loss_x_binary = tf.reduce_mean(-x_bin_pred.log_prob(x_bin))
            loss_x_cont = tf.reduce_mean(-x_cont_pred.log_prob(x_cont))

            loss_t_pred = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=t, logits=t_pred))
            loss_y_pred = tf.reduce_mean(tf.square(y-y_pred.mean()))

            if self.params['dataset_name'] in ['acic']:
                loss = loss_x_binary + self.alpha_t * loss_t_pred + self.alpha_y * loss_y_pred
            elif self.params['dataset_name'] in ['jobs', 'twins']:
                loss = loss_x_cont + self.alpha_t * loss_t_pred + self.alpha_y * loss_y_pred
            else:
                loss = loss_x_binary + loss_x_cont + self.alpha_t*loss_t_pred + self.alpha_y*loss_y_pred

        # Get the gradients w.r.t the generator loss
        trainable_variables = self.encoder_x.trainable_variables + self.encoder_t.trainable_variables + \
                              self.encoder_y.trainable_variables + self.decoder_x_bin.trainable_variables + \
                              self.decoder_x_cont.trainable_variables + self.decoder_t.trainable_variables + \
                              self.decoder_y0.trainable_variables + self.decoder_y1.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        # Update the weights of the optimizer
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        # Update metrics
        self.loss_metric.update_state(loss)
        return {
            "loss": self.loss_metric.result(),
        }



class TEDVAE(CausalModel):
    def __init__(self, params):
        super().__init__(params)
        self.params = params

    def fit_model(self, x, y, t, seed):
        directory_name = 'params/' + self.params['dataset_name']
        setSeed(seed)
        t = tf.cast(t, dtype=tf.float32)

        if self.dataset_name == 'acic':
            directory_name = directory_name + f'/{self.params["model_name"]}'
            project_name = str(self.folder_ind)
        else:
            project_name = self.params["model_name"]

        tuner = kt.RandomSearch(
            HyperTEDVAE(params=self.params),
            objective=kt.Objective("val_loss", direction="min"),
            directory=directory_name,
            tuner_id='1',
            overwrite=False,
            project_name=project_name,
            max_trials=10,
            seed=0)

        ytx = np.concatenate([y, t, x], 1)
        stop_early = [TerminateOnNaN(), EarlyStopping(monitor='val_loss', patience=5)]
        tuner.search(ytx, epochs=50, validation_split=0.2, batch_size=self.params['batch_size'],
                     callbacks=[stop_early], verbose=0)

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        model = tuner.hypermodel.build(best_hps)

        model.fit(ytx, epochs=self.params['epochs'], callbacks=callbacks('loss'),
                  batch_size=self.params['batch_size'], validation_split=0.0,
                  verbose=self.params['verbose'])

        return model

    @staticmethod
    def evaluate(x_test, model):
        z = model.encoder_x(x_test)
        zy = model.encoder_y(x_test)
        zy_concat = tf.concat([z, zy], axis=-1)
        y0_pred = model.decoder_y0(zy_concat)
        y1_pred = model.decoder_y1(zy_concat)

        return tf.concat([y0_pred.mean(), y1_pred.mean()], axis=-1)

    def train_and_evaluate(self, metric_list, **kwargs):
        data_train, data_test = self.load_data(**kwargs)

        self.folder_ind = kwargs.get('folder_ind')

        if self.params['binary']:
            model = self.fit_model(data_train['x'], data_train['y'], data_train['t'], seed=0)
        else:
            model = self.fit_model(data_train['x'], data_train['ys'], data_train['t'], seed=0)

        concat_pred = self.evaluate(data_test['x'], model)

        # don't forget to rescale the outcome before estimation!
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
