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


class HyperCEVAE(kt.HyperModel, CausalModel):
    def __init__(self, params, bs):
        super().__init__()
        self.params = params
        self.bs = bs

    def build(self, hp):
        model = CEVAEModel(name='cevae', params=self.params, hp=hp)
        model.compile(optimizer=Adam(learning_rate=self.params['lr']))

        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=self.params['batch_size'],
            **kwargs,
        )


class BernoulliNet(Model):
    def __init__(self, units, out_size, num_layers, params, name, **kwargs):
        super(BernoulliNet, self).__init__(name=name, **kwargs)
        self.fully_connected = FullyConnected(n_fc=num_layers, hidden_phi=units, out_size=out_size,
                                              final_activation=None, kernel_init=None,
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


class BernoulliNet_KL(Model):
    def __init__(self, units, out_size, num_layers, params, name, kl_weight=1.0, **kwargs):
        super(BernoulliNet_KL, self).__init__(name=name, **kwargs)
        self.fully_connected = FullyConnected(n_fc=num_layers, hidden_phi=units, out_size=out_size,
                                              final_activation=None, kernel_init=None,
                                              kernel_reg=regularizers.l2(params['reg_l2']), name=name)
        self.prior = tfd.Independent(tfd.Bernoulli(logits=tf.random.uniform(shape=[out_size], minval=0, maxval=1,
                                                                            dtype=tf.float32)))
        self.bern_dist = tfp.layers.DistributionLambda(lambda t:
                                                       tfd.Bernoulli(dtype=tf.float32,
                                                                     logits=tf.clip_by_value(t, clip_value_min=0,
                                                                                             clip_value_max=11)),
                                                       activity_regularizer=tfpl.KLDivergenceRegularizer(
                                                           self.prior, weight=kl_weight))

    def call(self, input):
        z = self.fully_connected(input)
        out = self.bern_dist(z)
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


class CEVAEModel(Model):
    def __init__(self, name, params, hp, **kwargs):
        super(CEVAEModel, self).__init__(name=name, **kwargs)
        self.params = params
        self.kl_weight = 1.0/params['batch_size']

        self.hp_fc_latent_y = hp.Int('hp_fc_latent_y', min_value=2, max_value=5, step=1)
        self.hp_hidden_phi_latent_y = hp.Int('hp_hidden_phi_latent_y', min_value=16, max_value=512, step=16)

        self.hp_fc_x_latent = hp.Int('hp_fc_x_latent', min_value=2, max_value=8, step=1)
        self.hp_hidden_x_latent = hp.Int('hp_hidden_x_latent', min_value=16, max_value=64, step=8)

        self.hp_hidden_phi_dec_t = hp.Int('hp_hidden_phi_dec_t', min_value=16, max_value=64, step=8)

        self.t_distribution = BernoulliNet(units=self.params['latent_dim'], out_size=1, num_layers=1, params=params,
                                           name='t_distribution')

        self.y_latent_space = FullyConnected(n_fc=self.hp_fc_latent_y, hidden_phi=self.hp_hidden_phi_latent_y,
                                             out_size=1, final_activation=None, kernel_init=None,
                                             kernel_reg=regularizers.l2(params['reg_l2']), name='y_latent_space')

        self.y0_distribution = GaussianNet_KL(units=self.hp_hidden_phi_latent_y, out_size=1, num_layers=2,
                                              params=params, name='y0_distribution', kl_weight=self.kl_weight)
        self.y1_distribution = GaussianNet_KL(units=self.hp_hidden_phi_latent_y, out_size=1, num_layers=2,
                                              params=params, name='y1_distribution', kl_weight=self.kl_weight)

        self.z_latent_space = FullyConnected(n_fc=self.hp_fc_latent_y, hidden_phi=self.hp_hidden_phi_latent_y,
                                             out_size=1, final_activation=None, kernel_init=params['kernel_init'],
                                             kernel_reg=regularizers.l2(params['reg_l2']), name='z_latent_space')

        self.encoder_y0 = GaussianNet_KL(units=self.hp_hidden_phi_latent_y, out_size=params['latent_dim'], num_layers=2,
                                         params=params, name='encoder_y0')
        self.encoder_y1 = GaussianNet_KL(units=self.hp_hidden_phi_latent_y, out_size=params['latent_dim'], num_layers=2,
                                         params=params, name='encoder_y1')

        self.x_latent_space = FullyConnected(n_fc=self.hp_fc_x_latent, hidden_phi=self.hp_hidden_x_latent,
                                             out_size=1, final_activation=None, kernel_init=params['kernel_init'],
                                             kernel_reg=regularizers.l2(params['reg_l2']), name='x_latent_space')

        self.decoder_x_bin = BernoulliNet(units=self.hp_hidden_x_latent, out_size=params['num_bin'], num_layers=2,
                                          params=params, name='x_bin_decoder')

        self.decoder_x_cont = GaussianNet(units=self.hp_hidden_x_latent, out_size=params['num_cont'], num_layers=2,
                                          params=params, name='x_cont_decoder')

        self.decoder_t = BernoulliNet(units=self.hp_hidden_phi_dec_t, out_size=1, num_layers=2, params=params,
                                      name='t_distribution')

        """Note that the original paper only samples mean and sets variance = 1 for y0|z and y1|z"""

        self.decoder_y0 = GaussianNet(units=self.hp_hidden_phi_latent_y, out_size=1,
                                      num_layers=self.hp_fc_latent_y, params=params, name='decoder_y0')

        self.decoder_y1 = GaussianNet(units=self.hp_hidden_phi_latent_y, out_size=1,
                                      num_layers=self.hp_fc_latent_y, params=params, name='decoder_y1')

        self.alpha_t = 50
        self.alpha_y = 100

    def compile(self, optimizer):
        super(CEVAEModel, self).compile()
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

        t_x = self.t_distribution(x)
        phi_y = self.y_latent_space(x)
        y0_dist = self.y0_distribution(phi_y)
        y1_dist = self.y1_distribution(phi_y)

        t_x = tf.cast(t_x, tf.float32)
        loc_y = (1-t_x)*y0_dist.mean() + t_x*y1_dist.mean()
        y0_var = tf.expand_dims(tf.math.reduce_variance(y0_dist.sample(), axis=1), axis=1)
        y1_var = tf.expand_dims(tf.math.reduce_variance(y1_dist.sample(), axis=1), axis=1)
        scale_y = (1 - t_x) * y0_var + t_x * y1_var

        y_x = tfd.Independent(tfp.distributions.Normal(loc=loc_y, scale=scale_y))

        x_yt = tf.concat([x, y_x.sample()], -1)

        z0_dist = self.encoder_y0(x_yt)
        z1_dist = self.encoder_y1(x_yt)

        loc_z = (1-t_x)*z0_dist.mean() + t_x*z1_dist.mean()
        scale_z = (1-t_x)*z0_dist.variance() + t_x*z1_dist.variance()

        z = tfp.distributions.Normal(loc=loc_z, scale=scale_z)

        x_bin_pred = self.decoder_x_bin(z.sample())
        x_cont_pred = self.decoder_x_cont(z.sample())

        t_pred = self.decoder_t(z.sample())

        y0_pred = self.decoder_y0(z.sample())
        y1_pred = self.decoder_y1(z.sample())

        t = tf.cast(t, tf.float32)
        loc_y = (1-t)*y0_pred.mean() + t*y1_pred.mean()
        scale_y = (1-t)*y0_pred.variance() + t*y1_pred.variance()

        y_pred = tfd.Independent(tfp.distributions.Normal(loc=loc_y, scale=scale_y))

        loss = -tf.reduce_mean(tf.reduce_sum(y_pred.log_prob(y) + t_pred.log_prob(t), axis=1) +
                               tf.reduce_sum(x_bin_pred.log_prob(x_bin), axis=1) +
                               tf.reduce_sum(x_cont_pred.log_prob(x_cont), axis=0))

        return {
            "loss": loss,
        }

    def train_step(self, data):
        y = tf.expand_dims(tf.cast(data[:, 0], dtype=tf.float32), axis=1)
        t = tf.expand_dims(data[:, 1], axis=1)
        x = tf.cast(data[:, 2:], dtype=tf.float32)
        x_cont = x[:, :self.params['num_cont']]
        x_bin = x[:, self.params['num_cont']:]

        with tf.GradientTape() as tape:

            """Encoder"""

            t_x = self.t_distribution(x)
            phi_y = self.y_latent_space(x)
            y0_dist = self.y0_distribution(phi_y)
            y1_dist = self.y1_distribution(phi_y)

            t_x = tf.cast(t_x, tf.float32)
            loc_y = (1-t_x)*y0_dist.mean() + t_x*y1_dist.mean()
            y0_var = tf.expand_dims(tf.math.reduce_variance(y0_dist.sample(), axis=1), axis=1)
            y1_var = tf.expand_dims(tf.math.reduce_variance(y1_dist.sample(), axis=1), axis=1)
            scale_y = (1 - t_x) * y0_var + t_x * y1_var

            y_x = tfd.Independent(tfp.distributions.Normal(loc=loc_y, scale=scale_y))

            x_yt = tf.concat([x, y_x.sample()], -1)


            z0_dist = self.encoder_y0(x_yt)
            z1_dist = self.encoder_y1(x_yt)

            loc_z = (1-t_x)*z0_dist.mean() + t_x*z1_dist.mean()
            scale_z = (1-t_x)*z0_dist.variance() + t_x*z1_dist.variance()

            z = tfp.distributions.Normal(loc=loc_z, scale=scale_z)

            """Decoder"""

            x_bin_pred = self.decoder_x_bin(z.sample())
            x_cont_pred = self.decoder_x_cont(z.sample())

            t_pred = self.decoder_t(z.sample())

            y0_pred = self.decoder_y0(z.sample())
            y1_pred = self.decoder_y1(z.sample())

            t = tf.cast(t, tf.float32)
            loc_y = (1-t)*y0_pred.mean() + t*y1_pred.mean()
            scale_y = (1-t)*y0_pred.variance() + t*y1_pred.variance()

            y_pred = tfd.Independent(tfp.distributions.Normal(loc=loc_y, scale=scale_y))

            loss = -tf.reduce_mean(tf.reduce_sum(y_pred.log_prob(y) + t_pred.log_prob(t), axis=1) +
                                   tf.reduce_sum(x_bin_pred.log_prob(x_bin), axis=1) +
                                   tf.reduce_sum(x_cont_pred.log_prob(x_cont), axis=0))

        trainable_variables = self.t_distribution.trainable_variables + self.y_latent_space.trainable_variables + \
                              self.y0_distribution.trainable_variables + self.y1_distribution.trainable_variables + \
                              self.z_latent_space.trainable_variables + self.encoder_y0.trainable_variables + \
                              self.encoder_y1.trainable_variables + self.x_latent_space.trainable_variables + \
                              self.decoder_x_bin.trainable_variables + self.decoder_x_cont.trainable_variables + \
                              self.decoder_t.trainable_variables + self.decoder_y0.trainable_variables + \
                              self.decoder_y1.trainable_variables

        gradients = tape.gradient(loss, trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        self.loss_metric.update_state(loss)
        return {
            "loss": self.loss_metric.result(),
        }


class CEVAE(CausalModel):
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
            HyperCEVAE(params=self.params, bs=self.params['batch_size']),
            objective=kt.Objective("val_loss", direction="min"),
            directory=directory_name,
            tuner_id='1',
            overwrite=False,
            project_name=project_name,
            max_trials=30,
            seed=0)

        ytx = np.concatenate([y, t, x], 1)
        stop_early = [TerminateOnNaN(), EarlyStopping(monitor='val_loss', patience=5)]
        tuner.search(ytx, epochs=50, validation_split=0.2,
                     callbacks=[stop_early], verbose=1)

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        model = tuner.hypermodel.build(best_hps)
        model.fit(ytx,
                  epochs=self.params['epochs'],
                  callbacks=callbacks('loss'),
                  batch_size=self.params['batch_size'],
                  verbose=self.params['verbose'],
                  validation_split=0.0)

        return model

    @staticmethod
    def evaluate(x_test, model):
        x_test = tf.cast(x_test, dtype=tf.float32)
        phi_y = model.y_latent_space(x_test)
        y0 = model.y0_distribution(phi_y)
        y1 = model.y1_distribution(phi_y)
        z0_dist = model.encoder_y0(tf.concat([x_test, y0], -1))
        z1_dist = model.encoder_y1(tf.concat([x_test, y1], -1))
        y0_pred = model.decoder_y0(z0_dist)
        y1_pred = model.decoder_y1(z1_dist)

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
