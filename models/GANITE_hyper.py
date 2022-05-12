from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from models.CausalModel import *
from utils.layers import FullyConnected
from utils.callback import callbacks
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN
from utils.set_seed import setSeed
import keras_tuner as kt


class HyperGANITE(kt.HyperModel, CausalModel):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.alpha = 2

    def build(self, hp):
        lr = hp.Choice("lr", [1e-5, 1e-4, 1e-3])

        generator = Generator(name='generator', params=self.params, hp=hp)
        discriminator = Discriminator(name='discriminator', params=self.params, hp=hp)

        gan = GAN(discriminator=discriminator, generator=generator, alpha=self.alpha, binary=self.params['binary'])

        optimizer_g = Adam(learning_rate=lr)
        optimizer_d = Adam(learning_rate=lr)

        gan.compile(d_optimizer=optimizer_d, g_optimizer=optimizer_g,)

        return gan

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice('batch_size', [64, 128, 256, 512]),
            **kwargs,
        )


class HyperInferenceNet(kt.HyperModel, CausalModel):
    def __init__(self, params, generator):
        super().__init__()
        self.params = params
        self.generator = generator

    def inference_loss(self, concat_true, concat_pred):
        y = concat_true[:, 0]  # get individual vectors
        t = concat_true[:, 1]
        y_tilde_0 = concat_pred[:, 0]
        y_tilde_1 = concat_pred[:, 1]

        y_hat_logit_0 = concat_pred[:, 2]
        y_hat_logit_1 = concat_pred[:, 3]
        if self.params['binary']:
            i_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=t * y + (1 - t) * y_tilde_1,
                logits=y_hat_logit_1))
            i_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=(1 - t) * y + t * y_tilde_0,
                logits=y_hat_logit_0))
        else:
            i_loss1 = tf.reduce_mean(tf.square(
                (t * y + (1 - t) * y_tilde_1 - y_hat_logit_1)))
            i_loss2 = tf.reduce_mean(tf.square(
                (1 - t) * y + t * y_tilde_0 - y_hat_logit_0))

        i_loss = i_loss1 + i_loss2

        return i_loss

    def build(self, hp):
        lr = hp.Choice("lr", [1e-5, 1e-4, 1e-3])
        optimizer_i = Adam(learning_rate=lr)

        inference_net = InferenceNet(name='inference', params=self.params, hp=hp)
        inference_learner = Inference(inference_net=inference_net, generator=self.generator, binary=self.params['binary'])

        inference_learner.compile(optimizer_i, loss=self.inference_loss)

        return inference_learner

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice('batch_size', [64, 128, 256, 512]),
            **kwargs,
        )


class Generator(Model):
    def __init__(self, name, params, hp, **kwargs):
        super(Generator, self).__init__(name=name, **kwargs)
        if params['dataset_name'] == 'ihdp_a':
            self.hp_fc = hp.Int('hp_fc', min_value=2, max_value=5, step=1)
            self.hp_hidden_phi = hp.Int('hp_hidden_phi', min_value=4, max_value=48, step=4)
        else:
            self.hp_fc = hp.Int('hp_fc', min_value=2, max_value=10, step=1)
            self.hp_hidden_phi = hp.Int('hp_hidden_phi', min_value=16, max_value=512, step=16)
        self.fc = FullyConnected(n_fc=self.hp_fc, hidden_phi=self.hp_hidden_phi, final_activation='elu',
                                 out_size=self.hp_hidden_phi, kernel_init=params['kernel_init'], kernel_reg=None,
                                 name=name)

        self.pred_y0 = FullyConnected(n_fc=self.hp_fc, hidden_phi=self.hp_hidden_phi,
                                      final_activation=params['activation'], out_size=1,
                                      kernel_init=params['kernel_init'],
                                      kernel_reg=regularizers.l2(params['reg_l2']), name=name+'pred_y0')

        self.pred_y1 = FullyConnected(n_fc=self.hp_fc, hidden_phi=self.hp_hidden_phi,
                                      final_activation=params['activation'], out_size=1,
                                      kernel_init=params['kernel_init'],
                                      kernel_reg=regularizers.l2(params['reg_l2']), name=name+'pred_y1')

    def call(self, inputs):
        x = inputs['x']
        t = inputs['t']
        y = inputs['y']
        inputs = tf.concat(axis=1, values=[x, t, y])

        phi = self.fc(inputs)
        y0_pred = self.pred_y0(phi)
        y1_pred = self.pred_y1(phi)
        concat_pred = tf.concat([y0_pred, y1_pred], axis=-1)

        return concat_pred


class Discriminator(tf.keras.Model):
    def __init__(self, name, params, hp, **kwargs):
        super(Discriminator, self).__init__(name=name, **kwargs)
        self.bias_initializer = tf.keras.initializers.Zeros()
        if params['dataset_name'] == 'ihdp_a':
            self.hp_fc = hp.Int('hp_fc', min_value=2, max_value=5, step=1)
            self.hp_hidden_phi = hp.Int('hp_hidden_phi', min_value=4, max_value=48, step=4)
        else:
            self.hp_fc = hp.Int('hp_fc', min_value=2, max_value=10, step=1)
            self.hp_hidden_phi = hp.Int('hp_hidden_phi', min_value=16, max_value=512, step=16)
        self.fc = FullyConnected(n_fc=self.hp_fc, hidden_phi=self.hp_hidden_phi, out_size=1,
                                 final_activation='linear', name='fcd', kernel_reg=None,
                                 kernel_init=params['kernel_init'],
                                 activation='relu', bias_initializer=self.bias_initializer)

    def call(self, inputs):
        x = inputs['x']
        t = inputs['t']
        y = inputs['y']
        y_tilde = inputs['y_tilde']
        # Concatenate factual & counterfactual outcomes
        input0 = (1. - t) * y + t * y_tilde[:, 0:1]  # if t = 0
        input1 = t * y + (1. - t) * y_tilde[:, 1:2]  # if t = 1

        inputs = tf.concat(axis=1, values=[x, input0, input1])

        out = self.fc(inputs)

        return out


class InferenceNet(tf.keras.Model):
    def __init__(self, name, params, hp, **kwargs):
        super(InferenceNet, self).__init__(name=name, **kwargs)
        self.params = params
        if params['dataset_name'] == 'ihdp_a':
            self.hp_fc = hp.Int('hp_fc', min_value=2, max_value=5, step=1)
            self.hp_hidden_phi = hp.Int('hp_hidden_phi', min_value=4, max_value=48, step=4)
        else:
            self.hp_fc = hp.Int('hp_fc', min_value=2, max_value=10, step=1)
            self.hp_hidden_phi = hp.Int('hp_hidden_phi', min_value=16, max_value=512, step=16)
        self.fc = FullyConnected(n_fc=self.hp_fc, hidden_phi=self.hp_hidden_phi, final_activation='elu',
                                 out_size=self.hp_hidden_phi, kernel_init=params['kernel_init'], kernel_reg=None,
                                 name=name)

        self.pred_y0 = FullyConnected(n_fc=self.hp_fc, hidden_phi=self.hp_hidden_phi,
                                      final_activation=params['activation'], out_size=1,
                                      kernel_init=params['kernel_init'],
                                      kernel_reg=regularizers.l2(params['reg_l2']), name=name + 'pred_y0')

        self.pred_y1 = FullyConnected(n_fc=self.hp_fc, hidden_phi=self.hp_hidden_phi,
                                      final_activation=params['activation'], out_size=1,
                                      kernel_init=params['kernel_init'],
                                      kernel_reg=regularizers.l2(params['reg_l2']), name=name + 'pred_y1')
    @tf.function
    def __call__(self, x):
        y_hidden = self.fc(x)
        y1_out = self.pred_y1(y_hidden)
        y0_out = self.pred_y0(y_hidden)

        out = tf.concat(axis=1, values=[y0_out, y1_out])
        return out


class Inference(Model):
    def __init__(self, inference_net, generator, binary):
        super(Inference, self).__init__()
        self.inference_net = inference_net
        self.generator = generator
        self.binary = binary

    def call(self, inputs):
        x = inputs[:, 2:]
        y = tf.expand_dims(inputs[:, 0], axis=1)  # get individual vectors
        t = tf.expand_dims(inputs[:, 1], axis=1)
        inputs = {'x': x, 't': t, 'y': y}
        if self.binary:
            y_tilde = tf.nn.sigmoid(self.generator(inputs))
        else:
            y_tilde = self.generator(inputs)

        y_hat_logit = self.inference_net(x)
        return tf.concat([y_tilde, y_hat_logit], axis=1)


class GAN(Model):
    def __init__(self, discriminator, generator, alpha, binary=True):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.alpha = alpha
        self.binary = binary

    def compile(self, d_optimizer, g_optimizer):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_metric = tf.keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = tf.keras.metrics.Mean(name="g_loss")
        self.d_loss_metric_val = tf.keras.metrics.Mean(name="d_loss")
        self.g_loss_metric_val = tf.keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, input):
        y = tf.expand_dims(input[:, 0], axis=1)  # get individual vectors
        t = tf.expand_dims(input[:, 1], axis=1)
        x = input[:, 2:]
        inputs = {'x': x, 't': t, 'y': y}
        # Train the discriminator
        for _ in range(2):
            y_tilde = self.generator(inputs)
            if self.binary:
                y_tilde = tf.nn.sigmoid(y_tilde)
            inputs['y_tilde'] = y_tilde
            with tf.GradientTape() as tape:
                d_logit = self.discriminator(inputs)

                # 1. Discriminator loss
                d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=t, logits=d_logit))

            trainable_variables_d = self.discriminator.trainable_weights
            gradients_d = tape.gradient(d_loss, trainable_variables_d)

            self.d_optimizer.apply_gradients(zip(gradients_d, trainable_variables_d))

        # Generator training
        with tf.GradientTape() as tape:
            y_tilde = self.generator(inputs)
            if self.binary:
                y_tilde = tf.nn.sigmoid(y_tilde)
            inputs['y_tilde'] = y_tilde
            d_logit = self.discriminator(inputs)

            # 1. Discriminator loss
            d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=t, logits=d_logit))
            # 2. Generator loss
            g_loss_gan = -d_loss
            if self.binary:
                g_loss_factual = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=y, logits=(t * tf.reshape(y_tilde[:, 1], [-1, 1]) +
                                      (1. - t) * tf.reshape(y_tilde[:, 0], [-1, 1]))))
            else:
                g_loss_factual = tf.reduce_mean(tf.square(y - (t * tf.reshape(y_tilde[:, 1], [-1, 1]) +
                                                               (1. - t) * tf.reshape(y_tilde[:, 0], [-1, 1]))))

            g_loss = g_loss_factual + self.alpha * g_loss_gan

        trainable_variables_g = self.generator.trainable_variables
        gradients_g = tape.gradient(g_loss, trainable_variables_g)
        # Update the weights of the discriminator using the discriminator optimizer
        self.g_optimizer.apply_gradients(zip(gradients_g, trainable_variables_g))

        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
             }

    def test_step(self, input):
        y = tf.expand_dims(input[:, 0], axis=1)  # get individual vectors
        t = tf.expand_dims(input[:, 1], axis=1)
        x = input[:, 2:]
        inputs = {'x': x, 't': t, 'y': y}
        y_tilde = self.generator(inputs)
        if self.binary:
            y_tilde = tf.nn.sigmoid(y_tilde)
        inputs['y_tilde'] = y_tilde
        d_logit = self.discriminator(inputs)
        ## Loss functions
        # Generator training
        y_tilde = self.generator(inputs)
        if self.binary:
            y_tilde = tf.nn.sigmoid(y_tilde)
        inputs['y_tilde'] = y_tilde
        d_logit = self.discriminator(inputs)

        # 1. Discriminator loss
        d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=t, logits=d_logit))
        # 2. Generator loss
        g_loss_gan = -d_loss
        if self.binary:
            g_loss_factual = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=y, logits=(t * tf.reshape(y_tilde[:, 1], [-1, 1]) +
                                  (1. - t) * tf.reshape(y_tilde[:, 0], [-1, 1]))))
        else:
            g_loss_factual = tf.reduce_mean(tf.square(
                y - (t * tf.reshape(y_tilde[:, 1], [-1, 1]) +
                     (1. - t) * tf.reshape(y_tilde[:, 0], [-1, 1]))))

        g_loss = g_loss_factual + self.alpha * g_loss_gan

        # Update metrics
        self.d_loss_metric_val.update_state(d_loss)
        self.g_loss_metric_val.update_state(g_loss)
        return {
            "d_loss": self.d_loss_metric_val.result(),
            "g_loss": self.g_loss_metric_val.result(),
        }


class GANITE(CausalModel):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        # type A or B
        self.dataset = params['dataset_name']
        # number of evaluations
        # Parameters
        self.h_dim = 8
        self.epochs = 10
        self.alpha = 2
        self.binary = params['binary']

    def fit_model(self, x, y, t, seed):
        directory_name = 'params/' + self.params['dataset_name'] + f'/{self.params["model_name"]}'
        setSeed(seed)
        ytx = np.concatenate([y, t, x], 1)

        if self.dataset_name == 'acic':
            directory_name = directory_name + f'/{self.folder_ind}'

        """GAN"""

        tuner_g = kt.RandomSearch(
            HyperGANITE(params=self.params),
            objective=kt.Objective("val_d_loss", direction="min"),
            directory=directory_name,
            tuner_id='1',
            overwrite=False,
            project_name='GAN',
            max_trials=10,
            seed=0)

        stop_early = [TerminateOnNaN(), EarlyStopping(monitor='val_d_loss', patience=5)]
        tuner_g.search(ytx, epochs=50, validation_split=0.2, callbacks=[stop_early], verbose=1)

        best_hps_g = tuner_g.get_best_hyperparameters(num_trials=1)[0]

        gan = tuner_g.hypermodel.build(best_hps_g)

        # print(f"""The hyperparameter search generator is complete.
        #           layer is n_fc={best_hps_g.get('hp_fc')} - hidden_phi = {best_hps_g.get('hp_hidden_phi')} -
        #           learning rate={best_hps_g.get('lr')} - batch size = {best_hps_g.get('batch_size')}""")

        gan.fit(ytx, epochs=self.params['epochs_g'], callbacks=callbacks('val_d_loss'),
                verbose=self.params['verbose'], validation_split=0,
                batch_size=best_hps_g.get('batch_size'))

        """Inference Net"""

        tuner_i = kt.RandomSearch(
            HyperInferenceNet(params=self.params, generator=gan.generator),
            objective=kt.Objective("val_loss", direction="min"),
            directory=directory_name,
            tuner_id='1',
            overwrite=False,
            project_name='Inference',
            max_trials=10,
            seed=0)

        stop_early = [TerminateOnNaN(), EarlyStopping(monitor='val_loss', patience=5)]
        tuner_i.search(ytx, ytx, epochs=50, validation_split=0.2, callbacks=[stop_early], verbose=1)

        best_hps_i = tuner_i.get_best_hyperparameters(num_trials=1)[0]

        inference_learner = tuner_i.hypermodel.build(best_hps_i)

        # print(f"""The hyperparameter search inference is complete.
        #           layer is n_fc={best_hps_i.get('hp_fc')} - hidden_phi = {best_hps_i.get('hp_hidden_phi')} -
        #           learning rate={best_hps_i.get('lr')} - batch size = {best_hps_i.get('batch_size')}""")

        inference_learner.fit(ytx, ytx, epochs=self.params['epochs_i'], callbacks=callbacks('loss'),
                              verbose=self.params['verbose'], validation_split=0.0,
                              batch_size=best_hps_i.get('batch_size'))

        return inference_learner.inference_net

    @staticmethod
    def evaluate(data_test, model):
        return model(data_test['x'])

    @staticmethod
    def compute_PEHE(tau_true, tau_test):
        return np.sqrt(np.mean((tau_true - tau_test) ** 2))

    def train_and_evaluate(self, metric_list, **kwargs):
        data_train, data_test = self.load_data(**kwargs)

        self.folder_ind = kwargs.get('folder_ind')

        if self.params['binary']:
            model = self.fit_model(data_train['x'], data_train['y'], data_train['t'], seed=0)
        else:
            model = self.fit_model(data_train['x'], data_train['ys'], data_train['t'], seed=0)

        concat_pred = self.evaluate(data_test, model)

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
