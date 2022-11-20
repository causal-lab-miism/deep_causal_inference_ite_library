from models.CausalModel import *
from tensorflow.keras import Model
from tensorflow.keras import regularizers
from utils.layers import FullyConnected
from utils.callback import callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN


import keras_tuner as kt


class HyperMuLearner(kt.HyperModel, CausalModel):
    def __init__(self, params):
        super().__init__()
        self.params = params

    def build(self, hp):
        mumodel = MuModel(name='mulearner', params=self.params, hp=hp)
        lr = hp.Choice("lr", [1e-4, 1e-3, 1e-2])
        mumodel.compile(loss='mse', optimizer=Adam(learning_rate=lr), metrics=['mse'])

        return mumodel

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice('batch_size', [32, 64, 128, 256, 512]),
            **kwargs,
        )


class HyperGLearner(kt.HyperModel, CausalModel):
    def __init__(self, params):
        super().__init__()
        self.params = params

    def build(self, hp):
        gmodel = GModel(name='glearner', params=self.params, hp=hp)
        lr = hp.Choice("lr", [1e-4, 1e-3, 1e-2])
        gmodel.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=lr))

        return gmodel

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice('batch_size', [32, 64, 128, 256, 512]),
            **kwargs,
        )


class HyperRLearner(kt.HyperModel, CausalModel):
    def __init__(self, mu_model, g_model, params):
        super().__init__()
        self.mu_model = mu_model
        self.g_model = g_model
        self.params = params

    @staticmethod
    def r_loss(concat_true, concat_pred):
        # source: https://github.com/kochbj/Deep-Learning-for-Causal-Inference

        y_pred = concat_pred[:, 0]
        tau_pred = concat_pred[:, 1]
        g_pred = concat_pred[:, 2]

        y_true = tf.cast(concat_true[:, 0], tf.float32)  # get individual vectors
        t_true = tf.cast(concat_true[:, 1], tf.float32)

        loss = tf.reduce_mean(tf.square((y_true - y_pred) - (t_true - g_pred) * tau_pred))

        return loss

    def build(self, hp):
        rmodel = RModel(self.mu_model, self.g_model, params=self.params, hp=hp)
        lr = hp.Choice("lr", [1e-3, 1e-2])
        rmodel.compile(Adam(learning_rate=lr), loss=self.r_loss)
        return rmodel

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice('batch_size', [32, 64, 128, 256, 512]),
            **kwargs,
        )


class MuModel(Model):
    def __init__(self, params, hp, name='mu_model', **kwargs):
        super(MuModel, self).__init__(name=name, **kwargs)
        self.params = params
        self.n_fc = hp.Int('n_fc', min_value=2, max_value=10, step=1)
        self.hidden_phi = hp.Int('hidden_phi', min_value=16, max_value=512, step=16)
        self.fc = FullyConnected(n_fc=self.n_fc, hidden_phi=self.hidden_phi,
                                 final_activation=params['activation'], out_size=1, activation='elu',
                                 kernel_init=params['kernel_init'],
                                 kernel_reg=regularizers.l2(params['reg_l2']), name='fc')

    def call(self, inputs):
        return self.fc(inputs)


class GModel(Model):
    def __init__(self, params, hp, name='g_model', **kwargs):
        super(GModel, self).__init__(name=name, **kwargs)
        self.params = params
        self.n_fc = hp.Int('n_fc', min_value=2, max_value=10, step=1)
        self.hidden_phi = hp.Int('hidden_phi', min_value=8, max_value=24, step=2)
        self.fc = FullyConnected(n_fc=self.n_fc, hidden_phi=self.hidden_phi,
                                 final_activation=params['activation'], out_size=1, activation='elu',
                                 kernel_init=params['kernel_init'],
                                 kernel_reg=regularizers.l2(params['reg_l2']), name='fc')

    def call(self, inputs):
        out = self.fc(inputs)
        return out


class TauModel(Model):
    def __init__(self, params, hp, name='tau_model', **kwargs):
        super(TauModel, self).__init__(name=name, **kwargs)
        self.params = params
        self.n_fc = hp.Int('n_fc', min_value=2, max_value=10, step=1)
        self.hidden_phi = hp.Int('hidden_phi', min_value=16, max_value=512, step=16)
        self.fc = FullyConnected(n_fc=self.n_fc, hidden_phi=self.hidden_phi,
                                 final_activation=self.params['activation'], out_size=1,
                                 kernel_init=params['kernel_init'], kernel_reg=regularizers.l2(self.params['reg_l2']),
                                 name='tau_model')

    def call(self, inputs):
        return self.fc(inputs)


class RModel(Model):
    def __init__(self, mu_model, g_model, params, hp, name="rlearner", **kwargs):
        super(RModel, self).__init__(name=name, **kwargs)
        self.params = params
        self.mu = mu_model
        self.tau = TauModel(name='taumodel', params=self.params, hp=hp)
        self.g = g_model

    def call(self, inputs):
        y = self.mu(inputs)
        tau = self.tau(inputs)
        g = self.g(inputs)
        return tf.concat([y, tau, g], axis=1)


class RLearner(CausalModel):
    def __init__(self, params):
        super().__init__(params)
        self.params = params

    @staticmethod
    def r_loss(concat_true, concat_pred):
        # source: https://github.com/kochbj/Deep-Learning-for-Causal-Inference

        y_pred = concat_pred[:, 0]
        tau_pred = concat_pred[:, 1]
        g_pred = concat_pred[:, 2]

        y_true = tf.cast(concat_true[:, 0], tf.float32)  # get individual vectors
        t_true = tf.cast(concat_true[:, 1], tf.float32)

        loss = tf.reduce_mean(tf.square((y_true - y_pred) - (t_true - g_pred) * tau_pred))

        return loss

    def fit_model(self, x, y, t, seed):
        directory_name = 'params/' + self.params['dataset_name'] + f'/{self.params["model_name"]}'
        setSeed(seed)

        if self.dataset_name == 'acic':
            directory_name = directory_name + f'/{self.folder_ind}'

        """Mu_Model"""

        tuner_mu = kt.RandomSearch(
            HyperMuLearner(params=self.params),
            objective=kt.Objective('val_loss', direction='min'),
            directory=directory_name,
            tuner_id='1',
            overwrite=False,
            project_name='MuModel',
            max_trials=self.params['max_trials'],
            seed=0)
        
        stop_early = [TerminateOnNaN(), EarlyStopping(monitor='val_mse', patience=5)]

        tuner_mu.search(x, y, epochs=50, validation_split=0.2, callbacks=[stop_early], verbose=1)

        best_hps_mu = tuner_mu.get_best_hyperparameters(num_trials=1)[0]

        model_mu = tuner_mu.hypermodel.build(best_hps_mu)

        print(f"""The hyperparameter search for Model_Mu is complete. the optimal hyperparameters are
              layer is n_fc={best_hps_mu.get('n_fc')} - hidden_phi = {best_hps_mu.get('hidden_phi')} -
              batch size = {best_hps_mu.get('batch_size')}""")

        model_mu.fit(x, y, epochs=self.params['epochs'], callbacks=callbacks('loss'),
                     batch_size=best_hps_mu.get('batch_size'), verbose=0, validation_split=0.0)

        """G Model"""
        
        tuner_g = kt.RandomSearch(
            HyperGLearner(params=self.params),
            objective=kt.Objective('val_loss', direction='min'),
            directory=directory_name,
            tuner_id='1',
            overwrite=False,
            project_name='GModel',
            max_trials=self.params['max_trials'],
            seed=0)

        stop_early = [TerminateOnNaN(), EarlyStopping(monitor='val_loss', patience=5)]

        tuner_g.search(x, t, epochs=50, validation_split=0.2, callbacks=[stop_early], verbose=1)

        best_hps_g = tuner_g.get_best_hyperparameters(num_trials=1)[0]

        model_g = tuner_mu.hypermodel.build(best_hps_g)

        print(f"""The hyperparameter search for Model_G is complete. the optimal hyperparameters are
              layer is n_fc={best_hps_g.get('n_fc')} - hidden_phi = {best_hps_g.get('hidden_phi')} -
              - batch size = {best_hps_g.get('batch_size')}""")

        model_g.fit(x, t, epochs=self.params['epochs'], callbacks=callbacks('loss'),
                    batch_size=best_hps_g.get('batch_size'), verbose=0, validation_split=0.0)

        """R Model"""

        tuner_r = kt.RandomSearch(
            HyperRLearner(mu_model=model_mu, g_model=model_g, params=self.params),
            objective=kt.Objective('val_loss', direction='min'),
            directory=directory_name + '/RLearner/',
            tuner_id='1',
            overwrite=False,
            project_name='RModel',
            max_trials=self.params['max_trials'],
            seed=0)

        yt = np.concatenate([y, t], axis=1)

        stop_early = [TerminateOnNaN(), EarlyStopping(monitor='val_loss', patience=5)]

        tuner_r.search(x, yt, epochs=50, validation_split=0.2, callbacks=[stop_early], verbose=1)

        best_hps_r = tuner_r.get_best_hyperparameters(num_trials=1)[0]

        model_r = tuner_r.hypermodel.build(best_hps_r)

        print(f"""The hyperparameter search for Model_R is complete. the optimal hyperparameters are
              layer is n_fc={best_hps_r.get('n_fc')} - hidden_phi = {best_hps_r.get('hidden_phi')} -
              - batch size = {best_hps_r.get('batch_size')}""")

        model_r.fit(x, yt, epochs=self.params['epochs'], callbacks=callbacks('loss'),
                    batch_size=best_hps_r.get('batch_size'), verbose=0, validation_split=0.0)

        return model_r

    @staticmethod
    def evaluate(x_test, model):
        return model.predict(x_test)

    @staticmethod
    def find_pehe(cate_pred, data):
        pred = cate_pred[:, 1:2]
        cate_true = (data['mu_1'] - data['mu_0']).squeeze()
        cate_pred = pred.squeeze()
        pehe = np.mean(np.square((cate_true - cate_pred)))
        sqrt_pehe = np.sqrt(pehe)

        return sqrt_pehe

    def find_policy_risk(self, cate_pred, data):
        cate_pred = cate_pred[:, 1:2]
        cate_true = data['tau']
        policy_value, policy_curve = self.policy_val(data['t'][cate_true > 0], data['y'][cate_true > 0],
                                                     cate_pred[cate_true > 0], False)
        policy_risk = 1 - policy_value

        return policy_value, policy_risk, policy_curve

    def train_and_evaluate(self, metric_list, **kwargs):
        data_train, data_test = self.load_data(**kwargs)

        self.folder_ind = kwargs.get('folder_ind')

        model = self.fit_model(data_train['x'], data_train['y'], data_train['t'], seed=0)

        cate_pred = self.evaluate(data_test['x'], model)

        if self.params['dataset_name'] == 'jobs':
            policy_value, policy_risk, policy_curve = self.find_policy_risk(cate_pred, data_test)
            print(kwargs.get('count'), 'Policy Risk Test = ', policy_risk)
            metric_list.append(policy_risk)
        else:
            pehe = self.find_pehe(cate_pred, data_test)
            if self.params['dataset_name'] == 'acic':
                print(kwargs.get('folder_ind'), kwargs.get('file_ind'), 'Pehe Test = ', pehe)
            else:
                print(kwargs.get('count'), 'Pehe Test = ', pehe)
            metric_list.append(pehe)

