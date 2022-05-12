from models.TLearner import *
from models.TARnet import *
from keras import regularizers
from utils.set_seed import setSeed
import keras_tuner as kt
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN


class HyperELearner(kt.HyperModel, CausalModel):
    def __init__(self, params):
        super().__init__()
        self.params = params

    def build(self, hp):
        model = EModel(name='emodel', params=self.params, hp=hp)
        optimizer = Adam(learning_rate=self.params['lr'])
        model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])

        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice('batch_size', [32, 64, 128, 256, 512, 1024]),
            **kwargs,
        )


class HyperGLearner(kt.HyperModel, CausalModel):
    def __init__(self, params):
        super().__init__()
        self.params = params

    def build(self, hp):
        model = GModel(name='gmodel', params=self.params, hp=hp)
        optimizer = Adam(learning_rate=self.params['lr'])
        model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])

        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice('batch_size', [32, 64, 128, 256, 512, 1024]),
            **kwargs,
        )


class GModel(Model):
    def __init__(self, params, hp, name='g_model', **kwargs):
        super(GModel, self).__init__(name=name, **kwargs)
        self.params = params
        self.hp_fc = hp.Int('hp_fc', min_value=2, max_value=5, step=1)
        self.hp_hidden_phi = hp.Int('hp_hidden_phi', min_value=16, max_value=64, step=8)
        self.fc = FullyConnected(n_fc=self.hp_fc, hidden_phi=self.hp_hidden_phi,
                                 final_activation='sigmoid', out_size=1, activation='elu',
                                 kernel_init=params['kernel_init'],
                                 kernel_reg=regularizers.l2(params['reg_l2']), name='fc')

    def call(self, inputs):
        out = self.fc(inputs)
        return out


class EModel(Model):
    def __init__(self, params, hp, name='e_model', **kwargs):
        super(EModel, self).__init__(name=name, **kwargs)
        self.params = params
        self.hp_fc = hp.Int('hp_fc', min_value=2, max_value=8, step=1)
        self.hp_hidden_phi = hp.Int('hp_hidden_phi', min_value=16, max_value=256, step=16)
        self.fc = FullyConnected(n_fc=self.hp_fc, hidden_phi=self.hp_hidden_phi,
                                 final_activation=params['activation'], out_size=1, activation='elu',
                                 kernel_init=params['kernel_init'],
                                 kernel_reg=regularizers.l2(params['reg_l2']), name='fc')

    def call(self, inputs):
        out = self.fc(inputs)
        return out


class XLearner(CausalModel):
    def __init__(self, params):
        super().__init__(params)
        self.params = params

    def fit_model(self, x, y, t, seed):
        directory_name = 'params/' + self.params['dataset_name'] + f'/{self.params["model_name"]}'
        setSeed(seed)

        t0_ind = np.squeeze(t == 0)
        t1_ind = np.squeeze(t == 1)

        y1 = y[t1_ind]
        y0 = y[t0_ind]

        x0 = x[t0_ind]
        x1 = x[t1_ind]

        """
        Stage 1.
        Estimate the average outcomes mu_0 and mu_1
        """

        if self.dataset_name == 'acic':
            directory_name = directory_name + f'/{self.folder_ind}'

        stop_early = [TerminateOnNaN(), EarlyStopping(monitor='val_mse', patience=5)]

        tuner_mu0 = kt.RandomSearch(
            HyperELearner(params=self.params),
            objective=kt.Objective("val_mse", direction="min"),
            directory=directory_name,
            tuner_id='1',
            overwrite=False,
            project_name='MuModel0',
            max_trials=10,
            seed=0)

        tuner_mu0.search(x0, y0, epochs=50, validation_split=0.2, callbacks=[stop_early], verbose=1)
        best_hps_mu0 = tuner_mu0.get_best_hyperparameters(num_trials=1)[0]

        print(f"""The hyperparameter search is complete. the optimal hyperparameters are
              number of layers ={best_hps_mu0.get('hp_fc')} - hidden_phi = {best_hps_mu0.get('hp_hidden_phi')}""")

        tuner_mu1 = kt.RandomSearch(
            HyperELearner(params=self.params),
            objective=kt.Objective("val_mse", direction="min"),
            directory=directory_name,
            tuner_id='1',
            overwrite=False,
            project_name='MuModel1',
            max_trials=10,
            seed=0)

        tuner_mu1.search(x1, y1, epochs=50, validation_split=0.2, callbacks=[stop_early], verbose=1)
        best_hps_mu1 = tuner_mu1.get_best_hyperparameters(num_trials=1)[0]

        print(f"""The hyperparameter search is complete. the optimal hyperparameters are
              number of layers ={best_hps_mu1.get('hp_fc')} - hidden_phi = {best_hps_mu1.get('hp_hidden_phi')}""")

        model_mu0 = tuner_mu0.hypermodel.build(best_hps_mu0)
        model_mu1 = tuner_mu0.hypermodel.build(best_hps_mu1)

        model_mu0.fit(x0, y0, epochs=self.params['epochs'], callbacks=callbacks('mse'),
                      batch_size=best_hps_mu0.get('batch_size'), validation_split=0.0,
                      verbose=self.params['verbose'])

        model_mu1.fit(x1, y1, epochs=self.params['epochs'], callbacks=callbacks('mse'),
                      batch_size=best_hps_mu1.get('batch_size'), validation_split=0.0,
                      verbose=self.params['verbose'])

        """
        Stage 2.
        Impute the user level treatment effects d1 and d0 for user i in the treatment group based on mu0 and user
        j in control group based on mu1
        """

        d1 = y1 - model_mu0.predict(x1)
        d0 = model_mu1.predict(x0) - y0

        stop_early = [TerminateOnNaN(), EarlyStopping(monitor='val_mse', patience=5)]


        tuner_d0 = kt.RandomSearch(
            HyperELearner(params=self.params),
            objective=kt.Objective("val_mse", direction="min"),
            directory=directory_name,
            tuner_id='1',
            overwrite=False,
            project_name='DModel0',
            max_trials=10,
            seed=0)

        tuner_d0.search(x0, d0, epochs=50, validation_split=0.2, callbacks=[stop_early], verbose=1)
        best_hps_d0 = tuner_d0.get_best_hyperparameters(num_trials=1)[0]

        print(f"""The hyperparameter search is complete. the optimal hyperparameters are
                      number of layers ={best_hps_d0.get('hp_fc')} - hidden_phi = {best_hps_d0.get('hp_hidden_phi')}""")

        tuner_d1 = kt.RandomSearch(
            HyperELearner(params=self.params),
            objective=kt.Objective("val_mse", direction="min"),
            directory=directory_name,
            tuner_id='1',
            overwrite=False,
            project_name='DModel1',
            max_trials=10,
            seed=0)

        tuner_d1.search(x0, d0, epochs=50, validation_split=0.2, callbacks=[stop_early], verbose=1)
        best_hps_d1 = tuner_d1.get_best_hyperparameters(num_trials=1)[0]

        print(f"""The hyperparameter search is complete. the optimal hyperparameters are
                      number of layers ={best_hps_d1.get('hp_fc')} - hidden_phi = {best_hps_d1.get('hp_hidden_phi')}""")

        model_du0 = tuner_d0.hypermodel.build(best_hps_d0)
        model_du1 = tuner_d1.hypermodel.build(best_hps_d1)

        model_du0.fit(x0, d0, epochs=self.params['epochs'], callbacks=callbacks('mse'),
                      batch_size=best_hps_d0.get('batch_size'), validation_split=0.0,
                      verbose=self.params['verbose'])

        model_du1.fit(x1, d1, epochs=self.params['epochs'], callbacks=callbacks('mse'),
                      batch_size=best_hps_d1.get('batch_size'), validation_split=0.0,
                      verbose=self.params['verbose'])

        tuner_g = kt.RandomSearch(
            HyperGLearner(params=self.params),
            objective=kt.Objective("val_mse", direction="min"),
            directory=directory_name,
            tuner_id='1',
            overwrite=False,
            project_name='GModel',
            max_trials=10,
            seed=0)

        stop_early = [TerminateOnNaN(), EarlyStopping(monitor='val_mse', patience=5)]
        tuner_g.search(x1, d1, epochs=50, validation_split=0.2, callbacks=[stop_early], verbose=1)

        best_hps_g = tuner_g.get_best_hyperparameters(num_trials=1)[0]

        print(f"""The hyperparameter search is complete. the optimal hyperparameters are
              number of layers={best_hps_g.get('hp_fc')} - hidden_phi = {best_hps_g.get('hp_hidden_phi')}""")

        model_g = tuner_g.hypermodel.build(best_hps_g)

        model_g.fit(x, t, epochs=self.params['epochs'], callbacks=callbacks('mse'),
                    batch_size=best_hps_g.get('batch_size'), validation_split=0.0, verbose=self.params['verbose'])

        return [model_du0, model_du1, model_g]

    @staticmethod
    def evaluate(x_test, models):
        tau0 = models[0]
        tau1 = models[1]
        g = models[2]
        return g(x_test) * tau0(x_test) + (1 - g(x_test)) * tau1(x_test)

    @staticmethod
    def find_pehe(cate_pred, data):
        cate_true = (data['mu_1'] - data['mu_0']).squeeze()
        cate_pred = (cate_pred.numpy()).squeeze()
        pehe = np.mean(np.square((cate_true - cate_pred)))
        sqrt_pehe = np.sqrt(pehe)

        return sqrt_pehe

    def find_policy_risk(self, cate_pred, data):
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
