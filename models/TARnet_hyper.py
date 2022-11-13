from tensorflow.keras import regularizers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD
from models.CausalModel import *
from utils.layers import FullyConnected
import keras_tuner as kt
from utils.callback import callbacks
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN


class HyperTarnet(kt.HyperModel, CausalModel):
    def __init__(self, params):
        super().__init__()
        self.params = params

    def build(self, hp):
        model = TarnetModel(name='tarnet', params=self.params, hp=hp)
        optimizer = SGD(learning_rate=self.params['lr'], momentum=0.9)
        model.compile(optimizer=optimizer,
                      loss=self.regression_loss,
                      metrics=self.regression_loss)
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=self.params['batch_size'],
            **kwargs,
        )


class TarnetModel(Model):
    def __init__(self, name, params, hp, **kwargs):
        super(TarnetModel, self).__init__(name=name, **kwargs)
        self.params = params
        self.hp_n_fc = hp.Int('n_fc', min_value=2, max_value=10, step=1)
        self.hp_n_hidden_phi = hp.Int('hidden_phi', min_value=16, max_value=512, step=16)
        self.fc = FullyConnected(n_fc=self.hp_n_fc, hidden_phi=self.hp_n_hidden_phi, final_activation='elu',
                                 out_size=self.hp_n_hidden_phi, kernel_init=params['kernel_init'], kernel_reg=None,
                                 name='fc')
        self.hp_n_hidden_0 = hp.Int('n_fc_y0', min_value=2, max_value=10, step=1)
        self.hp_hidden_y0 = hp.Int('hidden_y0', min_value=16, max_value=512, step=16)
        self.pred_y0 = FullyConnected(n_fc=self.hp_n_hidden_0, hidden_phi=self.hp_hidden_y0,
                                      final_activation=params['activation'], out_size=1,
                                      kernel_init=params['kernel_init'],
                                      kernel_reg=regularizers.l2(params['reg_l2']), name='y0')

        self.hp_n_hidden_1 = hp.Int('n_fc_y1', min_value=2, max_value=10, step=1)
        self.hp_hidden_y1 = hp.Int('hidden_y1', min_value=16, max_value=512, step=16)
        self.pred_y1 = FullyConnected(n_fc=self.hp_n_hidden_1, hidden_phi=self.hp_hidden_y1,
                                      final_activation=params['activation'], out_size=1,
                                      kernel_init=params['kernel_init'],
                                      kernel_reg=regularizers.l2(params['reg_l2']), name='y1')

    def call(self, inputs):
        x = self.fc(inputs)
        y0_pred = self.pred_y0(x)
        y1_pred = self.pred_y1(x)
        concat_pred = tf.concat([y0_pred, y1_pred], axis=-1)
        return concat_pred


class TARnet(CausalModel):

    def __init__(self, params):
        super().__init__(params)
        self.params = params

    def fit_model(self, x, y, t, count, seed):
        directory_name = 'params/' + self.params['dataset_name']
        setSeed(seed)
        t = tf.cast(t, dtype=tf.float32)

        if self.dataset_name == 'acic':
            directory_name = directory_name + f'/{self.params["model_name"]}'
            project_name = str(self.folder_ind)
        else:
            project_name = self.params["model_name"]

        tuner = kt.RandomSearch(
            HyperTarnet(params=self.params),
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

        if self.params['defaults']:
            best_hps.values = {'n_fc': self.params['n_fc'], 'hidden_phi': self.params['hidden_phi'],
                               'n_fc_y0': self.params['n_fc_y0'], 'n_fc_y1': self.params['n_fc_y1'],
                               'hidden_y1': self.params['hidden_y1'], 'hidden_y0': self.params['hidden_y0']}

        model = tuner.hypermodel.build(best_hps)

        model.fit(x=x, y=yt,
                  callbacks=callbacks('regression_loss'),
                  validation_split=0.0,
                  epochs=self.params['epochs'],
                  batch_size=self.params['batch_size'],
                  verbose=self.params['verbose'])

        if count == 0:
            print(f"""The hyperparameter search is complete. the optimal hyperparameters are
                  layer is n_fc={best_hps.get('n_fc')} hidden_phi = {best_hps.get('hidden_phi')}
                  hidden_y1 = {best_hps.get('hidden_y1')} n_fc_y1 = {best_hps.get('n_fc_y1')}
                  hidden_y0 = {best_hps.get('hidden_y0')}  n_fc_y0 = {best_hps.get('n_fc_y0')}""")
            print(model.summary())

        return model

    @staticmethod
    def evaluate(x_test, model):
        return model.predict(x_test)

    def train_and_evaluate(self, metric_list, **kwargs):
        data_train, data_test = self.load_data(**kwargs)

        count = kwargs.get('count')

        if self.params['binary']:
            model = self.fit_model(data_train['x'], data_train['y'], data_train['t'], count, seed=0)
        else:
            model = self.fit_model(data_train['x'], data_train['ys'], data_train['t'], count, seed=0)

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
