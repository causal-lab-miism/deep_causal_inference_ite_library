from keras import Model
from models.CausalModel import *
from utils.layers import FullyConnected
from utils.callback import callbacks
from utils.set_seed import setSeed
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class HyperSLearner(kt.HyperModel, CausalModel):
    def __init__(self, params):
        super().__init__()
        self.params = params

    def build(self, hp):
        model = SModel(name='slearner', params=self.params, hp=hp)
        optimizer = Adam(learning_rate=self.params['lr'])
        model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])

        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=self.params['batch_size'],
            **kwargs,
        )


class SModel(Model):
    def __init__(self, params, hp, name='slearner', **kwargs):
        super(SModel, self).__init__(name=name, **kwargs)
        self.params = params
        self.n_fc = hp.Int('n_fc', min_value=2, max_value=10, step=1)
        self.hidden_phi = hp.Int('hidden_phi', min_value=16, max_value=512, step=16)
        self.fc = FullyConnected(n_fc=self.n_fc, hidden_phi=self.hidden_phi,
                                 final_activation=params['activation'], out_size=1,
                                 kernel_init=params['kernel_init'], kernel_reg=None, name='fc')

    def call(self, inputs):
        return self.fc(inputs)


class SLearner(CausalModel):
    """
    This class can be used to train and create stacked model
    for IHDP dataset setting "b"
    """

    def __init__(self, params):
        super().__init__(params)
        self.params = params

    def fit_model(self, x, y, t, seed, count):
        directory_name = 'params/' + self.params['dataset_name']
        setSeed(seed)

        if self.dataset_name == 'acic':
            directory_name = directory_name + f'/{self.params["model_name"]}'
            project_name = str(self.folder_ind)
        else:
            project_name = self.params["model_name"]

        tuner = kt.RandomSearch(
            HyperSLearner(params=self.params),
            objective=kt.Objective("val_mse", direction="min"),
            directory=directory_name,
            tuner_id='1',
            overwrite=False,
            project_name=project_name,
            max_trials=self.params['max_trials'],
            seed=0)

        x_t = tf.concat([x, t], axis=1)

        stop_early = [TerminateOnNaN(), EarlyStopping(monitor='val_mse', patience=5)]
        tuner.search(x_t, y, epochs=50, validation_split=0.2, callbacks=[stop_early], verbose=1)

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        if self.params['defaults']:
            best_hps.values = {'n_fc': self.params['n_fc'], 'hidden_phi': self.params['hidden_phi']}

        model = tuner.hypermodel.build(best_hps)

        model.fit(x_t, y, epochs=self.params['epochs'], callbacks=callbacks('mse'),
                  batch_size=self.params['batch_size'], validation_split=0.0,
                  verbose=self.params['verbose'])
        if count == 0:
            print(f"""The hyperparameter search is complete. the optimal hyperparameters are
                  layer is n_fc={best_hps.get('n_fc')} - hidden_phi = {best_hps.get('hidden_phi')}""")
            print(model.summary())

        return model

    @staticmethod
    def evaluate(x_test, model):
        x_t0 = tf.concat([x_test, tf.zeros([x_test.shape[0], 1], dtype=tf.float64)], axis=1)
        x_t1 = tf.concat([x_test, tf.ones([x_test.shape[0], 1], dtype=tf.float64)], axis=1)

        out0 = model(x_t0)
        out1 = model(x_t1)
        return tf.concat((out0, out1), axis=1)

    def train_and_evaluate(self, metric_list, **kwargs):
        data_train, data_test = self.load_data(**kwargs)

        self.folder_ind = kwargs.get('folder_ind')
        count = kwargs.get('count')
        if self.params['binary']:
            model = self.fit_model(data_train['x'], data_train['y'], data_train['t'], seed=0, count=count)
        else:
            model = self.fit_model(data_train['x'], data_train['ys'], data_train['t'], seed=0, count=count)

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

