from tensorflow.keras import Model
from models.CausalModel import *
from utils.layers import FullyConnected
from utils.callback import callbacks
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN


class HyperTLearner(kt.HyperModel, CausalModel):
    def __init__(self, params):
        super().__init__()
        self.params = params

    def build(self, hp):
        model = TModel(name='tlearner', params=self.params, hp=hp)
        lr = hp.Choice("lr", [1e-3, 1e-2, 1e-4])
        optimizer = Adam(learning_rate=lr)
        model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])

        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice('batch_size', [64, 128, 256, 512]),
            **kwargs,
        )


class TModel(Model):
    def __init__(self, params, hp, name='tlearner', **kwargs):
        super(TModel, self).__init__(name=name, **kwargs)
        self.params = params
        self.n_fc = hp.Int('n_fc', min_value=2, max_value=10, step=1)
        self.hidden_phi = hp.Int('hidden_phi', min_value=16, max_value=512, step=16)
        self.fc = FullyConnected(n_fc=self.n_fc, hidden_phi=self.n_fc,
                                 final_activation=params['activation'], out_size=1,
                                 kernel_init=params['kernel_init'],
                                 kernel_reg=regularizers.l2(params['reg_l2']), name='fc')

    def call(self, inputs):
        # for reproducibility
        x = self.fc(inputs)
        return x


class TLearner(CausalModel):
    """
    This class can be used to train and create stacked model
    for IHDP dataset setting "b"
    """

    def __init__(self, params):
        super().__init__(params)
        self.params = params

    def fit_model(self, x, y, t, seed):
        directory_name = 'params/' + self.params['dataset_name'] + f'/{self.params["model_name"]}'
        setSeed(seed)

        if self.dataset_name == 'acic':
            directory_name = directory_name + f'/{self.folder_ind}'

        tuner_1 = kt.RandomSearch(
            HyperTLearner(params=self.params),
            objective=kt.Objective("val_mse", direction="min"),
            directory=directory_name,
            tuner_id='1',
            overwrite=False,
            project_name='Model1',
            max_trials=self.params['max_trials'],
            seed=0)

        tuner_0 = kt.RandomSearch(
            HyperTLearner(params=self.params),
            objective=kt.Objective("val_mse", direction="min"),
            directory=directory_name,
            tuner_id='2',
            overwrite=False,
            project_name='Model0',
            max_trials=self.params['max_trials'],
            seed=0)

        t0_ind = np.squeeze(t == 0)
        t1_ind = np.squeeze(t == 1)

        x0 = x[t0_ind]
        x1 = x[t1_ind]

        y0 = y[t0_ind]
        y1 = y[t1_ind]
        setSeed(seed)

        stop_early = [TerminateOnNaN(), EarlyStopping(monitor='val_mse', patience=5)]
        tuner_0.search(x0, y0, epochs=50, validation_split=0.2, callbacks=[stop_early], verbose=1)
        tuner_1.search(x1, y1, epochs=50, validation_split=0.2, callbacks=[stop_early], verbose=1)

        # Get the optimal hyperparameters
        best_hps_0 = tuner_0.get_best_hyperparameters(num_trials=10)[0]
        best_hps_1 = tuner_1.get_best_hyperparameters(num_trials=10)[0]

        # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
        model0 = tuner_0.hypermodel.build(best_hps_0)
        model1 = tuner_1.hypermodel.build(best_hps_1)

        # if seed == 0:
        #     print(f"""The hyperparameter search is complete. the optimal hyperparameters are
        #           layer0 is hp_fc_0={best_hps_0.get('n_fc')} - hp_hidden_phi_0 = {best_hps_0.get('hidden_phi')} -
        #           learning rate={best_hps_0.get('lr')} - batch size = {best_hps_0.get('batch_size')} """)
        #
        #     print(f"""The hyperparameter search is complete. the optimal hyperparameters are
        #           layer1 is hp_fc_1={best_hps_1.get('n_fc')} - hp_hidden_phi_1 = {best_hps_1.get('hidden_phi')} -
        #           learning rate={best_hps_1.get('lr')} - batch size = {best_hps_1.get('batch_size')} """)

        model0.fit(x0, y0, epochs=self.params['epochs'], callbacks=callbacks('loss'),
                   batch_size=best_hps_0.get('batch_size'), validation_split=0.0,
                   verbose=self.params['verbose'])

        model1.fit(x1, y1, epochs=self.params['epochs'], callbacks=callbacks('loss'),
                   batch_size=best_hps_1.get('batch_size'), validation_split=0.0,
                   verbose=self.params['verbose'])

        return [model0, model1]

    @staticmethod
    def evaluate(x_test, models):
        concat_pred = list()
        for model in models:
            y_pred = model.predict(x_test)
            concat_pred.append(y_pred)
        return tf.concat((concat_pred[0], concat_pred[1]), axis=1)

    def train_and_evaluate(self, metric_list, **kwargs):
        data_train, data_test = self.load_data(**kwargs)

        self.folder_ind = kwargs.get('folder_ind')

        if self.params['binary']:
            model = self.fit_model(data_train['x'], data_train['y'], data_train['t'], seed=kwargs.get('count'))
        else:
            model = self.fit_model(data_train['x'], data_train['ys'], data_train['t'], seed=kwargs.get('count'))

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
