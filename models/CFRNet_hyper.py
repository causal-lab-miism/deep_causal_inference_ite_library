from models.CausalModel import *
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import regularizers
from tensorflow.keras import Model
from tensorflow.keras.losses import Loss
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow_probability as tfp
from utils.callback import callbacks
from tensorflow.keras.metrics import binary_accuracy
from utils.layers import FullyConnected
from utils.custom_dataset import DataGen
import keras_tuner as kt
import warnings
tfd = tfp.distributions

warnings.filterwarnings('ignore')


class HyperCFRNet(kt.HyperModel, CausalModel):
    def __init__(self, params, pT):
        super().__init__()
        self.params = params
        self.pT = pT

    def build(self, hp):
        optimizer = Adam(learning_rate=self.params['lr'])
        if self.params['ipm_type'] == "weighted":
            model = Weighted_CFR(name='weighted_cfrnet', params=self.params, hp=hp)

            # print("Probability of treament:", pT)
            loss = Weighted_CFRNet_Loss(prob_treat=self.pT, alpha=0.1)

            model.compile(optimizer=optimizer, loss=loss,
                          metrics=[loss, loss.weights, loss.treatment_acc, loss.weighted_mmdsq_loss])
        else:
            model = CFRModel(name='cfrnet', params=self.params, hp=hp)

            loss = CFRNet_Loss(alpha=0.1, ipm=self.params['ipm_type'])
            if self.params['ipm_type'] == 'wasserstein':
                model.compile(optimizer=optimizer, loss=loss, metrics=[loss, loss.regression_loss, loss.wasserstein])
            elif self.params['ipm_type'] == 'mmdsq':
                model.compile(optimizer=optimizer, loss=loss, metrics=[loss, loss.regression_loss, loss.mmdsq_loss])
            else:
                raise ValueError(f'IPM type {self.params["ipm_type"]} not implemented yet.')

        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=self.params['tuner_batch_size'],
            **kwargs,
        )


class Base_Metrics(Callback):
    def __init__(self, data, verbose=0):
        super(Base_Metrics, self).__init__()
        self.data = data  # feed the callback the full dataset
        self.verbose = verbose

        # needed for PEHEnn; Called in self.find_ynn
        self.data['o_idx'] = tf.range(self.data['t'].shape[0])
        self.data['c_idx'] = self.data['o_idx'][
            self.data['t'].squeeze() == 0]  # These are the indices of the control units
        self.data['t_idx'] = self.data['o_idx'][
            self.data['t'].squeeze() == 1]  # These are the indices of the treated units

    def split_pred(self, concat_pred):
        preds = {}
        preds['y0_pred'] = self.data['y_scaler'].inverse_transform(concat_pred[:, 0].reshape(-1, 1))
        preds['y1_pred'] = self.data['y_scaler'].inverse_transform(concat_pred[:, 1].reshape(-1, 1))
        preds['phi'] = concat_pred[:, 2:]
        return preds

    def find_ynn(self, Phi):
        # helper for PEHEnn
        PhiC, PhiT = tf.dynamic_partition(Phi, tf.cast(tf.squeeze(self.data['t']), tf.int32),
                                          2)  # separate control and treated reps
        dists = tf.sqrt(CFRNet_Loss.pdist2sq(PhiC, PhiT))  # calculate squared distance then sqrt to get euclidean
        yT_nn_idx = tf.gather(self.data['c_idx'], tf.argmin(dists, axis=0),
                              1)  # get c_idxs of the smallest distances for treated units
        yC_nn_idx = tf.gather(self.data['t_idx'], tf.argmin(dists, axis=1),
                              1)  # get t_idxs of the smallest distances for control units
        yT_nn = tf.gather(self.data['y'], yT_nn_idx, 1)  # now use these to retrieve y values
        yC_nn = tf.gather(self.data['y'], yC_nn_idx, 1)
        y_nn = tf.dynamic_stitch([self.data['t_idx'], self.data['c_idx']], [yT_nn, yC_nn])  # stitch em back up!
        return y_nn

    def PEHEnn(self, concat_pred):
        p = self.split_pred(concat_pred)
        y_nn = self.find_ynn(p['phi'])  # now its 3 plus because
        cate_nn_err = tf.reduce_mean(
            tf.square((1 - 2 * self.data['t']) * (y_nn - self.data['y']) - (p['y1_pred'] - p['y0_pred'])))
        return cate_nn_err

    def ATE(self, concat_pred):
        p = self.split_pred(concat_pred)
        return p['y1_pred'] - p['y0_pred']

    def PEHE(self, concat_pred):
        # simulation only
        p = self.split_pred(concat_pred)
        cate_err = tf.reduce_mean(tf.square(((self.data['mu_1'] - self.data['mu_0']) - (p['y1_pred'] - p['y0_pred']))))
        return cate_err

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        concat_pred = self.model.predict(self.data['x'])
        # Calculate Empirical Metrics
        ate_pred = tf.reduce_mean(self.ATE(concat_pred))
        tf.summary.scalar('ate', data=ate_pred, step=epoch)
        pehe_nn = self.PEHEnn(concat_pred)
        tf.summary.scalar('cate_nn_err', data=tf.sqrt(pehe_nn), step=epoch)

        # Simulation Metrics
        ate_true = tf.reduce_mean(self.data['mu_1'] - self.data['mu_0'])
        ate_true = tf.dtypes.cast(ate_true, tf.float32)
        ate_err = tf.abs(ate_true - ate_pred)
        tf.summary.scalar('ate_err', data=ate_err, step=epoch)
        pehe = self.PEHE(concat_pred)
        tf.summary.scalar('cate_err', data=tf.sqrt(pehe), step=epoch)
        out_str = f' — ate_err: {ate_err:.4f}  — cate_err: {tf.sqrt(pehe):.4f} — cate_nn_err: {tf.sqrt(pehe_nn):.4f} '

        if self.verbose > 0:
            print(out_str)


class Weighted_Metrics(Base_Metrics):
    def __init__(self, data, verbose=0):
        super().__init__(data, verbose)

    def split_pred(self, concat_pred):
        preds = {}
        preds['y0_pred'] = self.data['y_scaler'].inverse_transform(concat_pred[:, 0].reshape(-1, 1))
        preds['y1_pred'] = self.data['y_scaler'].inverse_transform(concat_pred[:, 1].reshape(-1, 1))
        preds['t_pred'] = concat_pred[:, 2]
        preds['phi'] = concat_pred[:, 3:]

        return preds


class CFRNet_Loss(Loss):
    # initialize instance attributes
    def __init__(self, alpha=1., sigma=1., ipm='weighted'):
        super().__init__()
        self.alpha = alpha  # balances regression loss and MMD IPM
        self.rbf_sigma = sigma  # for gaussian kernel
        self.name = 'cfrnet_loss'
        self.ipm_type = ipm

    def split_pred(self, concat_pred):
        # generic helper to make sure we dont make mistakes
        preds = {}
        preds['y0_pred'] = concat_pred[:, 0]
        preds['y1_pred'] = concat_pred[:, 1]
        preds['phi'] = concat_pred[:, 2:]
        return preds

    @staticmethod
    def pdist2sq(A, B):
        # helper for PEHEnn and rbf_kernel
        # calculates squared euclidean distance between rows of two matrices
        # https://gist.github.com/mbsariyildiz/34cdc26afb630e8cae079048eef91865
        # return pairwise euclidean difference matrix
        D = tf.reduce_sum((tf.expand_dims(A, 1) - tf.expand_dims(B, 0)) ** 2, 2)
        return D

    def rbf_kernel(self, A, B):
        return tf.exp(-self.pdist2sq(A, B) / tf.square(self.rbf_sigma) * .5)

    def calc_mmdsq(self, Phi, t):
        Phic, Phit = tf.dynamic_partition(Phi, tf.cast(tf.squeeze(t), tf.int32), 2)
        Kcc = self.rbf_kernel(Phic, Phic)
        Kct = self.rbf_kernel(Phic, Phit)
        Ktt = self.rbf_kernel(Phit, Phit)

        m = tf.cast(tf.shape(Phic)[0], Phi.dtype)
        n = tf.cast(tf.shape(Phit)[0], Phi.dtype)

        mmd = 1.0 / (m * (m - 1.0)) * (tf.reduce_sum(Kcc) - m)
        mmd = mmd + 1.0 / (n * (n - 1.0)) * (tf.reduce_sum(Ktt) - n)
        mmd = mmd - 2.0 / (m * n) * tf.reduce_sum(Kct)
        return mmd * tf.ones_like(t)

    def mmdsq_loss(self, concat_true, concat_pred):
        t_true = concat_true[:, 1]
        p = self.split_pred(concat_pred)
        mmdsq_loss = tf.reduce_mean(self.calc_mmdsq(p['phi'], t_true))
        return mmdsq_loss

    def regression_loss(self, concat_true, concat_pred):
        y_true = concat_true[:, 0]
        t_true = concat_true[:, 1]
        p = self.split_pred(concat_pred)
        loss0 = tf.reduce_mean((1. - t_true) * tf.square(y_true - p['y0_pred']))
        loss1 = tf.reduce_mean(t_true * tf.square(y_true - p['y1_pred']))
        return loss0 + loss1

    def safe_sqrt(self, x, lbound=1e-10):
        """ Numerically safe version of TensorFlow sqrt """
        return tf.sqrt(tf.clip_by_value(x, lbound, np.inf))

    def wasserstein(self, concat_true, concat_pred, p=0.5, lam=10, its=10, sq=False, backpropT=False):
        t = concat_true[:, 1]
        preds = self.split_pred(concat_pred)
        X = preds['phi']

        """ Returns the Wasserstein distance between treatment groups """
        it = tf.where(t > 0)[:, 0]
        ic = tf.where(t < 1)[:, 0]
        Xc = tf.gather(X, ic)
        Xt = tf.gather(X, it)
        nc = tf.cast(tf.shape(Xc)[0], tf.float32)
        nt = tf.cast(tf.shape(Xt)[0], tf.float32)

        """ Compute distance matrix"""
        if sq:
            M = self.pdist2sq(Xt, Xc)
        else:
            M = self.safe_sqrt(self.pdist2sq(Xt, Xc))

        """ Estimate lambda and delta """
        M_mean = tf.reduce_mean(M)
        M_drop = tf.nn.dropout(M, 10 / (nc * nt))
        delta = tf.stop_gradient(tf.reduce_max(M))
        eff_lam = tf.stop_gradient(lam / M_mean)

        """ Compute new distance matrix """
        Mt = M
        row = delta * tf.ones(tf.shape(M[0:1, :]))
        col = tf.concat([delta * tf.ones(tf.shape(M[:, 0:1])), tf.zeros((1, 1))], 0)
        Mt = tf.concat([M, row], 0)
        Mt = tf.concat([Mt, col], 1)

        """ Compute marginal vectors """
        a = tf.concat([p * tf.ones(tf.shape(tf.where(t > 0)[:, 0:1])) / nt, (1 - p) * tf.ones((1, 1))], 0)
        b = tf.concat([(1 - p) * tf.ones(tf.shape(tf.where(t < 1)[:, 0:1])) / nc, p * tf.ones((1, 1))], 0)

        """ Compute kernel matrix"""
        Mlam = eff_lam * Mt
        K = tf.exp(-Mlam) + 1e-6  # added constant to avoid nan
        U = K * Mt
        ainvK = K / a

        u = a
        for i in range(0, its):
            u = 1.0 / (tf.matmul(ainvK, (b / tf.transpose(tf.matmul(tf.transpose(u), K)))))
        v = b / (tf.transpose(tf.matmul(tf.transpose(u), K)))

        T = u * (tf.transpose(v) * K)

        if not backpropT:
            T = tf.stop_gradient(T)

        E = T * Mt
        D = 2 * tf.reduce_sum(E)

        return D

    def cfr_loss(self, concat_true, concat_pred):
        lossR = self.regression_loss(concat_true, concat_pred)
        if self.ipm_type == 'wasserstein':
            lossIPM = self.wasserstein(concat_true, concat_pred)
        else:
            lossIPM = self.mmdsq_loss(concat_true, concat_pred)
        return lossR + self.alpha * lossIPM

    # compute loss
    def call(self, concat_true, concat_pred):
        return self.cfr_loss(concat_true, concat_pred)


class Weighted_CFRNet_Loss(CFRNet_Loss):
    # initialize instance attributes
    def __init__(self, prob_treat, alpha=1., sigma=1.0):
        super().__init__()
        self.pT = prob_treat
        self.alpha = alpha
        self.rbf_sigma = sigma
        self.name = 'cfrnet_loss'

    def split_pred(self, concat_pred):
        # generic helper to make sure we dont make mistakes
        preds = {}
        preds['y0_pred'] = concat_pred[:, 0]
        preds['y1_pred'] = concat_pred[:, 1]
        preds['t_pred'] = concat_pred[:, 2]
        preds['t_pred'] = (preds['t_pred'] + 0.001) / 1.002
        preds['phi'] = concat_pred[:, 3:]
        return preds

    # for logging purposes only
    def treatment_acc(self, concat_true, concat_pred):
        t_true = concat_true[:, 1]
        p = self.split_pred(concat_pred)
        # Since this isn't used as a loss, I've used tf.reduce_mean for interpretability
        return tf.reduce_mean(binary_accuracy(t_true, p['t_pred'], threshold=0.5))

    def calc_weighted_mmdsq(self, Phi, t_true, t_pred):
        t_predC, t_predT = tf.dynamic_partition(t_pred, tf.cast(tf.squeeze(t_true), tf.int32), 2)  # propensity
        PhiC, PhiT = tf.dynamic_partition(Phi, tf.cast(tf.squeeze(t_true), tf.int32), 2)  # representation
        weightC = tf.expand_dims((1 - self.pT) / (1 - t_predC), axis=-1)
        weightT = tf.expand_dims(self.pT / t_predT, axis=-1)

        wPhiC = weightC * PhiC
        wPhiT = weightT * PhiT

        Kcc = self.rbf_kernel(wPhiC, wPhiC)
        Kct = self.rbf_kernel(wPhiC, wPhiT)
        Ktt = self.rbf_kernel(wPhiT, wPhiT)

        m = tf.cast(tf.shape(PhiC)[0], Phi.dtype)
        n = tf.cast(tf.shape(PhiT)[0], Phi.dtype)

        mmd = 1.0 / (m * (m - 1.0)) * (tf.reduce_sum(Kcc) - m)
        mmd = mmd + 1.0 / (n * (n - 1.0)) * (tf.reduce_sum(Ktt) - n)
        mmd = mmd - 2.0 / (m * n) * tf.reduce_sum(Kct)
        return mmd * tf.ones_like(t_true)

    def weighted_mmdsq_loss(self, concat_true, concat_pred):
        t_true = concat_true[:, 1]
        p = self.split_pred(concat_pred)
        mmdsq = tf.reduce_mean(self.calc_weighted_mmdsq(p['phi'], t_true, p['t_pred']))
        return mmdsq

    def weights(self, concat_true, concat_pred):
        p = self.split_pred(concat_pred)
        weightT = tf.expand_dims(self.pT / p['t_pred'], axis=-1)
        return tf.reduce_mean(weightT)

    def weighted_regression_loss(self, concat_true, concat_pred):
        y_true = concat_true[:, 0]
        t_true = concat_true[:, 1]
        p = self.split_pred(concat_pred)

        weightC = tf.expand_dims((1 - self.pT) / (1. - p['t_pred']), axis=-1)
        weightT = tf.expand_dims(self.pT / p['t_pred'], axis=-1)

        loss0 = tf.reduce_mean((1. - t_true) * tf.square(y_true - p['y0_pred']) * weightC)
        loss1 = tf.reduce_mean(t_true * tf.square(y_true - p['y1_pred']) * weightT)
        return loss0 + loss1

    def call(self, concat_true, concat_pred):
        return self.weighted_regression_loss(concat_true, concat_pred) + self.alpha * self.weighted_mmdsq_loss(
            concat_true, concat_pred)


class CFRModel(Model):
    def __init__(self, name, params, hp, **kwargs):
        super(CFRModel, self).__init__(name=name, **kwargs)
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

    def call(self, inputs):
        phi = self.fc(inputs)
        y0_pred = self.pred_y0(phi)
        y1_pred = self.pred_y1(phi)
        concat_pred = tf.concat([y0_pred, y1_pred, phi], axis=-1)
        return concat_pred


class Weighted_CFR(Model):
    def __init__(self, name, params, hp, **kwargs):
        super(Weighted_CFR, self).__init__(name=name, **kwargs)
        self.params = params
        self.hp_fc = hp.Int('n_fc', min_value=2, max_value=10, step=1)
        self.hp_hidden_phi = hp.Int('hidden_phi', min_value=16, max_value=512, step=16)
        self.fc = FullyConnected(n_fc=self.hp_fc, hidden_phi=self.hp_hidden_phi, final_activation='elu',
                                 out_size=params['hidden_phi'], kernel_init=params['kernel_init'], kernel_reg=None,
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

    def call(self, inputs):
        phi = self.fc(inputs)
        y0_pred = self.pred_y0(phi)
        y1_pred = self.pred_y1(phi)
        t_pred = self.pred_t(phi)
        concat_pred = tf.concat([y0_pred, y1_pred, t_pred, phi], axis=-1)
        return concat_pred


class CFRNet(CausalModel):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.ipm_type = params['ipm_type']

        self.ipm_list = {'mmdsq', 'wasserstein', 'weighted'}
        if self.ipm_type not in self.ipm_list:
            raise ValueError(f'IPM type {self.ipm_type} not defined!')

    def fit_model(self, x, y, t, seed, count):
        directory_name = 'params/' + self.params['dataset_name']
        setSeed(seed)
        t = tf.cast(t, dtype=tf.float32)
        pT = t[t == 1].shape[0] / t.shape[0]
        yt = tf.concat([y, t], axis=1)

        if self.dataset_name == 'acic':
            directory_name = directory_name + f'/{self.params["model_name"]}_{self.params["ipm_type"]}'
            project_name = str(self.folder_ind)
        else:
            project_name = f'{self.params["model_name"]}_{self.params["ipm_type"]}'

        if self.params['ipm_type'] == 'weighted':
            val_monitor_metric = 'val_weighted_mmdsq_loss'
        elif self.params['ipm_type'] == 'mmdsq':
            val_monitor_metric = 'val_mmdsq_loss'
        else:
            val_monitor_metric = 'val_cfrnet_loss'

        tuner = kt.RandomSearch(
            HyperCFRNet(params=self.params, pT=pT),
            objective=kt.Objective(val_monitor_metric, direction="min"),
            directory=directory_name,
            tuner_id='1',
            overwrite=False,
            project_name=project_name,
            max_trials=10,
            seed=0)

        stop_early = [TerminateOnNaN(), EarlyStopping(monitor=val_monitor_metric, patience=5)]
        tuner.search(x, yt, validation_split=0.2, epochs=50, callbacks=[stop_early], verbose=1)

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        if self.params['defaults']:
            if self.params['ipm_type'] == 'weighted':
                best_hps.values = {'n_fc': self.params['n_fc'], 'hidden_phi': self.params['hidden_phi'],
                                   'n_fc_y0': self.params['n_fc_y0'], 'hidden_y0': self.params['hidden_y0'],
                                   'n_fc_y1': self.params['n_fc_y1'], 'hidden_y1': self.params['hidden_y1'],
                                   'n_fc_t': self.params['n_fc_t'], 'hidden_t': self.params['hidden_t']}
            else:
                best_hps.values = {'n_fc': self.params['n_fc'], 'hidden_phi': self.params['hidden_phi'],
                                   'n_fc_y0': self.params['n_fc_y0'], 'hidden_y0': self.params['hidden_y0'],
                                   'n_fc_y1': self.params['n_fc_y1'], 'hidden_y1': self.params['hidden_y1']}
        model = tuner.hypermodel.build(best_hps)

        if self.params['ipm_type'] == 'weighted':
            monitor_metric = 'loss'
        elif self.params['ipm_type'] == 'mmdsq':
            monitor_metric = 'cfrnet_loss'
        else:
            monitor_metric = 'cfrnet_loss'

        model.fit(x, yt,
                  callbacks=callbacks(monitor_metric),
                  validation_split=0.0,
                  batch_size=self.params['batch_size'],
                  epochs=self.params['epochs'],
                  verbose=self.params['verbose'])
        if count == 0:
            if self.params['ipm_type'] == 'weighted':
                print(f"""The hyperparameter search is complete. the optimal hyperparameters are
                                  layer is n_fc={best_hps.get('n_fc')} hidden_phi = {best_hps.get('hidden_phi')}
                                  hidden_y1 = {best_hps.get('hidden_y1')} n_fc_y1 = {best_hps.get('n_fc_y1')}
                                  hidden_y0 = {best_hps.get('hidden_y0')}  n_fc_y0 = {best_hps.get('n_fc_y0')},
                                  n_fc_t ={best_hps.get('n_fc_t')}, hidden_t={best_hps.get('hidden_t')} """)
            else:
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

        self.folder_ind = kwargs.get('folder_ind')
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
