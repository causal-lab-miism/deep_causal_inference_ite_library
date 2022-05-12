from sklearn.preprocessing import StandardScaler
import pandas as pd
from scipy.special import expit
from utils.set_seed import *


class CausalModel:
    def __init__(self, params):
        self.dataset_name = params['dataset_name']
        self.num = params['num']
        self.params = params
        self.binary = params['binary']

    @staticmethod
    def setSeed(seed):
        os.environ['PYTHONHASHSEED'] = '0'

        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)

    def train_and_evaluate(self, pehe_list, **kwargs):
        pass

    def evaluate_performance(self):
        if self.dataset_name in ['ihdp_a', 'ihdp_b']:
            return self.evaluate_performance_ihdp()
        if self.dataset_name == 'twins':
            return self.evaluate_performance_twins()
        if self.dataset_name == 'jobs':
            return self.evaluate_performance_jobs()
        if self.dataset_name == 'acic':
            return self.evaluate_performance_acic()

    # evaluate the model performance
    def evaluate_performance_ihdp(self):
        num = self.num
        pehe_list = list()
        for count in range(num):
            # if count == 20:
            #     break
            kwargs = {'count': count}
            self.train_and_evaluate(pehe_list, **kwargs)
        return pehe_list

    def evaluate_performance_twins(self):
        num = self.num
        pehe_list = list()
        for count in range(num):
            kwargs = {'count': count}
            self.train_and_evaluate(pehe_list, **kwargs)
        return pehe_list

    def evaluate_performance_jobs(self):
        num = self.num
        policy_risk_list = list()
        for count in range(num):
            kwargs = {'count': count}
            self.train_and_evaluate(policy_risk_list, **kwargs)
        return policy_risk_list

    def evaluate_performance_acic(self):
        num = self.num
        pehe_list = list()
        for folder in range(1, num+1):
            len_folder_files = len(os.listdir('./ACIC/' + str(folder) + '/'))
            for file in range(len_folder_files):
                kwargs = {'folder_ind': folder, 'file_ind': file}
                self.train_and_evaluate(pehe_list, **kwargs)
        return pehe_list

    def find_pehe(self, y0_pred, y1_pred, data):
        if self.binary:
            cate_pred = y1_pred - y0_pred
        else:
            y0_pred = data['y_scaler'].inverse_transform(y0_pred)
            y1_pred = data['y_scaler'].inverse_transform(y1_pred)
            cate_pred = (y1_pred - y0_pred).squeeze()

        cate_true = (data['mu_1'] - data['mu_0']).squeeze()

        pehe = tf.reduce_mean(tf.square((cate_true - cate_pred)))
        sqrt_pehe = tf.sqrt(pehe).numpy()

        return sqrt_pehe

    def find_policy_risk(self, y0_pred, y1_pred, data):
        if self.binary:
            cate_pred = y1_pred - y0_pred
        else:
            cate_pred = (y1_pred - y0_pred).squeeze()

        # data['t'] = tf.cast(data['t'], tf.int32)
        # cate_pred[data['t'] > 0] = -cate_pred[data['t'] > 0]

        cate_true = data['tau']

        policy_value, policy_curve = self.policy_val(data['t'][cate_true > 0], data['y'][cate_true > 0],
                                                     cate_pred[cate_true > 0], False)
        policy_risk = 1 - policy_value

        return policy_value, policy_risk, policy_curve

    @staticmethod
    def policy_range(n, res=10):
        step = int(float(n) / float(res))
        n_range = range(0, int(n + 1), step)
        if not n_range[-1] == n:
            n_range.append(n)

        # To make sure every curve is same length. Incurs a small error if res high.
        # Only occurs if number of units considered differs.
        # For example if resampling validation sets (with different number of
        # units in the randomized sub-population)

        while len(n_range) > res:
            k = np.random.randint(len(n_range) - 2) + 1
            del n_range[k]

        return n_range

    def policy_val(self, t, yf, eff_pred, compute_policy_curve=False):
        """ Computes the value of the policy defined by predicted effect """

        pol_curve_res = 40

        if np.any(np.isnan(eff_pred)):
            return np.nan, np.nan

        policy = eff_pred > 0

        if isinstance(policy, np.ndarray) and isinstance(t, np.ndarray):
            treat_overlap = (policy == t) * (t > 0)
            control_overlap = (policy == t) * (t < 1)
        else:
            treat_overlap = (policy == t).numpy() * (t > 0)
            control_overlap = (policy == t).numpy() * (t < 1)

        if np.sum(treat_overlap) == 0:
            treat_value = 0
        else:
            treat_value = np.mean(yf[treat_overlap])

        if np.sum(control_overlap) == 0:
            control_value = 0
        else:
            control_value = np.mean(yf[control_overlap])

        pit = np.mean(policy)
        policy_value = pit * treat_value + (1 - pit) * control_value

        policy_curve = []

        if compute_policy_curve:
            n = t.shape[0]
            I_sort = np.argsort(-eff_pred)

            n_range = self.policy_range(n, pol_curve_res)

            for i in n_range:
                I = I_sort[0:i]

                policy_i = 0 * policy
                policy_i[I] = 1
                pit_i = np.mean(policy_i)

                treat_overlap = (policy_i > 0) * (t > 0)
                control_overlap = (policy_i < 1) * (t < 1)

                if np.sum(treat_overlap) == 0:
                    treat_value = 0
                else:
                    treat_value = np.mean(yf[treat_overlap])

                if np.sum(control_overlap) == 0:
                    control_value = 0
                else:
                    control_value = np.mean(yf[control_overlap])

                policy_curve.append(pit_i * treat_value + (1 - pit_i) * control_value)

        return policy_value, policy_curve

    def regression_loss(self, concat_true, concat_pred):
        y_true = concat_true[:, 0]
        t_true = concat_true[:, 1]

        y0_pred = concat_pred[:, 0]
        y1_pred = concat_pred[:, 1]

        if self.params['binary']:
            y0_pred = tf.cast((y0_pred > 0.5), tf.float32)
            y1_pred = tf.cast((y1_pred > 0.5), tf.float32)
            # bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
            # loss0 = bce(((1 - t_true) * y_true), y0_pred, sample_weight=[0.84, 0.16])
            # loss1 = bce((t_true * y_true, y1_pred), sample_weight=[0.84, 0.16])

            loss0 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=(1 - t_true) * y_true,
                                                                           logits=y0_pred))
            loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=t_true * y_true, logits=y1_pred))

            # loss0 = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=(1 - t_true) * y_true,
            #                                                                 logits=y0_pred, pos_weight=0))
            # loss1 = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=t_true * y_true, logits=y1_pred,
            #                                                                 pos_weight=0))

            # loss0 = tf.keras.losses.binary_crossentropy((1 - t_true) * y_true, y0_pred)
            # loss1 = tf.keras.losses.binary_crossentropy(t_true * y_true, y1_pred)
        else:
            loss0 = tf.reduce_sum((1. - t_true) * tf.square(y_true - y0_pred))
            loss1 = tf.reduce_sum(t_true * tf.square(y_true - y1_pred))
        return loss0 + loss1

    def load_data(self, **kwargs):
        if self.dataset_name == 'ihdp_a':
            path_data = "./IHDP_a"
            return self.load_ihdp_data(path_data, kwargs.get('count'))
        elif self.dataset_name == 'ihdp_b':
            path_data = "./IHDP_b"
            return self.load_ihdp_data(path_data, kwargs.get('count'))
        elif self.dataset_name == 'acic':
            return self.load_acic_data(kwargs.get('folder_ind'), kwargs.get('file_ind'))
        elif self.dataset_name == 'twins':
            return self.load_twins_data(kwargs.get('count'))
        elif self.dataset_name == 'jobs':
            path_data = "./JOBS"
            return self.load_jobs_data(path_data, kwargs.get('count'))
        else:
            print('No such dataset. The available datasets are: ', 'ihdp_a, ihdp_b, acic, twins, jobs')

    @staticmethod
    def load_ihdp_data(path_data, i=7):

        data_train = np.loadtxt(path_data + '/ihdp_npci_train_' + str(i + 1) + '.csv', delimiter=',', skiprows=1)
        data_test = np.loadtxt(path_data + '/ihdp_npci_test_' + str(i + 1) + '.csv', delimiter=',', skiprows=1)

        t_train, y_train = data_train[:, 0], data_train[:, 1][:, np.newaxis]
        mu_0_train, mu_1_train, x_train = data_train[:, 3][:, np.newaxis], data_train[:, 4][:, np.newaxis], data_train[
                                                                                                            :, 5:]

        t_test, y_test = data_test[:, 0].astype('float32'), data_test[:, 1][:, np.newaxis].astype('float32')
        mu_0_test, mu_1_test, x_test = data_test[:, 3][:, np.newaxis].astype('float32'), data_test[:, 4][:,
                                                                                         np.newaxis].astype('float32'), \
                                       data_test[:, 5:].astype('float32')

        data_train = {'x': x_train, 't': t_train, 'y': y_train, 'mu_0': mu_0_train, 'mu_1': mu_1_train}
        # data_train = remove_anomalies(data_train)

        data_train['t'] = data_train['t'].reshape(-1,
                                                  1)  # we're just padding one dimensional vectors with an additional dimension
        data_train['y'] = data_train['y'].reshape(-1, 1)
        # rescaling y between 0 and 1 often makes training of DL regressors easier
        data_train['y_scaler'] = StandardScaler().fit(data_train['y'])
        data_train['ys'] = data_train['y_scaler'].transform(data_train['y'])

        data_test = {'x': x_test, 't': t_test, 'y': y_test, 'mu_0': mu_0_test, 'mu_1': mu_1_test}
        data_test['t'] = data_test['t'].reshape(-1,
                                                1)  # we're just padding one dimensional vectors with an additional dimension
        data_test['y'] = data_test['y'].reshape(-1, 1)
        # rescaling y between 0 and 1 often makes training of DL regressors easier
        data_test['y_scaler'] = data_train['y_scaler']
        data_test['ys'] = data_test['y_scaler'].transform(data_test['y'])

        return data_train, data_test

    @staticmethod
    def load_jobs_data(path_data, i=7):
        data_file_train = path_data + f'/jobs_train_{i}.csv'

        data_file_test = path_data + f'/jobs_test_{i}.csv'

        df_train = np.loadtxt(data_file_train, delimiter=',', skiprows=1)

        x_train = np.squeeze(df_train[:, 0:17])  # confounders
        t_train = df_train[:, 17:18]  # factual observation
        y_train = df_train[:, 18:19].astype(np.float32)  # treatment
        e_train = df_train[:, 19:20]  # randomized trial

        data_mean = np.mean(x_train, axis=0, keepdims=True)
        data_std = np.std(x_train, axis=0, keepdims=True)

        x_train = (x_train - data_mean) / data_std

        data_train = {'x': x_train, 'y': y_train, 't': t_train, 'tau': e_train}

        df_test = np.loadtxt(data_file_test, delimiter=',', skiprows=1)

        x_test = np.squeeze(df_test[:, 0:17])  # confounders
        t_test = df_test[:, 17:18]  # factual observation
        y_test = df_test[:, 18:19].astype(np.float32)  # treatment
        e_test = df_test[:, 19:20]  # randomized trial
        
        x_test = (x_test - data_mean) /data_std

        data_test = {'x': x_test, 'y': y_test, 't': t_test, 'tau': e_test}

        return data_train, data_test

    @staticmethod
    def load_cfdata(file_dir):
        df = pd.read_csv(file_dir)
        z = df['z'].values[:, np.newaxis].astype('float32')
        y0 = df['y0'].values[:, np.newaxis].astype('float32')
        y1 = df['y1'].values[:, np.newaxis].astype('float32')
        y = y0 * (1 - z) + y1 * z
        mu_0, mu_1 = df['mu0'].values[:, np.newaxis].astype('float32'), df['mu1'].values[:, np.newaxis].astype(
            'float32')

        data_cf = {'t': z, 'y': y, 'mu_0': mu_0, 'mu_1': mu_1}
        return data_cf

    def load_acic_data(self, folder_ind=1, file_ind=1):
        data = pd.read_csv('./ACIC/x.csv')
        del data['x_2']
        del data['x_21']
        del data['x_24']

        data = data.dropna()
        data = data.values

        # load y and simulations
        folder_dir = './ACIC/' + str(folder_ind) + '/'
        filelist = os.listdir(folder_dir)
        data_cf = self.load_cfdata(folder_dir + filelist[file_ind])

        # number of observations
        n = data.shape[0]
        test_ind = 4000

        # create train data
        x_train = data[:test_ind, :]

        data_mean = np.mean(x_train, axis=0, keepdims=True)
        data_std = np.std(x_train, axis=0, keepdims=True)
        x_train = (x_train - data_mean) / data_std

        y_train = data_cf['y'][:test_ind, :]
        t_train = data_cf['t'][:test_ind, :]
        mu_0_train = data_cf['mu_0'][:test_ind]
        mu_1_train = data_cf['mu_1'][:test_ind]

        data_train = {'x': x_train, 't': t_train, 'y': y_train, 'mu_0': mu_0_train, 'mu_1': mu_1_train}

        data_train['t'] = data_train['t'].reshape(-1, 1)  # we're just padding one dimensional vectors with an additional dimension
        data_train['y'] = data_train['y'].reshape(-1, 1)
        # rescaling y between 0 and 1 often makes training of DL regressors easier
        data_train['y_scaler'] = StandardScaler().fit(data_train['y'])
        data_train['ys'] = data_train['y_scaler'].transform(data_train['y'])
        # create test data

        x_test = data[test_ind:, :]
        x_test = (x_test - data_mean) / data_std
        y_test = data_cf['y'][test_ind:, :]
        t_test = data_cf['t'][test_ind:, :]
        mu_0_test = data_cf['mu_0'][test_ind:]
        mu_1_test = data_cf['mu_1'][test_ind:]

        data_test = {'x': x_test, 't': t_test, 'y': y_test, 'mu_0': mu_0_test, 'mu_1': mu_1_test}
        data_test['t'] = data_test['t'].reshape(-1, 1)  # we're just padding one dimensional vectors with an additional dimension
        data_test['y'] = data_test['y'].reshape(-1, 1)
        # rescaling y between 0 and 1 often makes training of DL regressors easier
        data_test['y_scaler'] = StandardScaler().fit(data_test['y'])
        data_test['ys'] = data_test['y_scaler'].transform(data_test['y'])

        return data_train, data_test

    @staticmethod
    def load_twins_data(count=1):
        train_rate = 0.8
        # Load original data (11400 patients, 30 features, 2 dimensional potential outcomes)
        df = np.loadtxt("./TWINS/Twin_data.csv", delimiter=",", skiprows=1)

        # Define features
        x = df[:, :30]
        no, dim = x.shape
        # Define potential outcomes
        mu0_mu1 = df[:, 30:]
        # Die within 1 year = 1, otherwise = 0
        mu0_mu1 = np.array(mu0_mu1 < 9999, dtype=float)

        ## Assign treatment
        setSeed(count)
        np.random.seed(count)
        coef = np.random.uniform(-0.01, 0.01, size=[dim, 1])

        prob_temp = expit(np.matmul(x, coef) + np.random.normal(0, 0.01, size=[no, 1]))

        prob_t = prob_temp / (2 * np.mean(prob_temp))
        prob_t[prob_t > 1] = 1

        t = np.random.binomial(1, prob_t, [no, 1])
        t = t.reshape([no, ])

        ## Define observable outcomes
        y = np.zeros([no, 1])
        y = np.transpose(t) * mu0_mu1[:, 1] + np.transpose(1 - t) * mu0_mu1[:, 0]
        y = np.reshape(np.transpose(y), [no, ])

        ## Train/test division
        idx = np.random.permutation(no)
        train_idx = idx[:int(train_rate * no)]
        test_idx = idx[int(train_rate * no):]

        x_train = x[train_idx, :]
        data_mean = np.mean(x_train, axis=0, keepdims=True)
        data_std = np.std(x_train, axis=0, keepdims=True)
        # x_train = (x_train - data_mean) / data_std

        t_train = t[train_idx]
        y_train = y[train_idx]
        mu0_mu1_train = mu0_mu1[train_idx, :]
        mu_0_train = mu0_mu1_train[:, 0]
        mu_1_train = mu0_mu1_train[:, 1]

        data_train = {'x': x_train, 't': t_train, 'y': y_train, 'mu_0': mu_0_train, 'mu_1': mu_1_train}
        data_train['t'] = data_train['t'].reshape(-1, 1)  # we're just padding one dimensional vectors with
        # an additional dimension
        data_train['y'] = data_train['y'].reshape(-1, 1)

        x_test = x[test_idx, :]
        # x_test = (x_test - data_mean) / data_std
        t_test = t[test_idx]
        y_test = y[test_idx]
        mu0_mu1_test = mu0_mu1[test_idx, :]
        mu_0_test = mu0_mu1_test[:, 0]
        mu_1_test = mu0_mu1_test[:, 1]

        data_test = {'x': x_test, 't': t_test, 'y': y_test, 'mu_0': mu_0_test, 'mu_1': mu_1_test}
        data_test['t'] = data_test['t'].reshape(-1, 1)  # we're just padding one dimensional vectors with
        # an additional dimension
        data_test['y'] = data_test['y'].reshape(-1, 1)

        return data_train, data_test
