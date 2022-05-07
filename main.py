from models.HyperCFRNet import *
from models.HyperDKLITE import *
# from models.DKLITE import *
from models.TARnet1 import *
from models.HyperRLearner import *
from models.HyperSLearner import *
from models.XLearner import *
from models.HyperTLearner import *
from models.TEDVAE import *
from models.DragonNet import *
from models.HyperGANITE import *
from models.HyperCEVAE import *
from models.BVNICEModel import *
import scipy.stats
import argparse
from hyperparameters import *
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h


def main(args):
    model_names = {"TARnet": TARnetModel1,  "XLearner": XLearner, "TLearner": HyperTLearnerModel,
                   "CFRNet": CFRNet1, "DragonNet": DragonNet, "DKLITE": HyperDKLITE,
                   "GANITE": GANITE, "SLearner": SLearner, "RLearner": RLearner, "TEDVAE": TEDVAE,
                   "CEVAE": CEVAE}

    datasets = {'ihdp_a', 'ihdp_b', 'acic', 'twins', 'jobs'}
    ipm_list = {'mmdsq', 'wasserstein', 'weighted', None}

    if args.model_name in model_names and args.dataset_name in datasets and args.ipm_type in ipm_list:
        print('Chosen model is', args.model_name, args.dataset_name, args.ipm_type)
        params = find_params(args.model_name, args.dataset_name)
        model_name = model_names[args.model_name]
        params['model_name'] = args.model_name
        params['dataset_name'] = args.dataset_name
        params['ipm_type'] = args.ipm_type
        model = model_name(params)
        pehe_list = model.evaluate_performance()
        m, h = mean_confidence_interval(pehe_list, confidence=0.95)
        print(f'mean PEHE test: {m} | std PEHE test: {h}')
        return 0
    else:
        raise ValueError(f'{args.model_name} has not been implemented yet!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Causal Model')
    parser.add_argument("--model-name", default="TLearner", type=str)
    parser.add_argument("--ipm-type", default='wasserstein', type=str)
    parser.add_argument("--dataset-name", default="ihdp_b", type=str)
    parser.add_argument("--num", default=100, type=int)
    args = parser.parse_args()
    main(args)
