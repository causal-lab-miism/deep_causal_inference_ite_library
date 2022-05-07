import tensorflow as tf
import numpy as np
import random


def find_params(model_name, dataset_name):

    """SLEARNER"""

    # 0.43 ± 0.04 - Adam 1e-3
    params_SLearner_IHDP_a = {'dataset_name': "ihdp_a", 'num': 100, 'lr': 1e-3, 'patience': 40,
                              'batch_size': 64, 'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 112,
                              'epochs': 300, 'binary': False, 'n_fc': 9, 'verbose': 0, 'val_split': 0.0,
                              'kernel_init': 'RandomNormal'}

    #2.24 +- 0.06 - Adam 1e-3
    params_SLearner_IHDP_b = {'dataset_name': "ihdp_b", 'num': 100, 'lr': 1e-3, 'patience': 40,
                              'batch_size': 64, 'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 352,
                              'epochs': 300, 'binary': False, 'n_fc': 6, 'verbose': 0, 'val_split': 0.1,
                              'kernel_init': 'GlorotNormal'}

    params_SLearner_ACIC = {'dataset_name': "acic", 'num': 77, 'lr': 1e-3, 'patience': 40,
                            'batch_size': 256, 'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 256, 
                            'val_split': 0.1, 'epochs': 300, 'binary': False, 'n_fc': 7, 'verbose': 0, 
                            'kernel_init': 'RandomNormal'}

    params_SLearner_JOBS = {'dataset_name': "jobs", 'num': 100, 'lr': 1e-3, 'patience': 40,
                            'batch_size': 673, 'reg_l2': .01, 'activation': 'sigmoid', 'hidden_phi': 128, 
                            'val_split': 0.1, 'epochs': 300, 'binary': True, 'n_fc': 5, 'verbose': 0, 
                            'kernel_init': 'RandomNormal'}

    """TLEARNER"""

    #0.57 +- 0.04
    params_TLearner_IHDP_a = {'dataset_name': "ihdp_a", 'num': 100, 'lr_0': 1e-2, 'lr_1': 1e-3, 'patience': 40,
                              'batch_size': 64, 'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 256,
                              'epochs': 1000, 'binary': False, 'n_fc': 6, 'verbose': 0, 'val_split': 0.1,
                              'kernel_init': 'RandomNormal', 'model_name_0': 'TLearner0',  'model_name_1': 'TLearner1'}

    params_TLearner_IHDP_b = {'dataset_name': "ihdp_b", 'num': 100, 'lr': 1e-3, 'patience': 40,
                              'batch_size': 64, 'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 128,
                              'epochs': 1000, 'binary': False, 'n_fc': 7, 'verbose': 0, 'val_split': 0.1,
                              'kernel_init': 'GlorotNormal', 'model_name_0': 'TLearner0',  'model_name_1': 'TLearner1'}

    params_TLearner_ACIC = {'dataset_name': "acic", 'num': 77, 'lr': 1e-2, 'patience': 40,
                            'batch_size': 128, 'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 128, 'val_split': 0.1,
                            'epochs': 1000, 'binary': False, 'n_fc': 7, 'verbose': 0, 'kernel_init': 'GlorotNormal',
                            'model_name_0': 'TLearner0',  'model_name_1': 'TLearner1'}

    # mean Policy Risk test: 0.2436| std Policy Risk test: 0.01175
    params_TLearner_JOBS = {'dataset_name': "jobs", 'num': 100, 'lr': 1e-3, 'patience': 40,
                            'batch_size': 256, 'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 64, 'val_split': 0.1,
                            'epochs': 1000, 'binary': True, 'n_fc': 3, 'verbose': 0, 'kernel_init': 'GlorotNormal',
                            'model_name_0': 'TLearner0',  'model_name_1': 'TLearner1'}

    """RLEARNER"""

    # 0.597 +- 0.06
    params_RLearner_IHDP_a = {'dataset_name': "ihdp_a", 'num': 100, 'lr': 1e-3, 'patience': 40, 'batch_size_g_mu': 256,
                              'batch_size_r': 32, 'reg_l2': .01, 'activation': 'linear', 'hidden_mu': 128,
                              "hidden_g": 12, 'hidden_tau': 128, 'epochs': 100, 'binary': False, 'n_fc_mu': 2,
                              'val_split': 0.1, 'n_fc_g': 3, 'n_fc_tau': 3, 'verbose': 0, 'kernel_init': 'RandomNormal'}

    params_RLearner_IHDP_b = {'dataset_name': "ihdp_b", 'num': 100, 'lr': 1e-3, 'patience': 40, 'batch_size_g_mu': 256,
                              'batch_size_r': 32, 'reg_l2': .01, 'activation': 'linear', 'hidden_mu': 128,
                              "hidden_g": 12, 'hidden_tau': 128, 'epochs': 100, 'binary': False, 'n_fc_mu': 2,
                              'val_split': 0.1, 'n_fc_g': 3, 'n_fc_tau': 3, 'verbose': 0, 'kernel_init': 'RandomNormal'}

    params_RLearner_ACIC = {'dataset_name': "acic", 'num': 10,'lr': 1e-3, 'patience': 40, 'batch_size_g_mu': 256,
                            'batch_size_r': 128, 'reg_l2': .01, 'activation': 'linear', 'hidden_mu': 128, "hidden_g": 12,
                            'hidden_tau': 128, 'epochs': 300, 'binary': False, 'n_fc_mu': 3, 'n_fc_g': 4, 'n_fc_tau': 4,
                            'val_split': 0.1, 'verbose': 0, 'kernel_init': 'GlorotNormal'}

    # mean Policy Risk test: 0.2387| std Policy Risk test: 0.0155
    params_RLearner_JOBS = {'dataset_name': "jobs", 'num': 100, 'lr': 1e-4, 'patience': 40, 'batch_size_g_mu': 512,
                            'batch_size': 32, 'reg_l2': .001, 'activation': 'linear', 'hidden_mu_tau': 128,
                            'hidden_g': 64, 'epochs': 300, 'binary': True, 'n_fc_mu': 3, 'n_fc_tau': 6,
                            'val_split': 0.1,  'verbose': 0, 'kernel_init': 'GlorotNormal'}

    """XLEARNER"""

    params_XLearner_IHDP_a = {'dataset_name': "ihdp_a", 'num': 100, 'lr': 1e-3, 'patience': 40, 'batch_size_g_mu': 256,
                              'batch_size': 673, 'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 128,
                              'epochs': 300, 'binary': False, 'n_fc': 7, 'kernel_init': 'RandomNormal',
                              'n_fc_g': 2, "hidden_g": 12, 'hidden_d': 128, 'model_name': 'xlearner',
                              'params': 'ihdp_a4', 'val_split': 0.0, 'verbose': 0}

    params_XLearner_IHDP_b = {'dataset_name': "ihdp_b", 'num': 100, 'lr': 1e-2, 'patience': 40, 'batch_size_g_mu': 256,
                              'batch_size': 256, 'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 128,
                              'epochs': 200, 'binary': False, 'n_fc': 3, 'kernel_init': 'RandomNormal',
                              'n_fc_g': 3, "hidden_g": 12, 'hidden_d': 128, 'verbose': 0}

    params_XLearner_ACIC = {'dataset_name': "acic", 'num': 100, 'lr': 1e-3, 'patience': 40, 'batch_size_g_mu': 256,
                            'batch_size': 673, 'reg_l2': .01, 'activation': None, 'hidden_phi': 256,
                            'epochs': 1000, 'binary': False, 'n_fc': 5, 'verbose': 0, 'kernel_init': 'RandomNormal',
                            'val_split': 0.0}

    params_XLearner_JOBS = {'dataset_name': "jobs", 'num': 100, 'lr': 1e-3, 'patience': 40, 'batch_size_g_mu': 256,
                            'batch_size': 673, 'reg_l2': .01, 'activation': None, 'hidden_phi': 256,
                            'epochs': 1000, 'binary': True, 'n_fc': 5, 'verbose': 0, 'kernel_init': 'GlorotNormal'}

    """TARNET"""

    params_TARnet_IHDP_a = {'dataset_name': "ihdp_a", 'num': 100, 'lr': 1e-4, 'batch_size': 64,'epochs': 300,
                            'reg_l2': .01, 'activation': 'linear', 'binary': False, 'verbose': 0,
                            'kernel_init': 'RandomNormal',  'model_name': 'tarnet',
                            'optimizer': 'sgd'}

    #
    # params_TARnet_IHDP_a = {'dataset_name': "ihdp_a", 'num': 100, 'lr': 1e-4, 'patience': 40, 'batch_size': 64,
    #                         'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 208, 'hidden_y1': 480, 'hidden_y0': 416,
    #                         'epochs': 300, 'binary': False, 'n_fc': 10, 'n_hidden_1': 1, 'n_hidden_0': 4, 'verbose': 0,
    #                         'kernel_init': 'RandomNormal', 'params': 'params_ihdp_52', 'model_name': 'tarnet',
    #                         'optimizer': 'sgd'}


    params_TARnet_IHDP_b = {'dataset_name': "ihdp_b", 'num': 100, 'lr': 1e-4, 'batch_size': 32,'epochs': 300,
                            'reg_l2': .01, 'activation': 'linear', 'binary': False, 'verbose': 0,
                            'kernel_init': 'GlorotNormal', 'model_name': 'tarnet',
                            'optimizer': 'sgd'}


    params_TARnet_ACIC = {'dataset_name': "acic", 'num': 77, 'lr': 1e-3, 'patience': 40, 'batch_size': 256,
                          'reg_l2': .01, 'activation': 'linear', 'epochs': 300, 'binary': False,  'verbose': 0,
                          'kernel_init': 'RandomNormal', 'model_name': 'tarnet'}

    params_TARnet_JOBS = {'dataset_name': "jobs", 'num': 100, 'lr': 1e-3,  'batch_size': 128,
                          'reg_l2': .01,  'activation': 'sigmoid', 'epochs': 30, 'binary': True,
                          'verbose': 0, 'kernel_init': 'GlorotNormal', 'optimizer': 'sgd', 'model_name': 'tarnet'}

    """DRAGONNET"""

    params_DragonNet_IHDP_a = {'dataset_name': "ihdp_a", 'num': 100, 'lr': 1e-4, 'patience': 40, 'batch_size': 64,
                               'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 208, 'hidden_y1': 480,
                               'hidden_y0': 416, 'epochs': 300, 'binary': False, 'n_fc': 8, 'n_hidden_1': 4,
                               'n_hidden_0': 10, 'verbose': 0, 'kernel_init': 'RandomNormal', 'params': 'params_ihdp_a',
                               'hidden_t': 25, 'n_hidden_t': 1, 'val_split': 0.1}

    params_DragonNet_IHDP_b = {'dataset_name': "ihdp_b", 'num': 100, 'lr': 1e-4, 'patience': 5, 'batch_size': 32,
                               'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 240, 'hidden_y1': 400,
                               'hidden_y0': 240, 'epochs': 300, 'binary': False, 'n_fc': 4, 'n_hidden_1': 2,
                               'n_hidden_0': 5, 'verbose': 0, 'kernel_init': 'GlorotNormal', 'hidden_t': 25,
                               'n_hidden_t': 1, 'val_split': 0.1}

    params_DragonNet_ACIC = {'dataset_name': "acic", 'num': 77, 'lr': 1e-4, 'patience': 40, 'batch_size': 256,
                             'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 64, 'hidden_y1': 32, 'hidden_y0': 32,
                             'hidden_t': 32, 'epochs': 300, 'binary': False, 'n_fc': 3, 'n_hidden_1': 2,
                             'n_hidden_0': 2, 'n_hidden_t': 2, 'val_split': 0.1, 'verbose': 0, 
                             'kernel_init': 'RandomNormal'}

    # mean Policy Risk test: 0.2369 | std Policy Risk test: 0.0159
    params_DragonNet_JOBS = {'dataset_name': "jobs", 'num': 100, 'lr': 1e-3, 'patience': 40, 'batch_size': 256,
                             'reg_l2': .01, 'activation': 'sigmoid', 'hidden_phi': 128, 'hidden_y1': 128,
                             'hidden_y0': 128, 'hidden_t': 32, 'epochs': 30, 'binary': True, 'n_fc': 5, 'n_hidden_1': 3,
                             'n_hidden_0': 3, 'n_hidden_t': 3, 'val_split': 0.1, 'verbose': 0,
                             'kernel_init': 'RandomNormal'}

    """CEVAE"""

    params_CEVAE_IHDP_a = {'dataset_name': "ihdp_a", 'num': 100, 'num_bin': 19, 'num_cont': 6, 'lr': 1e-3,
                           'patience': 40, 'batch_size': 64, 'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 200,
                           'latent_dim': 20, 'epochs': 300, 'binary': False, 'n_fc': 4, 'val_split': 0.1, 'verbose': 0,
                           'kernel_init': 'RandomNormal'}

    params_CEVAE_IHDP_b = {'dataset_name': "ihdp_b", 'num': 100, 'num_bin': 19, 'num_cont': 6, 'lr': 1e-3,
                           'patience': 40, 'batch_size': 523, 'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 200,
                           'latent_dim': 20, 'epochs': 400, 'binary': False, 'n_fc': 4, 'val_split': 0.1, 'verbose': 0,
                           'kernel_init': 'GlorotNormal'}

    params_CEVAE_ACIC = {'dataset_name': "acic", 'num': 77,  'num_bin': 55, 'num_cont': 0, 'lr': 1e-3, 'patience': 40,
                         'batch_size': 400, 'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 200, 'latent_dim': 20,
                         'epochs': 1000, 'binary': False, 'n_fc': 5, 'val_split': 0.1, 'verbose': 0,
                         'kernel_init': 'RandomNormal'}

    params_CEVAE_JOBS = {'dataset_name': "jobs", 'num': 100, 'lr': 1e-3, 'num_bin': 0, 'num_cont': 17, 'patience': 40,
                         'batch_size': 1024, 'reg_l2': .01, 'activation': 'sigmoid', 'hidden_phi': 25, 'latent_dim': 20,
                         'epochs': 30, 'binary': True, 'n_fc': 3, 'val_split': 0.1, 'verbose': 0,
                         'kernel_init': 'GlorotNormal'}

    """TEDVAE"""

    params_TEDVAE_IHDP_a = {'dataset_name': "ihdp_a", 'num': 100, 'lr': 1e-4, 'patience': 40, 'batch_size': 673,
                            'reg_l2': .01, 'activation': 'linear', 'hp_hidden_phi_enc_x': 496, 'latent_dim_z': 15,
                            'num_bin': 19,'num_cont': 6, 'hp_hidden_phi_dec_x': 448, 'latent_dim_zt': 15,
                            'latent_dim_zy': 5, 'epochs': 400, 'binary': False, 'hp_fc_enc_x': 8, 'hp_fc_dec_x': 6,
                            'verbose': 0, 'kernel_init': 'RandomNormal', 'model_name': 'tedvae'}

    params_TEDVAE_IHDP_b = {'dataset_name': "ihdp_b", 'num': 100, 'lr': 1e-3, 'patience': 40, 'batch_size': 673,
                            'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 32, 'latent_dim_z': 15, 'num_bin': 19,
                            'num_cont': 6, 'latent_dim_zt': 15, 'latent_dim_zy': 5, 'epochs': 400, 'binary': False,
                            'n_fc': 5, 'verbose': 0, 'val_split': 0.1, 'kernel_init': 'GlorotNormal'}

    params_TEDVAE_ACIC = {'dataset_name': "acic", 'num': 77, 'lr': 1e-3, 'patience': 25, 'batch_size': 32,
                          'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 200, 'latent_dim_z': 15, 'num_bin': 55,
                          'num_cont': 0, 'latent_dim_zt': 15, 'latent_dim_zy': 5, 'epochs': 300, 'binary': False,
                          'n_fc': 5, 'val_split': 0.1, 'verbose': 0, 'kernel_init': 'RandomNormal'}

    params_TEDVAE_JOBS = {'dataset_name': "jobs", 'num': 100, 'lr': 1e-3, 'patience': 40, 'batch_size': 1024,
                          'reg_l2': .01, 'activation': 'sigmoid', 'hidden_phi': 32, 'latent_dim_z': 15, 'num_bin': 0,
                          'num_cont': 17, 'latent_dim_zt': 15, 'latent_dim_zy': 5, 'epochs': 30, 'binary': True,
                          'n_fc': 5, 'val_split': 0.1, 'verbose': 0, 'kernel_init': 'GlorotNormal'}

    """CFRNET"""

    # 0.368 +- 0.039 (tuned with batch_size = 256 but trained with 1024)
    params_CFRNet_IHDP_a = {'dataset_name': "ihdp_a", 'num': 100, 'lr': 1e-4, 'patience': 40, 'batch_size': 1024,
                            'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 208, 'hidden_y1': 480,
                            'hidden_y0': 416, 'epochs': 300, 'binary': False, 'n_fc': 8, 'n_hidden_1': 4,
                            'n_hidden_0': 10, 'verbose': 0, 'kernel_init': 'RandomNormal', 'params': 'params_ihdp_a',
                            'val_split': 0.1}

    params_CFRNet_IHDP_b = {'dataset_name': "ihdp_b", 'num': 100, 'lr': 1e-4, 'patience': 5, 'batch_size': 256,
                            'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 200, 'hidden_y1': 200,
                            'hidden_y0': 200, 'epochs': 300, 'binary': False, 'n_fc': 6, 'n_hidden_1': 4,
                            'n_hidden_0': 4, 'verbose': 1, 'kernel_init': 'GlorotNormal', 'params': 'params_ihdp_b',
                            'val_split': 0.1}

    params_CFRNet_ACIC = {'dataset_name': "acic", 'num': 77, 'lr': 1e-3, 'patience': 40, 'batch_size': 512,
                          'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 25, 'hidden_y1': 25, 'hidden_y0': 25,
                          'hidden_t': 25, 'epochs': 500, 'binary': False, 'n_fc': 4, 'n_hidden_1': 5, 'n_hidden_0': 5,
                          'n_hidden_t': 5, 'val_split': 0.1, 'verbose': 0, 'kernel_init': 'GlorotNormal'}

    params_CFRNet_JOBS = {'dataset_name': "jobs", 'num': 100, 'lr': 1e-3, 'patience': 40, 'batch_size': 1024,
                          'reg_l2': .01, 'activation': 'sigmoid', 'hidden_phi': 25, 'hidden_y1': 25, 'hidden_y0': 25,
                          'hidden_t': 25, 'epochs': 30, 'binary': True, 'n_fc': 3, 'n_hidden_1': 2, 'n_hidden_0': 2,
                          'n_hidden_t': 2, 'val_split': 0.1, 'verbose': 0, 'kernel_init': 'GlorotNormal'}

    """BVNICE"""

    params_BVNICE_IHDP_a = {'dataset_name': "ihdp_a", 'n_fc': 6, 'hidden_phi': 64, 'patience': 40,
                            'activation': 'linear', 'verbose': 0, 'reg_l2': .01,
                            'num': 100, 'binary': False, 'batch_size': 673, 'lr': 1e-3, 'epochs': 300,
                            'val_split': 0.1, 'kernel_init': 'RandomNormal'}

    params_BVNICE_IHDP_b = {'dataset_name': "ihdp_b", 'n_fc': 6, 'hidden_phi': 64, 'patience': 40,
                            'activation': 'linear', 'verbose': 1, 'reg_l2': .01,
                            'num': 100, 'binary': False, 'batch_size': 538, 'lr': 1e-2, 'epochs': 300,
                            'val_split': 0.1, 'kernel_init': 'RandomNormal'}

    params_BVNICE_ACIC = {'dataset_name': "acic", 'n_fc': 6, 'hidden_phi': 64, 'patience': 40,
                          'activation': 'linear', 'verbose': 0, 'reg_l2': .01,
                          'num': 100, 'binary': False, 'batch_size': 538, 'lr': 1e-2, 'epochs': 300,
                          'val_split': 0.1, 'kernel_init': 'RandomNormal'}

    """GANITE"""

    params_GANITE_IHDP_a = {'dataset_name': "ihdp_a", 'num': 100, 'lr': 1e-4, 'patience': 40, 'batch_size': 64,
                            'reg_l2': .01, 'activation': 'linear', 'binary': False, 'n_fc': 2, 'hidden_phi': 8,
                            'n_hidden': 2, 'hidden_y': 8, 'epochs_g': 1000, 'verbose': 0, 'epochs_i': 500,
                            'val_split': 0.1, 'kernel_init': 'RandomNormal'}

    params_GANITE_IHDP_b = {'dataset_name': "ihdp_b", 'num': 100, 'lr': 1e-5, 'patience': 40, 'batch_size': 128,
                            'reg_l2': .01, 'activation': 'linear', 'binary': False, 'n_fc': 2, 'hidden_phi': 8,
                            'n_hidden': 2, 'hidden_y': 8, 'epochs_g': 1000, 'verbose': 0, 'epochs_i': 500,
                            'val_split': 0.1, 'kernel_init': 'GlorotNormal'}
    
    params_GANITE_ACIC = {'dataset_name': "acic", 'num': 77, 'lr': 1e-4, 'patience': 40, 'batch_size': 128,
                          'reg_l2': .01, 'activation': 'linear', 'binary': False, 'n_fc': 2, 'hidden_phi': 8,
                          'n_hidden': 2, 'hidden_y': 8, 'epochs_g': 1000, 'verbose': 0, 'epochs_i': 500,
                          'val_split': 0.1, 'kernel_init': 'RandomNormal'}

    # mean Policy Risk test: 0.2209 | std Policy Risk test: 0.0142
    params_GANITE_JOBS = {'dataset_name': "jobs", 'num': 100, 'lr': 1e-3, 'patience': 40, 'batch_size': 64,
                          'reg_l2': .01, 'activation': 'linear', 'binary': True, 'n_fc': 4, 'hidden_phi': 64,
                          'n_hidden': 2, 'hidden_y': 8, 'epochs_g': 1000, 'verbose': 0, 'epochs_i': 500,
                          'val_split': 0.1, 'kernel_init': 'RandomNormal'}

    """DKLITE"""

    params_DKLITE_IHDP_a = {'dataset_name': "ihdp_a", 'num': 100, 'lr': 1e-3, 'patience': 40, 'batch_size': 128,
                            'reg_l2': .01,  'activation': 'linear', 'hidden_phi_encoder': 80, 'hidden_phi_decoder': 416,
                            'dim_z': 80, 'epochs': 300, 'binary': False, 'n_fc_encoder': 7, 'n_fc_decoder': 6,
                            'reg_var': 1.0, 'reg_rec': 0.7, 'verbose': 0, 'kernel_init': 'RandomNormal',
                            'model_name': 'dklite', 'x_size': 25}

    params_DKLITE_IHDP_b = {'dataset_name': "ihdp_b", 'num': 100, 'lr': 1e-2, 'patience': 25, 'batch_size': 32,
                            'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 50, 'dim_z': 50, 'epochs': 300,
                            'binary': False, 'n_fc': 2, 'reg_var': 1.0, 'reg_rec': 0.7, 'verbose': 0, 'val_split': 0.1,
                            'x_size': 25, 'kernel_init': 'GlorotNormal'}

    params_DKLITE_ACIC = {'dataset_name': "acic", 'num': 77, 'lr': 1e-3, 'patience': 40, 'batch_size': 32,
                          'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 50, 'dim_z': 50, 'epochs': 300,
                          'binary': False, 'n_fc': 3, 'reg_var': 1.0, 'reg_rec': 0.7, 'verbose': 0, 'val_split': 0.1,
                          'kernel_init': 'RandomNormal', 'x_size': 55}

    params_DKLITE_JOBS = {'dataset_name': "jobs", 'num': 100, 'lr': 1e-3, 'patience': 40, 'batch_size': 32,
                          'reg_l2': .01, 'activation': 'sigmoid', 'hidden_phi': 50, 'dim_z': 50, 'epochs': 30,
                          'binary': True, 'n_fc': 3, 'reg_var': 1.0, 'reg_rec': 0.7, 'verbose': 0, 'val_split': 0.1,
                          'kernel_init': 'RandomNormal', 'x_size': 17}


    """-------------------------------------------------------------"""

    params_TARnet = {'ihdp_a': params_TARnet_IHDP_a, 'ihdp_b': params_TARnet_IHDP_b, 'acic': params_TARnet_ACIC,
                     'jobs': params_TARnet_JOBS}

    params_CEVAE = {'ihdp_a': params_CEVAE_IHDP_a, 'ihdp_b': params_CEVAE_IHDP_b, 'acic': params_CEVAE_ACIC,
                    'jobs': params_CEVAE_JOBS}

    params_TEDVAE = {'ihdp_a': params_TEDVAE_IHDP_a, 'ihdp_b': params_TEDVAE_IHDP_b, 'acic': params_TEDVAE_ACIC,
                     'jobs': params_TEDVAE_JOBS}

    params_DKLITE = {'ihdp_a': params_DKLITE_IHDP_a, 'ihdp_b': params_DKLITE_IHDP_b, 'acic': params_DKLITE_ACIC,
                     'jobs': params_DKLITE_JOBS}

    params_GANITE = {'ihdp_a': params_GANITE_IHDP_a, 'ihdp_b': params_GANITE_IHDP_b, 'acic': params_GANITE_ACIC,
                     'jobs': params_GANITE_JOBS}

    params_DragonNet = {'ihdp_a': params_DragonNet_IHDP_a, 'ihdp_b': params_DragonNet_IHDP_b,
                        'acic': params_DragonNet_ACIC, 'jobs': params_DragonNet_JOBS}

    params_TLearner = {'ihdp_a': params_TLearner_IHDP_a, 'ihdp_b': params_TLearner_IHDP_b, 'acic': params_TLearner_ACIC,
                       'jobs': params_TLearner_JOBS}

    params_SLearner = {'ihdp_a': params_SLearner_IHDP_a, 'ihdp_b': params_SLearner_IHDP_b, 'acic': params_SLearner_ACIC,
                       'jobs': params_SLearner_JOBS}

    params_RLearner = {'ihdp_a': params_RLearner_IHDP_a, 'ihdp_b': params_RLearner_IHDP_b, 'acic': params_RLearner_ACIC,
                       'jobs': params_RLearner_JOBS}

    params_XLearner = {'ihdp_a': params_XLearner_IHDP_a, 'ihdp_b': params_XLearner_IHDP_b, 'acic': params_XLearner_ACIC,
                       'jobs': params_XLearner_JOBS}

    params_CFRNet = {'ihdp_a': params_CFRNet_IHDP_a, 'ihdp_b': params_CFRNet_IHDP_b, 'acic': params_CFRNet_ACIC,
                     'jobs': params_CFRNet_JOBS}

    params_BVNICE = {'ihdp_a': params_BVNICE_IHDP_a, 'ihdp_b': params_BVNICE_IHDP_b, 'acic': params_BVNICE_ACIC}

    """-------------------------------------------------------------"""

    params = {'TARnet': params_TARnet, 'CEVAE': params_CEVAE, 'TEDVAE': params_TEDVAE, 'DKLITE': params_DKLITE, 'DragonNet': params_DragonNet,
              'TLearner': params_TLearner, 'SLearner': params_SLearner, 'RLearner': params_RLearner, 'XLearner': params_XLearner,
              'CFRNet': params_CFRNet, 'GANITE': params_GANITE, 'BVNICE': params_BVNICE}

    return params[model_name][dataset_name]
