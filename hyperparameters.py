import tensorflow as tf
import numpy as np
import random

def find_params(model_name, dataset_name):
    # mean PEHE test: 0.7415817851802669 | std PEHE test: 0.25842569006094
    """BVNICE"""
    params_BVNICE_IHDP_a = {'dataset_name': "ihdp_a",
                            'n_fc': 7, 'hidden_phi': 64,
                            'activation': 'linear', 'verbose': 0, 'reg_l2': .01,
                            'num': 100, 'binary': False, 'batch_size': 128, 'lr': 1e-3, 'epochs': 1000,
                            'kernel_init': 'RandomNormal'}

    params_BVNICE_IHDP_b = {'dataset_name': "ihdp_b",
                            'n_fc': 2, 'hidden_phi': 128,
                            'activation': 'linear', 'verbose': 0, 'reg_l2': .01,
                            'num': 100, 'binary': False, 'batch_size': 128, 'lr': 1e-2, 'epochs': 1000,
                            'kernel_init': 'GlorotNormal'}
    """TARNET"""
    tf.random.set_seed(0)
    np.random.seed(0)
    random.seed(0)
    params_TARnet_IHDP_a = {'dataset_name': "ihdp_a", 'num': 100, 'lr': 1e-5, 'patience': 40, 'batch_size': 32,
                            'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 200, 'hidden_y1': 100, 'hidden_y0': 100,
                            'epochs': 300, 'binary': False, 'n_fc': 4, 'n_hidden_1': 3, 'n_hidden_0': 3, 'verbose': 0,
                            'kernel_init': 'RandomNormal'}

    params_TARnet_IHDP_a = {'dataset_name': "ihdp_a", 'num': 100, 'lr': 1e-4, 'patience': 40, 'batch_size': 64,
                            'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 208, 'hidden_y1': 416, 'hidden_y0': 480,
                            'epochs': 300, 'binary': False, 'n_fc': 8, 'n_hidden_1': 10, 'n_hidden_0': 4, 'verbose': 0,
                            'kernel_init': 'RandomNormal', 'params': 'params_ihdp_52', 'model_name': 'tarnet'}

    params_TARnet_IHDP_a = {'dataset_name': "ihdp_a", 'num': 100, 'lr': 1e-4, 'patience': 40, 'batch_size': 64,
                            'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 208, 'hidden_y1': 480, 'hidden_y0': 416,
                            'epochs': 300, 'binary': False, 'n_fc': 10, 'n_hidden_1': 1, 'n_hidden_0': 4, 'verbose': 0,
                            'kernel_init': 'RandomNormal', 'params': 'params_ihdp_52', 'model_name': 'tarnet',
                            'optimizer': 'sgd'}

    # params_TARnet_IHDP_a = {'dataset_name': "ihdp_a", 'num': 100, 'lr': 1e-4, 'patience': 40, 'batch_size': 64,
    #                         'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 192, 'hidden_y1': 240,
    #                         'hidden_y0': 416,
    #                         'epochs': 300, 'binary': False, 'n_fc': 10, 'n_hidden_1': 1, 'n_hidden_0': 4, 'verbose': 0,
    #                         'kernel_init': 'RandomNormal', 'params': 'params_ihdp_52', 'model_name': 'tarnet'}
    #
    # params_TARnet_IHDP_a = {'dataset_name': "ihdp_a", 'num': 100, 'lr': 1e-4, 'patience': 40, 'batch_size': 64,
    #                         'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 240, 'hidden_y1': 400,
    #                         'hidden_y0': 240, 'optimizer': 'sgd',
    #                         'epochs': 300, 'binary': False, 'n_fc': 4, 'n_hidden_1': 2, 'n_hidden_0': 5, 'verbose': 0,
    #                         'kernel_init': 'RandomNormal', 'params': 'params_ihdp_52', 'model_name': 'tarnet'}

    # params_TARnet_IHDP_a = {'dataset_name': "ihdp_a", 'num': 100, 'lr': 1e-4, 'patience': 40, 'batch_size': 128,
    #                         'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 300, 'hidden_y1': 400, 'hidden_y0': 200,
    #                         'epochs': 300, 'binary': False, 'n_fc': 6, 'n_hidden_1': 3, 'n_hidden_0': 5, 'verbose': 0,
    #                         'kernel_init': 'RandomNormal', 'params': 'params_ihdp_52', 'model_name': 'tarnet'}

    params_TARnet_IHDP_b = {'dataset_name': "ihdp_b", 'num': 100, 'lr': 1e-4, 'patience': 5, 'batch_size': 32,
                            'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 240, 'hidden_y1': 400, 'hidden_y0': 240,
                            'epochs': 300, 'binary': False, 'n_fc': 4, 'n_hidden_1': 2, 'n_hidden_0': 5, 'verbose': 0,
                            'kernel_init': 'GlorotNormal', 'params': 'params_ihdp_b', 'model_name': 'tarnet',
                            'optimizer': 'sgd'}
    # params_TARnet_IHDP_b = {'dataset_name': "ihdp_b", 'num': 100, 'lr': 1e-4, 'patience': 5, 'batch_size': 32,
    #                         'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 240, 'hidden_y1': 400, 'hidden_y0': 240,
    #                         'epochs': 300, 'binary': False, 'n_fc': 3, 'n_hidden_1': 1, 'n_hidden_0': 5, 'verbose': 0,
    #                         'kernel_init': 'GlorotNormal'}

    params_TARnet_ACIC = {'dataset_name': "acic", 'num': 77, 'lr': 1e-3, 'patience': 40, 'batch_size': 256,
                            'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 256, 'hidden_y1': 160,
                            'hidden_y0': 336,
                            'epochs': 300, 'binary': False, 'n_fc': 10, 'n_hidden_1': 5, 'n_hidden_0': 9, 'verbose': 0,
                            'kernel_init': 'RandomNormal', 'params': 'params_acic_0', 'model_name': 'tarnet'}

    params_TARnet_TWINS = {'dataset_name': "twins", 'num': 10, 'lr': 1e-3, 'patience': 40, 'batch_size': 32,
                           'reg_l2': .01, 'activation': 'sigmoid', 'hidden_phi': 200, 'hidden_y1': 100,
                           'hidden_y0': 100, 'epochs': 10, 'binary': True, 'n_fc': 3, 'n_hidden_1': 1, 'n_hidden_0': 9,
                           'verbose': 0}

    params_TARnet_JOBS= {'dataset_name': "jobs", 'num': 100, 'lr': 1e-3, 'patience': 40, 'batch_size': 128,
                            'reg_l2': .01, 'activation': 'sigmoid', 'hidden_phi': 208, 'hidden_y1': 416, 'hidden_y0': 480,
                            'epochs': 100, 'binary': True, 'n_fc': 8, 'n_hidden_1': 10, 'n_hidden_0': 4, 'verbose': 0,
                            'kernel_init': 'GlorotNormal', 'params': 'params_jobs131', 'model_name': 'tarnet'}

    params_TARnet_JOBS= {'dataset_name': "jobs", 'num': 100, 'lr': 1e-3, 'patience': 40, 'batch_size': 128,
                            'reg_l2': .01, 'activation': 'sigmoid', 'hidden_phi': 16, 'hidden_y1': 256, 'hidden_y0': 464,
                            'epochs': 100, 'binary': True, 'n_fc': 2, 'n_hidden_1': 4, 'n_hidden_0': 4, 'verbose': 0,
                            'kernel_init': 'GlorotNormal', 'optimizer': 'adam', 'params': 'params_jobs131', 'model_name': 'tarnet'}

    """CEVAE"""

    params_CEVAE_IHDP_a = {'dataset_name': "ihdp_a", 'num': 100, 'num_bin': 19, 'num_cont': 6, 'lr': 1e-3,
                           'patience': 25, 'batch_size': 673, 'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 300,
                           'latent_dim': 20, 'epochs': 100, 'binary': False, 'n_fc': 3, 'verbose': 0, 'val_split': 0.0,
                           'kernel_init': 'RandomNormal'}

    params_CEVAE_IHDP_b = {'dataset_name': "ihdp_b", 'num': 100, 'num_bin': 19, 'num_cont': 6, 'lr': 1e-3,
                           'patience': 25, 'batch_size': 523, 'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 200,
                           'latent_dim': 20, 'epochs': 400, 'binary': False, 'n_fc': 7, 'verbose': 0, 'val_split': 0.0,
                           'kernel_init': 'GlorotNormal'}


    params_CEVAE_ACIC = {'dataset_name': "acic", 'num': 100,  'num_bin': 49, 'num_cont': 6, 'lr': 1e-3, 'patience': 40,
                         'batch_size': 32, 'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 25, 'latent_dim': 20,
                         'epochs': 300, 'binary': False, 'n_fc': 3, 'verbose': 0}

    params_CEVAE_TWINS = {'dataset_name': "twins", 'num': 10, 'num_bin': 0, 'num_cont': 30, 'lr': 1e-2, 'patience': 40,
                          'batch_size': 31, 'reg_l2': .01, 'activation': 'sigmoid', 'hidden_phi': 25, 'latent_dim': 20,
                          'epochs': 10, 'binary': True, 'n_fc': 3, 'verbose': 0}

    params_CEVAE_JOBS = {'dataset_name': "jobs", 'num': 100, 'lr': 1e-3, 'num_bin': 0, 'num_cont': 17, 'patience': 40,
                         'batch_size': 59, 'reg_l2': .01, 'activation': 'sigmoid', 'hidden_phi': 25, 'latent_dim': 20,
                         'epochs': 30, 'binary': True, 'n_fc': 3, 'verbose': 1}

    """TEDVAE"""

    params_TEDVAE_IHDP_a = {'dataset_name': "ihdp_a", 'num': 100, 'lr': 1e-4, 'patience': 40, 'batch_size': 673,
                            'reg_l2': .01, 'activation': 'linear', 'hidden_phi_encoder': 496, 'hidden_phi_decoder': 496,
                            'latent_dim_z': 15, 'num_bin': 19, 'n_fc_encoder': 8, 'n_fc_decoder': 6,
                            'num_cont': 6, 'latent_dim_zt': 15, 'latent_dim_zy': 5, 'epochs': 400, 'binary': False,
                            'n_fc': 5, 'verbose': 0, 'kernel_init': 'RandomNormal', 'model_name': 'tedvae'}

    params_TEDVAE_IHDP_b = {'dataset_name': "ihdp_b", 'num': 100, 'lr': 1e-3, 'patience': 25, 'batch_size': 673,
                            'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 32, 'latent_dim_z': 15, 'num_bin': 19,
                            'num_cont': 6, 'latent_dim_zt': 15, 'latent_dim_zy': 5, 'epochs': 300, 'binary': False,
                            'n_fc': 5, 'verbose': 0, 'kernel_init': 'GlorotNormal', 'model_name': 'tedvae'}

    params_TEDVAE_ACIC = {'dataset_name': "acic", 'num': 100, 'lr': 1e-3, 'patience': 40, 'batch_size': 673,
                          'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 32, 'latent_dim_z': 15, 'num_bin': 49,
                          'num_cont': 6, 'latent_dim_zt': 15, 'latent_dim_zy': 5, 'epochs': 300, 'binary': False,
                          'n_fc': 5, 'verbose': 0}

    params_TEDVAE_TWINS = {'dataset_name': "twins", 'num': 10, 'lr': 1e-3, 'patience': 40, 'batch_size': 673,
                           'reg_l2': .01, 'activation': 'sigmoid', 'hidden_phi': 32, 'latent_dim_z': 15, 'num_bin': 0,
                           'num_cont': 30, 'latent_dim_zt': 15, 'latent_dim_zy': 5, 'epochs': 10, 'binary': True,
                           'n_fc': 5, 'verbose': 0}



    params_TEDVAE_JOBS = {'dataset_name': "jobs", 'num': 100, 'lr': 1e-3, 'patience': 40, 'batch_size': 673,
                          'reg_l2': .01, 'activation': 'sigmoid', 'hidden_phi': 32, 'latent_dim_z': 15, 'num_bin': 0,
                          'num_cont': 17, 'latent_dim_zt': 15, 'latent_dim_zy': 5, 'epochs': 30, 'binary': True,
                          'n_fc': 5, 'verbose': 1}

    """DKLITE"""




    params_DKLITE_IHDP_a = {'dataset_name': "ihdp_a", 'num': 100, 'lr': 1e-3, 'patience': 40, 'batch_size': 128,
                            'reg_l2': .01,  'activation': 'linear', 'hidden_phi_encoder': 80, 'hidden_phi_decoder': 416,
                            'dim_z': 80, 'epochs': 200,
                            'binary': False, 'n_fc_encoder': 7, 'n_fc_decoder': 6, 'reg_var': 1.0, 'reg_rec': 0.7, 'verbose': 0,
                            'kernel_init': 'RandomNormal', 'model_name': 'dklite', 'params': 'params_ihdp_10',
                            'x_size': 25
                            }
    params_DKLITE_IHDP_a = {'dataset_name': "ihdp_a", 'num': 100, 'lr': 1e-4, 'patience': 40, 'batch_size': 128,
                            'reg_l2': .01,  'activation': 'linear', 'hidden_phi_encoder': 80, 'hidden_phi_decoder': 416,
                            'dim_z': 80, 'epochs': 200,
                            'binary': False, 'n_fc_encoder': 7, 'n_fc_decoder': 6, 'reg_var': 1.0, 'reg_rec': 0.7, 'verbose': 0,
                            'kernel_init': 'RandomNormal', 'model_name': 'dklite', 'params': 'params_ihdp20',
                            'x_size': 25
                            }
    # params_DKLITE_IHDP_a = {'dataset_name': "ihdp_a", 'num': 100, 'lr': 1e-3, 'patience': 40, 'batch_size': 32,
    #                         'reg_l2': .01,  'activation': 'linear', 'hidden_phi': 50, 'dim_z': 50, 'epochs': 200,
    #                         'binary': False, 'n_fc': 5, 'reg_var': 1.0, 'reg_rec': 0.7, 'verbose': 0,
    #                         'kernel_init': 'RandomNormal', 'params': 'params_ihdp_13', 'x_size': 25
    #                         }
    # params_DKLITE_IHDP_a = {'dataset_name': "ihdp_a", 'num': 100, 'lr': 1e-4, 'patience': 40, 'batch_size': 128,
    #                         'reg_l2': .01,  'activation': 'linear', 'hidden_phi_encoder': 512, 'hidden_phi_decoder': 32,
    #                         'dim_z': 50, 'epochs': 200,
    #                         'binary': False, 'n_fc_encoder': 20, 'n_fc_decoder': 6, 'reg_var': 1.0, 'reg_rec': 0.7, 'verbose': 0,
    #                         'kernel_init': 'RandomNormal', 'model_name': 'dklite', 'params': 'params_ihd',
    #                         'x_size': 25
    #                         }

    params_DKLITE_IHDP_b = {'dataset_name': "ihdp_b", 'num': 100, 'lr': 1e-2, 'patience': 25, 'batch_size': 32,
                            'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 50, 'dim_z': 50, 'epochs': 300,
                            'binary': False, 'n_fc': 2, 'reg_var': 1.0, 'reg_rec': 0.7, 'verbose': 0,
                            'kernel_init': 'GlorotNormal'
                            }
    params_DKLITE_IHDP_b = {'dataset_name': "ihdp_b", 'num': 100, 'lr': 1e-3, 'patience': 40, 'batch_size': 128,
                             'reg_l2': .01, 'activation': 'linear', 'hidden_phi_encoder': 240, 'hidden_phi_decoder': 336,
                            'dim_z': 50, 'epochs': 150, 'n_fc_decoder': 4,
                             'binary': False, 'n_fc_encoder': 4, 'reg_var': 1.0, 'reg_rec': 0.7, 'verbose': 0,
                             'kernel_init': 'GlorotNormal', 'model_name': 'dklite', 'params': 'params_ihdp_new1000',
                             'x_size': 25
                             }

    params_DKLITE_ACIC = {'dataset_name': "acic", 'num': 100, 'lr': 1e-5, 'patience': 40, 'batch_size': 32,
                          'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 50, 'dim_z': 50, 'epochs': 300,
                          'binary': False, 'n_fc': 3, 'reg_var': 1.0, 'reg_rec': 0.7, 'verbose': 0}

    params_DKLITE_TWINS = {'dataset_name': "twins", 'num': 10, 'lr': 1e-3, 'patience': 40, 'batch_size': 32,
                           'reg_l2': .01, 'activation': 'sigmoid', 'hidden_phi': 50, 'dim_z': 50, 'epochs': 300,
                           'binary': True, 'n_fc': 3, 'reg_var': 1.0, 'reg_rec': 0.7, 'verbose': 0}

    params_DKLITE_JOBS = {'dataset_name': "jobs", 'num': 100, 'lr': 1e-5, 'patience': 40, 'batch_size': 32,
                          'reg_l2': .01, 'activation': 'sigmoid', 'hidden_phi': 50, 'dim_z': 50, 'epochs': 300,
                          'binary': True, 'n_fc': 3, 'reg_var': 1.0, 'reg_rec': 0.7, 'verbose': 0}

    """GANITE"""

    params_GANITE_IHDP_a = {'dataset_name': "ihdp_a", 'num': 100, 'lr': 1e-3, 'patience': 40, 'batch_size': 256,
                            'reg_l2': .01, 'activation': 'linear', 'binary': False, 'n_fc': 2, 'hidden_phi': 8,
                            'n_hidden': 2, 'hidden_y': 8, 'epochs_g': 1000, 'verbose': 0, 'epochs_i': 500,
                            'kernel_init': 'RandomNormal', 'model_name': 'ganite'}

    # params_GANITE_IHDP_a = {'name': 'ganite', 'dataset_name': "ihdp_a", 'num': 1, 'lr': 1e-5, 'patience': 40, 'batch_size': 64,
    #  'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 400, 'hidden_y1': 400, 'hidden_y0': 400,
    #  'epochs': 100, 'binary': False, 'n_fc': 3, 'n_hidden_1': 3, 'n_hidden_0': 3, 'verbose': 0,
    #  'kernel_init': 'RandomNormal', 'n_hidden_d': 5, 'hidden_phi_d': 8}

    params_GANITE_IHDP_b = {'dataset_name': "ihdp_b", 'num': 100, 'lr': 1e-5, 'patience': 40, 'batch_size': 128,
                            'reg_l2': .01, 'activation': 'sigmoid', 'hidden_phi_g': 16, 'hidden_y1_g': 8,
                            'hidden_y0_g': 8, 'epochs_g': 10, 'binary': False, 'n_fc_g': 2, 'n_hidden_1_g': 2,
                            'n_hidden_0_g': 2, 'verbose': 0, 'n_hidden_d': 2, 'hidden_phi_d': 8, 'n_fc_i': 2,
                            'hidden_phi_i': 2, 'n_hidden_0_i': 2, 'hidden_y0_i': 8, 'n_hidden_1_i': 2, 'hidden_y1_i': 8,
                            'epochs_i': 20}

    params_GANITE_IHDP_b = {'dataset_name': "ihdp_b", 'num': 100, 'lr': 1e-3, 'patience': 500, 'batch_size': 64,
                            'reg_l2': .01, 'activation': 'linear', 'binary': False, 'n_fc': 2, 'hidden_phi': 8,
                            'n_hidden': 2, 'hidden_y': 8, 'epochs_g': 500, 'verbose': 0, 'epochs_i': 700,
                            'kernel_init': 'RandomNormal', 'model_name': 'ganite'
                            }


    params_GANITE_ACIC = {'dataset_name': "acic", 'num': 100, 'lr': 1e-5, 'patience': 40, 'batch_size': 128,
                            'reg_l2': .01, 'activation': 'sigmoid', 'hidden_phi_g': 16, 'hidden_y1_g': 8,
                            'hidden_y0_g': 8, 'epochs_g': 10, 'binary': False, 'n_fc_g': 2, 'n_hidden_1_g': 2,
                            'n_hidden_0_g': 2, 'verbose': 0, 'n_hidden_d': 5, 'hidden_phi_d': 8, 'n_fc_i': 6,
                            'hidden_phi_i': 8, 'n_hidden_0_i': 2, 'hidden_y0_i': 8, 'n_hidden_1_i': 2, 'hidden_y1_i': 8,
                            'epochs_i': 20}
    params_GANITE_TWINS = {'dataset_name': "twins", 'num': 100, 'lr': 1e-4, 'patience': 40, 'batch_size': 64,
                            'reg_l2': .01, 'activation': 'linear', 'binary': True, 'n_fc': 2, 'hidden_phi': 8,
                            'n_hidden': 2, 'hidden_y': 8, 'epochs_g': 100, 'verbose': 1, 'epochs_i': 50,
                            'kernel_init': 'RandomNormal'}
    params_GANITE_JOBS = {'dataset_name': "jobs", 'num': 100, 'lr': 1e-5, 'patience': 40, 'batch_size': 128,
                           'reg_l2': .01, 'activation': 'sigmoid', 'hidden_phi_g': 16, 'hidden_y1_g': 8,
                           'hidden_y0_g': 8, 'epochs_g': 20, 'binary': True, 'n_fc_g': 2, 'n_hidden_1_g': 2,
                           'n_hidden_0_g': 2, 'verbose': 0, 'n_hidden_d': 5, 'hidden_phi_d': 8, 'n_fc_i': 6,
                           'hidden_phi_i': 8, 'n_hidden_0_i': 2, 'hidden_y0_i': 8, 'n_hidden_1_i': 2, 'hidden_y1_i': 8,
                           'epochs_i': 20}

    """DRAGONNET"""

    params_DragonNet_IHDP_a = {'dataset_name': "ihdp_a", 'num': 100, 'lr': 1e-4, 'patience': 40, 'batch_size': 64,
                               'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 200, 'hidden_y1': 100, 'hidden_y0': 100,
                               'hidden_t': 25, 'epochs': 300, 'binary': False, 'n_fc': 4, 'n_hidden_1': 3, 'n_hidden_t': 1,
                               'n_hidden_0': 3, 'verbose': 0,
                               'kernel_init': 'RandomNormal'}

    params_DragonNet_IHDP_b = {'dataset_name': "ihdp_b", 'num': 100, 'lr': 1e-4, 'patience': 40, 'batch_size': 64,
                               'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 200, 'hidden_y1': 100, 'hidden_y0': 100,
                               'hidden_t': 25, 'epochs': 300, 'binary': False, 'n_fc': 5, 'n_hidden_1': 2, 'n_hidden_t': 1,
                               'n_hidden_0': 2, 'verbose': 0,
                               'kernel_init': 'GlorotNormal'}

    params_DragonNet_ACIC = {'dataset_name': "acic", 'num': 100, 'lr': 1e-5, 'patience': 40, 'batch_size': 64,
                             'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 200, 'hidden_y1': 100, 'hidden_y0': 100,
                             'hidden_t': 25, 'epochs': 300, 'binary': False, 'n_fc': 3, 'n_hidden_1': 2, 'n_hidden_0': 1,
                             'n_hidden_t': 1, 'verbose': 0}

    params_DragonNet_TWINS = {'dataset_name': "twins", 'num': 10, 'lr': 1e-5, 'patience': 40, 'batch_size': 64,
                              'reg_l2': .01, 'activation': 'sigmoid', 'hidden_phi': 200, 'hidden_y1': 100,
                              'hidden_y0': 100,
                              'hidden_t': 1, 'epochs': 10, 'binary': True, 'n_fc': 3, 'n_hidden_1': 2, 'n_hidden_0': 2,
                              'n_hidden_t': 1, 'verbose': 0}

    params_DragonNet_JOBS = {'dataset_name': "jobs", 'num': 100, 'lr': 1e-5, 'patience': 40, 'batch_size': 64,
                             'reg_l2': .01, 'activation': 'sigmoid', 'hidden_phi': 200, 'hidden_y1': 100, 'hidden_y0': 100,
                             'hidden_t': 25, 'epochs': 30, 'binary': True, 'n_fc': 3, 'n_hidden_1': 2, 'n_hidden_0': 2,
                             'n_hidden_t': 1, 'verbose': 1}

    """TLEARNER"""

    params_TLearner_IHDP_a = {'dataset_name': "ihdp_a", 'num': 100, 'lr_0': 1e-2, 'lr_1': 1e-3, 'patience': 40, 'batch_size_0': 64,
                              'batch_size_1': 64, 'reg_l2': .01, 'activation': 'linear', 'hidden_phi_0': 96,
                              'hidden_phi_1': 32, 'epochs': 300, 'binary': False, 'n_fc_0': 4, 'n_fc_1': 6, 'verbose': 0,
                              'params': 'ihdp_tlearner_a_12',
                              'model_name_0': 'TLearner0',  'model_name_1': 'TLearner1', 'model_name': 'TLearner',
                              'kernel_init': 'RandomNormal'}


    params_TLearner_IHDP_b = {'dataset_name': "ihdp_b", 'num': 100, 'lr_0': 1e-3, 'lr_1': 1e-3, 'patience': 40, 'batch_size_0': 512,
                              'batch_size_1': 512, 'reg_l2': .01, 'activation': 'linear', 'hidden_phi_0': 128,
                              'hidden_phi_1': 128, 'epochs': 1300, 'binary': False, 'n_fc_0': 7, 'n_fc_1': 7, 'verbose': 0,
                              'params': 'ihdp_tlearner_a_12',
                              'model_name_0': 'TLearner0',  'model_name_1': 'TLearner1', 'model_name': 'TLearner',
                              'kernel_init': 'GlorotNormal'}

    params_TLearner_IHDP_b = {'dataset_name': "ihdp_b", 'num': 100, 'lr_0': 1e-3, 'lr_1': 1e-3, 'patience': 40, 'batch_size_0': 512,
                              'batch_size_1': 512, 'reg_l2': .01, 'activation': 'linear', 'hidden_phi_0': 128,
                              'hidden_phi_1': 128, 'epochs': 300, 'binary': False, 'n_fc_0': 7, 'n_fc_1': 7, 'verbose': 0,
                              'params': 'ihdp_tlearner_a_12',
                              'model_name_0': 'TLearner0',  'model_name_1': 'TLearner1', 'model_name': 'TLearner',
                              'kernel_init': 'GlorotNormal'}

    params_TLearner_ACIC = {'dataset_name': "acic", 'num': 100, 'lr': 1e-3, 'patience': 40, 'batch_size_g_mu': 256,
                            'batch_size': 673, 'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 256,
                            'epochs': 1000, 'binary': False, 'n_fc': 6, 'verbose': 0}

    params_TLearner_TWINS = {'dataset_name': "twins", 'num': 10, 'lr': 1e-3, 'patience': 40, 'batch_size_g_mu': 256,
                             'batch_size': 673, 'reg_l2': .001, 'activation': 'sigmoid', 'hidden_phi': 256,
                             'epochs': 1000, 'binary': True, 'n_fc': 6, 'verbose': 0}

    params_TLearner_JOBS =  {'dataset_name': "jobs", 'num': 100, 'lr_0': 1e-3, 'lr_1': 1e-3, 'patience': 40, 'batch_size_0': 512,
                              'batch_size_1': 512, 'reg_l2': .01, 'activation': 'sigmoid', 'hidden_phi_0': 64,
                              'hidden_phi_1': 64, 'epochs': 1000, 'binary': True, 'n_fc_0': 3, 'n_fc_1': 3, 'verbose': 0,
                              'params': 'ihdp_tlearner_a_12',
                              'model_name_0': 'TLearner0',  'model_name_1': 'TLearner1', 'model_name': 'TLearner',
                              'kernel_init': 'RandomNormal'}

    """SLEARNER"""
    # 0.44 +- 0.05
    params_SLearner_IHDP_a = {'dataset_name': "ihdp_a", 'num': 100, 'lr': 1e-3, 'patience': 40,
                              'batch_size': 64, 'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 256,
                              'epochs': 300, 'binary': False, 'n_fc': 7, 'verbose': 0,
                              'kernel_init': 'RandomNormal', 'model_name': 'slearner', 'params': 'params_ihdp_a_s6'}

    # 2.33 +- 0.06
    params_SLearner_IHDP_b = {'dataset_name': "ihdp_b", 'num': 100, 'lr': 1e-3, 'patience': 40,
                              'batch_size': 128, 'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 256,
                              'epochs': 400, 'binary': False, 'n_fc': 6, 'verbose': 0,
                              'kernel_init': 'GlorotNormal', 'model_name': 'slearner2', 'params': 'params_ihdp_b_32'}

    params_SLearner_ACIC = {'dataset_name': "acic", 'num': 100, 'lr': 1e-3, 'patience': 40, 'batch_size_g_mu': 256,
                            'batch_size': 673, 'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 128,
                            'epochs': 1000, 'binary': False, 'n_fc': 5, 'verbose': 0}

    params_SLearner_TWINS = {'dataset_name': "twins", 'num': 10, 'lr': 1e-3, 'patience': 40, 'batch_size_g_mu': 256,
                             'batch_size': 673, 'reg_l2': .001, 'activation': 'sigmoid', 'hidden_phi': 128,
                             'epochs': 1000, 'binary': True, 'n_fc': 5, 'verbose': 0}

    params_SLearner_JOBS = {'dataset_name': "jobs", 'num': 100, 'lr': 1e-3, 'patience': 40, 'batch_size_g_mu': 256,
                            'batch_size': 128, 'reg_l2': .01, 'activation': 'sigmoid', 'hidden_phi': 128,
                            'epochs': 1000, 'binary': True, 'n_fc': 3, 'verbose': 0, 'model_name': 'slearner2',
                            'params': 'params_jobs_2', 'kernel_init': 'GlorotNormal'}

    params_SLearner_JOBS = {'dataset_name': "jobs", 'num': 100, 'lr': 1e-3, 'patience': 40, 'batch_size_g_mu': 256,
                            'batch_size': 64, 'reg_l2': .01, 'activation': 'sigmoid', 'hidden_phi': 256,
                            'epochs': 300, 'binary': True, 'n_fc': 5, 'verbose': 0, 'model_name': 'slearner2',
                            'params': 'params_jobs_9', 'kernel_init': 'GlorotNormal'}

    # params_SLearner_JOBS = {'dataset_name': "jobs", 'num': 100, 'lr': 1e-4, 'patience': 40, 'batch_size_g_mu': 256,
    #                         'batch_size': 64, 'reg_l2': .01, 'activation': 'sigmoid', 'hidden_phi': 32,
    #                         'epochs': 300, 'binary': True, 'n_fc': 6, 'verbose': 0, 'model_name': 'slearner2',
    #                         'params': 'params_jobs_3', 'kernel_init': 'GlorotNormal'}

    """RLEARNER"""

    """RLEARNER"""

    params_RLearner_IHDP_a = {'dataset_name': "ihdp_a", 'num': 100, 'lr': 1e-3, 'patience': 10, 'batch_size_g_mu': 256,
                              'batch_size_r': 32, 'reg_l2': .01, 'activation': 'linear', 'hidden_mu': 128, "hidden_g": 12,
                              'hidden_tau': 128,
                              'epochs': 300, 'binary': False, 'n_fc_mu': 2, 'n_fc_g': 5, 'n_fc_tau': 5,
                              'verbose': 0, 'kernel_init': 'RandomNormal'}

    params_RLearner_IHDP_b = {'dataset_name': "ihdp_b", 'num': 100, 'lr': 1e-2, 'patience': 10, 'batch_size_g_mu': 256,
                              'batch_size_r': 32, 'reg_l2': .01, 'activation': 'linear', 'hidden_mu': 128, "hidden_g": 8,
                              'hidden_tau': 96,
                              'epochs': 200, 'binary': False, 'n_fc_mu': 2, 'n_fc_g': 6, 'n_fc_tau': 5,
                              'verbose': 0, 'kernel_init': 'GlorotNormal'}

    params_RLearner_ACIC = {'dataset_name': "acic", 'num': 100, 'lr': 1e-4, 'patience': 40, 'batch_size_g_mu': 256,
                            'batch_size': 32, 'reg_l2': .001, 'activation': None, 'hidden_phi': 128, "hidden_g": 12,
                            'epochs': 100, 'binary': False, 'n_fc': 4, 'verbose': 0}

    params_RLearner_TWINS = {'dataset_name': "twins", 'num': 10, 'lr': 1e-4, 'patience': 40, 'batch_size_g_mu': 256,
                             'batch_size': 32, 'reg_l2': .001, 'activation': None, 'hidden_phi': 128, "hidden_g": 12,
                             'epochs': 100, 'binary': True, 'n_fc': 4, 'verbose': 0}

    params_RLearner_JOBS = {'dataset_name': "jobs", 'num': 100, 'lr': 1e-4, 'patience': 40, 'batch_size_g_mu': 256,
                            'batch_size': 32, 'reg_l2': .001, 'activation': None, 'hidden_phi': 128, "hidden_g": 12,
                            'epochs': 100, 'binary': True, 'n_fc': 4, 'verbose': 0}

    """XLEARNER"""

    params_XLearner_IHDP_a = {'dataset_name': "ihdp_a", 'num': 100, 'lr': 1e-3, 'patience': 40, 'batch_size_g_mu': 256,
                              'batch_size': 673, 'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 128,
                              'epochs': 400, 'binary': False, 'n_fc': 7, 'kernel_init': 'RandomNormal',
                              'n_fc_g': 3, "hidden_g": 12, 'hidden_d': 128,
                              'verbose': 0}

    params_XLearner_IHDP_a = {'dataset_name': "ihdp_a", 'num': 100, 'lr': 1e-3, 'patience': 40, 'batch_size_g_mu': 256,
                              'batch_size': 673, 'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 128,
                              'epochs': 400, 'binary': False, 'n_fc': 7, 'kernel_init': 'RandomNormal',
                              'n_fc_g': 3, "hidden_g": 12, 'hidden_d': 128, 'model_name': 'xlearner',
                              'params': 'ihdp_a0', 'val_split': 0.0,
                              'verbose': 0}

    params_XLearner_IHDP_b = {'dataset_name': "ihdp_b", 'num': 100, 'lr': 1e-2, 'patience': 40, 'batch_size_g_mu': 256,
                              'batch_size': 256, 'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 128,
                              'epochs': 200, 'binary': False, 'n_fc': 3, 'kernel_init': 'RandomNormal',
                              'n_fc_g': 3, "hidden_g": 12, 'hidden_d': 128,
                              'verbose': 0}

    params_XLearner_ACIC = {'dataset_name': "acic", 'num': 100, 'lr': 1e-3, 'patience': 40, 'batch_size_g_mu': 256,
                            'batch_size': 673, 'reg_l2': .01, 'activation': None, 'hidden_phi': 256,
                            'epochs': 1000, 'binary': False, 'n_fc': 5, 'verbose': 0}

    params_XLearner_TWINS = {'dataset_name': "twins", 'num': 10, 'lr': 1e-3, 'patience': 40, 'batch_size_g_mu': 256,
                             'batch_size': 673, 'reg_l2': .001, 'activation': None, 'hidden_phi': 256,
                             'epochs': 1000, 'binary': True, 'n_fc': 5, 'verbose': 0}

    params_XLearner_JOBS = {'dataset_name': "jobs", 'num': 100, 'lr': 1e-3, 'patience': 40, 'batch_size_g_mu': 256,
                            'batch_size': 673, 'reg_l2': .01, 'activation': None, 'hidden_phi': 256,
                            'epochs': 1000, 'binary': True, 'n_fc': 5, 'verbose': 0}

    """CFRNET"""

    params_CFRNet_IHDP_a = {'dataset_name': "ihdp_a", 'num': 100, 'lr': 1e-3, 'patience': 40, 'batch_size': 600,
                            'reg_l2': .01, 'activation': None, 'hidden_phi': 64, 'hidden_y1': 256, 'hidden_y0': 64,
                            'hidden_t': 100, 'epochs': 300, 'binary': False, 'n_fc': 6  , 'n_hidden_1': 4, 'n_hidden_t': 5,
                            'n_hidden_0': 4, 'verbose': 0,
                            'kernel_init': 'RandomNormal', 'model_name': 'cfrnet'}


    params_CFRNet_IHDP_a = {'dataset_name': "ihdp_a", 'num': 100, 'lr': 1e-3, 'patience': 40, 'batch_size': 600,
                            'reg_l2': .01, 'activation': None, 'hidden_phi': 240, 'hidden_y1': 400, 'hidden_y0': 240,
                            'hidden_t': 240, 'epochs': 150, 'binary': False, 'n_fc': 3, 'n_hidden_1': 1, 'n_hidden_t': 5,
                            'n_hidden_0': 5, 'verbose': 0,
                            'kernel_init': 'RandomNormal', 'model_name': 'cfrnet'}


    # # best
    # params_CFRNet_IHDP_a = {'dataset_name': "ihdp_a", 'num': 100, 'lr': 1e-3, 'patience': 40, 'batch_size': 600,
    #                         'reg_l2': .01, 'activation': None, 'hidden_phi': 240, 'hidden_y1': 400, 'hidden_y0': 240,
    #                         'hidden_t': 240, 'epochs': 150, 'binary': False, 'n_fc': 4, 'n_hidden_1': 2, 'n_hidden_t': 8,
    #                         'n_hidden_0': 5, 'verbose': 0,
    #                         'kernel_init': 'RandomNormal', 'model_name': 'cfrnet'}

    params_CFRNet_IHDP_a = {'dataset_name': "ihdp_a", 'num': 100, 'lr': 1e-3, 'patience': 40, 'batch_size': 600,
                            'reg_l2': .01, 'activation': None, 'hidden_phi': 208, 'hidden_y1': 480, 'hidden_y0': 416,
                            'hidden_t': 24, 'epochs': 150, 'binary': False, 'n_fc': 8, 'n_hidden_1': 4,
                            'n_hidden_t': 8,
                            'n_hidden_0': 10, 'verbose': 0,
                            'kernel_init': 'RandomNormal', 'model_name': 'cfrnet1'}

    params_CFRNet_IHDP_a = {'dataset_name': "ihdp_a", 'num': 100, 'lr': 1e-3, 'patience': 40, 'batch_size': 32,
                            'reg_l2': .01, 'activation': None, 'hidden_phi': 240, 'hidden_y1': 400, 'hidden_y0': 240,
                            'hidden_t': 24, 'epochs': 150, 'binary': False, 'n_fc': 8, 'n_hidden_1': 4,
                            'n_hidden_t': 8,
                            'n_hidden_0': 10, 'verbose': 0,
                            'kernel_init': 'RandomNormal', 'model_name': 'cfrnet1'}

    params_CFRNet_IHDP_a = {'dataset_name': "ihdp_a", 'num': 100, 'lr': 1e-4, 'patience': 40, 'batch_size': 64,
                            'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 192, 'hidden_y1': 240,
                            'hidden_y0': 416, 'hidden_t': 24,     'n_hidden_t': 8,
                            'epochs': 300, 'binary': False, 'n_fc': 10, 'n_hidden_1': 1, 'n_hidden_0': 4, 'verbose': 1,
                            'kernel_init': 'RandomNormal', 'params': 'params_ihdp_52', 'model_name': 'cfrnet'}

    params_CFRNet_IHDP_b = {'dataset_name': "ihdp_b", 'num': 100, 'lr': 1e-3, 'patience': 10, 'batch_size': 256,
                            'reg_l2': .01, 'activation': None, 'hidden_phi': 240, 'hidden_y1': 400, 'hidden_y0': 240,
                            'hidden_t': 100, 'epochs': 300, 'binary': False, 'n_fc': 4, 'n_hidden_1': 2, 'n_hidden_t': 5,
                            'n_hidden_0': 5, 'verbose': 0, 'model_name': 'cfrnet',
                            'kernel_init': 'GlorotNormal'}

    params_CFRNet_ACIC = {'dataset_name': "acic", 'num': 100, 'lr': 1e-3, 'patience': 40, 'batch_size': 128,
                            'reg_l2': .01, 'activation': None, 'hidden_phi': 240, 'hidden_y1': 400, 'hidden_y0': 240,
                            'hidden_t': 24, 'epochs': 150, 'binary': False, 'n_fc': 8, 'n_hidden_1': 4,
                            'n_hidden_t': 8,
                            'n_hidden_0': 10, 'verbose': 0,
                            'kernel_init': 'RandomNormal', 'model_name': 'cfrnet8'}


    params_CFRNet_TWINS = {'dataset_name': "twins", 'num': 10, 'lr': 1e-3, 'patience': 40, 'batch_size': 600,
                           'reg_l2': .01, 'activation': 'sigmoid', 'hidden_phi': 25, 'hidden_y1': 25, 'hidden_y0': 25,
                           'hidden_t': 25, 'epochs': 10, 'binary': True, 'n_fc': 3, 'n_hidden_1': 2, 'n_hidden_0': 2,
                           'n_hidden_t': 2, 'verbose': 0}

    # params_CFRNet_JOBS = {'dataset_name': "jobs", 'num': 100, 'lr': 1e-3, 'patience': 40, 'batch_size': 32,
    #                         'reg_l2': .01, 'activation': None, 'hidden_phi': 240, 'hidden_y1': 400, 'hidden_y0': 240,
    #                         'hidden_t': 24, 'epochs': 150, 'binary': True, 'n_fc': 8, 'n_hidden_1': 4,
    #                         'n_hidden_t': 8,
    #                         'n_hidden_0': 10, 'verbose': 0,
    #                         'kernel_init': 'RandomNormal', 'model_name': 'cfrnet1'}

    params_CFRNet_JOBS = {'dataset_name': "jobs", 'num': 100, 'lr': 1e-3, 'patience': 40, 'batch_size': 32,
                          'reg_l2': .01, 'activation': 'sigmoid', 'hidden_phi': 16, 'hidden_y1': 256, 'hidden_y0': 464,
                          'hidden_t': 24, 'epochs': 150, 'binary': True, 'n_fc': 2, 'n_hidden_1': 4,
                          'n_hidden_t': 4,
                          'n_hidden_0': 10, 'verbose': 0,
                          'kernel_init': 'RandomNormal', 'model_name': 'cfrnet2'}

    """-------------------------------------------------------------"""

    params_TARnet = {'ihdp_a': params_TARnet_IHDP_a, 'ihdp_b': params_TARnet_IHDP_b, 'acic': params_TARnet_ACIC,
                     'twins': params_TARnet_TWINS, 'jobs': params_TARnet_JOBS}

    params_CEVAE = {'ihdp_a': params_CEVAE_IHDP_a, 'ihdp_b': params_CEVAE_IHDP_b, 'acic': params_CEVAE_ACIC,
                     'twins': params_CEVAE_TWINS, 'jobs': params_CEVAE_JOBS}

    params_TEDVAE = {'ihdp_a': params_TEDVAE_IHDP_a, 'ihdp_b': params_TEDVAE_IHDP_b, 'acic': params_TEDVAE_ACIC,
                     'twins': params_TEDVAE_TWINS, 'jobs': params_TEDVAE_JOBS}

    params_DKLITE = {'ihdp_a': params_DKLITE_IHDP_a, 'ihdp_b': params_DKLITE_IHDP_b, 'acic': params_DKLITE_ACIC,
                     'twins': params_DKLITE_TWINS, 'jobs': params_DKLITE_JOBS}

    params_GANITE = {'ihdp_a': params_GANITE_IHDP_a, 'ihdp_b': params_GANITE_IHDP_b, 'acic': params_GANITE_ACIC,
                     'twins': params_GANITE_TWINS, 'jobs': params_GANITE_JOBS}

    params_DragonNet = {'ihdp_a': params_DragonNet_IHDP_a, 'ihdp_b': params_DragonNet_IHDP_b,
                        'acic': params_DragonNet_ACIC, 'twins': params_DragonNet_TWINS, 'jobs': params_DragonNet_JOBS}

    params_TLearner = {'ihdp_a': params_TLearner_IHDP_a, 'ihdp_b': params_TLearner_IHDP_b, 'acic': params_TLearner_ACIC,
                       'twins': params_TLearner_TWINS, 'jobs': params_TLearner_JOBS}

    params_SLearner = {'ihdp_a': params_SLearner_IHDP_a, 'ihdp_b': params_SLearner_IHDP_b, 'acic': params_SLearner_ACIC,
                       'twins': params_SLearner_TWINS, 'jobs': params_SLearner_JOBS}

    params_RLearner = {'ihdp_a': params_RLearner_IHDP_a, 'ihdp_b': params_RLearner_IHDP_b, 'acic': params_RLearner_ACIC,
                       'twins': params_RLearner_TWINS, 'jobs': params_RLearner_JOBS}

    params_XLearner = {'ihdp_a': params_XLearner_IHDP_a, 'ihdp_b': params_XLearner_IHDP_b, 'acic': params_XLearner_ACIC,
                       'twins': params_XLearner_TWINS, 'jobs': params_XLearner_JOBS}

    params_CFRNet = {'ihdp_a': params_CFRNet_IHDP_a, 'ihdp_b': params_CFRNet_IHDP_b, 'acic': params_CFRNet_ACIC,
                     'twins': params_CFRNet_TWINS, 'jobs': params_CFRNet_JOBS}

    params_BVNICE= {'ihdp_a': params_BVNICE_IHDP_a, 'ihdp_b': params_BVNICE_IHDP_b}

    """-------------------------------------------------------------"""

    params = {'TARnet': params_TARnet, 'CEVAE': params_CEVAE, 'TEDVAE': params_TEDVAE, 'DKLITE': params_DKLITE, 'DragonNet': params_DragonNet,
              'TLearner': params_TLearner, 'SLearner': params_SLearner, 'RLearner': params_RLearner, 'XLearner': params_XLearner,
              'CFRNet': params_CFRNet, 'GANITE': params_GANITE, 'BVNICE': params_BVNICE}

    return params[model_name][dataset_name]