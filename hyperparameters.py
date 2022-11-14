
def find_params(model_name, dataset_name):

    """SLEARNER"""

    params_SLearner_IHDP_a = {'dataset_name': "ihdp_a", 'num': 100, 'lr': 1e-3, 'patience': 40,
                              'batch_size': 64, 'reg_l2': .01, 'activation': 'linear',
                              'epochs': 300, 'binary': False, 'n_fc': 9, 'verbose': 0, 'val_split': 0.0,
                              'kernel_init': 'RandomNormal', 'max_trials': 10, 'defaults': True,
                              'hp_fc': 3, 'hp_hidden_phi': 200}

    params_SLearner_IHDP_b = {'dataset_name': "ihdp_b", 'num': 100, 'lr': 1e-3, 'patience': 40,
                              'batch_size': 64, 'reg_l2': .01, 'activation': 'linear',
                              'epochs': 300, 'binary': False, 'n_fc': 6, 'verbose': 0, 'val_split': 0.0,
                              'kernel_init': 'GlorotNormal', 'max_trials': 30}

    params_SLearner_ACIC = {'dataset_name': "acic", 'num': 77, 'lr': 1e-3, 'patience': 40,
                            'batch_size': 256, 'reg_l2': .01, 'activation': 'linear',
                            'val_split': 0.0, 'epochs': 300, 'binary': False, 'n_fc': 7, 'verbose': 0, 
                            'kernel_init': 'RandomNormal', 'max_trials': 30}

    params_SLearner_JOBS = {'dataset_name': "jobs", 'num': 100, 'lr': 1e-3, 'patience': 40,
                            'batch_size': 128, 'reg_l2': .01, 'activation': 'sigmoid',
                            'val_split': 0.0, 'epochs': 50, 'binary': True, 'n_fc': 5, 'verbose': 0,
                            'kernel_init': 'RandomNormal', 'max_trials': 20}

    """TLEARNER"""

    params_TLearner_IHDP_a = {'dataset_name': "ihdp_a", 'num': 100, 'lr_0': 1e-2, 'lr_1': 1e-3, 'patience': 40,
                              'batch_size_0': 64, 'batch_size_1': 64, 'reg_l2': .01, 'activation': 'linear',
                              'epochs': 300, 'binary': False, 'verbose': 0, 'model_name': 'TLearner',
                              'kernel_init': 'RandomNormal', 'max_trials': 10}

    params_TLearner_IHDP_b = {'dataset_name': "ihdp_b", 'num': 100, 'lr_0': 1e-3, 'lr_1': 1e-3, 'patience': 40,
                              'batch_size_0': 512, 'batch_size_1': 512, 'reg_l2': .01, 'activation': 'linear',
                              'epochs': 1200, 'binary': False, 'verbose': 0, 'model_name': 'TLearner',
                              'kernel_init': 'GlorotNormal', 'max_trials': 10}

    params_TLearner_ACIC = {'dataset_name': "acic", 'num': 77, 'lr': 1e-2, 'patience': 40, 'max_trials': 10,
                            'batch_size_0': 128, 'batch_size_1': 128, 'reg_l2': .01, 'activation': 'linear',
                            'epochs': 1000, 'binary': False, 'verbose': 0, 'kernel_init': 'GlorotNormal'}

    params_TLearner_JOBS = {'dataset_name': "jobs", 'num': 100, 'lr': 1e-2, 'patience': 40,
                            'max_trials': 20, 'batch_size_0': 256, 'batch_size_1': 256, 'reg_l2': .01,
                            'activation': 'sigmoid', 'epochs': 30,  'binary': True, 'verbose': 0,
                            'kernel_init': 'RandomNormal'}

    """RLEARNER"""

    params_RLearner_IHDP_a = {'dataset_name': "ihdp_a", 'num': 100, 'lr': 1e-3, 'patience': 40, 'batch_size_g_mu': 256,
                              'batch_size_r': 32, 'reg_l2': .01, 'activation': 'linear','epochs': 100, 'binary': False,
                              'val_split': 0.0, 'verbose': 0, 'kernel_init': 'RandomNormal', 'max_trials': 20}

    params_RLearner_IHDP_b = {'dataset_name': "ihdp_b", 'num': 100, 'lr': 1e-3, 'patience': 40, 'batch_size_g_mu': 256,
                              'batch_size_r': 32, 'reg_l2': .01, 'activation': 'linear', 'epochs': 100, 'binary': False,
                              'val_split': 0.0, 'verbose': 0, 'kernel_init': 'RandomNormal', 'max_trials': 10}

    params_RLearner_ACIC = {'dataset_name': "acic", 'num': 10,'lr': 1e-3, 'patience': 40, 'batch_size_g_mu': 256,
                            'batch_size_r': 128, 'reg_l2': .01, 'activation': 'linear', 'epochs': 300, 'binary': False,
                            'val_split': 0.0, 'verbose': 0, 'kernel_init': 'GlorotNormal', 'max_trials': 10}

    params_RLearner_JOBS = {'dataset_name': "jobs", 'num': 100, 'lr': 1e-4, 'patience': 40, 'batch_size_g_mu': 512,
                            'batch_size': 32, 'reg_l2': .001, 'activation': 'sigmoid', 'epochs': 30, 'binary': True,
                            'val_split': 0.0,  'verbose': 0, 'kernel_init': 'GlorotNormal', 'max_trials': 10}

    """XLEARNER"""

    params_XLearner_IHDP_a = {'dataset_name': "ihdp_a", 'num': 100, 'lr': 1e-3, 'patience': 40, 'batch_size_g_mu': 256,
                              'batch_size': 673, 'reg_l2': .01, 'activation': 'linear', 'epochs': 300, 'binary': False,
                              'kernel_init': 'RandomNormal','val_split': 0.0, 'verbose': 0}

    params_XLearner_IHDP_b = {'dataset_name': "ihdp_b", 'num': 100, 'lr': 1e-2, 'patience': 40, 'batch_size_g_mu': 256,
                              'batch_size': 256, 'reg_l2': .01, 'activation': 'linear', 'epochs': 200, 'binary': False,
                              'kernel_init': 'RandomNormal', 'hidden_d': 128, 'verbose': 0}

    params_XLearner_ACIC = {'dataset_name': "acic", 'num': 100, 'lr': 1e-3, 'patience': 40, 'batch_size_g_mu': 256,
                            'batch_size': 673, 'reg_l2': .01, 'activation': None, 'epochs': 1000, 'binary': False,
                            'verbose': 0, 'kernel_init': 'RandomNormal', 'val_split': 0.0}

    params_XLearner_JOBS = {'dataset_name': "jobs", 'num': 100, 'lr': 1e-3, 'patience': 40, 'batch_size_g_mu': 256,
                            'batch_size': 673, 'reg_l2': .01, 'activation': 'sigmoid', 'epochs': 1000, 'binary': True,
                            'verbose': 0, 'kernel_init': 'GlorotNormal', 'val_split': 0.0}

    """TARNET"""

    params_TARnet_IHDP_a = {'dataset_name': "ihdp_a", 'num': 100, 'lr': 1e-4, 'patience': 40, 'batch_size': 64,
                            'reg_l2': .01, 'activation': 'linear', 'epochs': 300, 'binary': False, 'verbose': 0,
                            'val_split': 0.0, 'kernel_init': 'RandomNormal', 'max_trials': 10, 'defaults': True,
                            'n_fc': 3, 'hidden_phi': 200, 'n_fc_y0': 3, 'hidden_y0': 100, 'n_fc_y1': 3,
                            'hidden_y1': 100}

    params_TARnet_IHDP_b = {'dataset_name': "ihdp_b", 'num': 100, 'lr': 1e-4, 'patience': 5, 'batch_size': 32,
                            'reg_l2': .01, 'activation': 'linear', 'epochs': 300, 'binary': False, 'verbose': 0,
                            'val_split': 0.0, 'kernel_init': 'GlorotNormal', 'max_trials': 10, 'defaults': True,
                            'n_fc': 3, 'hidden_phi': 200, 'n_fc_y0': 3, 'hidden_y0': 100, 'n_fc_y1': 3,
                            'hidden_y1': 100}

    params_TARnet_ACIC = {'dataset_name': "acic", 'num': 77, 'lr': 1e-3, 'patience': 40, 'batch_size': 256,
                          'reg_l2': .01, 'activation': 'linear', 'epochs': 300, 'binary': False, 'verbose': 0,
                          'val_split': 0.0, 'kernel_init': 'RandomNormal', 'max_trials': 10, 'defaults': True,
                            'n_fc': 3, 'hidden_phi': 200, 'n_fc_y0': 3, 'hidden_y0': 200, 'n_fc_y1': 3,
                            'hidden_y1': 200}

    params_TARnet_JOBS = {'dataset_name': "jobs", 'num': 100, 'lr': 1e-2, 'patience': 40, 'batch_size': 256,
                          'reg_l2': .01,  'activation': 'sigmoid', 'epochs': 30, 'binary': True,
                          'val_split': 0.0, 'verbose': 0, 'kernel_init': 'GlorotNormal', 'max_trials': 20,
                          'defaults': True, 'n_fc': 3, 'hidden_phi': 200, 'n_fc_y0': 3, 'hidden_y0': 200,
                          'n_fc_y1': 3, 'hidden_y1': 200}

    """DRAGONNET"""

    params_DragonNet_IHDP_a = {'dataset_name': "ihdp_a", 'num': 100, 'lr': 1e-4, 'patience': 40, 'batch_size': 64,
                               'reg_l2': .01, 'activation': 'linear', 'epochs': 300, 'binary': False, 'dafaults': True,
                               'verbose': 0, 'kernel_init': 'RandomNormal', 'val_split': 0.0, 'max_trials': 10,
                               'n_fc': 3, 'hidden_phi': 200, 'n_fc_y0': 3, 'hidden_y0': 100, 'n_fc_y1': 3,
                                'hidden_y1': 100, 'n_fc_t': 1, 'hidden_t': 1}

    params_DragonNet_IHDP_b = {'dataset_name': "ihdp_b", 'num': 100, 'lr': 1e-4, 'patience': 5, 'batch_size': 32,
                               'reg_l2': .01, 'activation': 'linear', 'epochs': 300, 'binary': False, 'verbose': 0,
                               'kernel_init': 'GlorotNormal', 'val_split': 0.0, 'max_trials': 10,  'n_fc': 3,
                               'hidden_phi': 200, 'n_fc_y0': 3, 'hidden_y0': 100, 'n_fc_y1': 3, 'dafaults': True,
                                'hidden_y1': 100, 'n_fc_t': 1, 'hidden_t': 1}

    params_DragonNet_ACIC = {'dataset_name': "acic", 'num': 77, 'lr': 1e-4, 'patience': 40, 'batch_size': 256,
                             'reg_l2': .01, 'activation': 'linear', 'epochs': 300, 'binary': False, 'val_split': 0.0,
                             'verbose': 0, 'kernel_init': 'RandomNormal', 'max_trials': 10}

    params_DragonNet_JOBS = {'dataset_name': "jobs", 'num': 100, 'lr': 1e-3, 'patience': 40, 'batch_size': 512,
                             'reg_l2': .01, 'activation': 'sigmoid', 'epochs': 30, 'binary': True, 'val_split': 0.0,
                             'verbose': 0, 'kernel_init': 'RandomNormal', 'max_trials': 20}

    """CEVAE"""

    params_CEVAE_IHDP_a = {'dataset_name': "ihdp_a", 'num': 100, 'num_bin': 19, 'num_cont': 6, 'lr': 1e-3,
                           'patience': 40, 'batch_size': 64, 'reg_l2': .01, 'activation': 'linear',
                           'latent_dim': 20, 'epochs': 300, 'binary': False,'val_split': 0.0, 'verbose': 0,
                           'kernel_init': 'RandomNormal'}

    params_CEVAE_IHDP_b = {'dataset_name': "ihdp_b", 'num': 100, 'num_bin': 19, 'num_cont': 6, 'lr': 1e-3,
                           'patience': 40, 'batch_size': 64, 'reg_l2': .01, 'activation': 'linear',
                           'latent_dim': 20, 'epochs': 300, 'binary': False, 'val_split': 0.0, 'verbose': 0,
                           'kernel_init': 'GlorotNormal'}

    params_CEVAE_ACIC = {'dataset_name': "acic", 'num': 77,  'num_bin': 55, 'num_cont': 0, 'lr': 1e-3, 'patience': 40,
                         'batch_size': 400, 'reg_l2': .01, 'activation': 'linear', 'latent_dim': 20,
                         'epochs': 1000, 'binary': False, 'val_split': 0.0, 'verbose': 0,
                         'kernel_init': 'RandomNormal'}

    params_CEVAE_JOBS = {'dataset_name': "jobs", 'num': 100, 'lr': 1e-3, 'num_bin': 0, 'num_cont': 17, 'patience': 40,
                         'batch_size': 1024, 'reg_l2': .01, 'activation': 'sigmoid', 'latent_dim': 20,
                         'epochs': 50, 'binary': True, 'val_split': 0.0, 'verbose': 0,
                         'kernel_init': 'GlorotNormal'}

    """TEDVAE"""

    params_TEDVAE_IHDP_a = {'dataset_name': "ihdp_a", 'num': 100, 'lr': 1e-4, 'patience': 40, 'batch_size': 1024,
                            'reg_l2': .01, 'activation': 'linear', 'latent_dim_z': 15,
                            'num_bin': 19,'num_cont': 6, 'latent_dim_zt': 15,
                            'latent_dim_zy': 5, 'epochs': 400, 'binary': False,
                            'verbose': 0, 'kernel_init': 'RandomNormal'}

    params_TEDVAE_IHDP_b = {'dataset_name': "ihdp_b", 'num': 100, 'lr': 1e-3, 'patience': 40, 'batch_size': 1024,
                            'reg_l2': .01, 'activation': 'linear', 'latent_dim_z': 15, 'num_bin': 19,
                            'num_cont': 6, 'latent_dim_zt': 15, 'latent_dim_zy': 5, 'epochs': 400, 'binary': False,
                            'verbose': 0, 'val_split': 0.0, 'kernel_init': 'GlorotNormal'}

    params_TEDVAE_ACIC = {'dataset_name': "acic", 'num': 77, 'lr': 1e-3, 'patience': 25, 'batch_size': 32,
                          'reg_l2': .01, 'activation': 'linear', 'latent_dim_z': 15, 'num_bin': 55,
                          'num_cont': 0, 'latent_dim_zt': 15, 'latent_dim_zy': 5, 'epochs': 300, 'binary': False,
                          'val_split': 0.0, 'verbose': 0, 'kernel_init': 'RandomNormal'}

    params_TEDVAE_JOBS = {'dataset_name': "jobs", 'num': 100, 'lr': 1e-3, 'patience': 40, 'batch_size': 256,
                          'reg_l2': .01, 'activation': 'sigmoid', 'latent_dim_z': 15, 'num_bin': 0,
                          'num_cont': 17, 'latent_dim_zt': 15, 'latent_dim_zy': 5, 'epochs': 30, 'binary': True,
                          'val_split': 0.0, 'verbose': 0, 'kernel_init': 'GlorotNormal'}

    """CFRNET"""

    params_CFRNet_IHDP_a = {'dataset_name': "ihdp_a", 'num': 100, 'lr': 1e-4, 'patience': 40, 'tuner_batch_size': 256,
                            'batch_size': 1024, 'reg_l2': .01, 'activation': 'linear', 'epochs': 300,
                            'binary': False, 'verbose': 0, 'kernel_init': 'RandomNormal',  'defaults': True,
                            'n_fc': 3, 'hidden_phi': 200, 'n_fc_y0': 3, 'hidden_y0': 100, 'n_fc_y1': 3, 'hidden_y1': 100,
                            'n_fc_t': 3, 'hidden_t': 100
                            }

    params_CFRNet_IHDP_b = {'dataset_name': "ihdp_b", 'num': 100, 'lr': 1e-3, 'patience': 5,
                            'tuner_batch_size': 256, 'batch_size': 1024, 'reg_l2': .01, 'activation': 'linear',
                            'epochs': 300, 'binary': False, 'verbose': 0, 'kernel_init': 'GlorotNormal', 'defaults': True,
                             'n_fc': 3, 'hidden_phi': 200, 'n_fc_y0': 3, 'hidden_y0': 100, 'n_fc_y1': 3,
                            'hidden_y1': 100, 'n_fc_t': 3, 'hidden_t': 100,
                            }

    params_CFRNet_ACIC = {'dataset_name': "acic", 'num': 77, 'lr': 1e-3, 'patience': 40, 'tuner_batch_size': 256,
                          'batch_size': 512, 'reg_l2': .01, 'activation': 'linear', 'epochs': 500, 'binary': False,
                          'verbose': 0, 'kernel_init': 'GlorotNormal'}

    params_CFRNet_JOBS = {'dataset_name': "jobs", 'num': 100, 'lr': 1e-2, 'patience': 40, 'tuner_batch_size': 512,
                          'hidden_phi': 200, 'batch_size': 1024, 'reg_l2': .01, 'activation': 'sigmoid', 'epochs': 50,
                          'binary': True, 'verbose': 0, 'kernel_init': 'RandomNormal'}

    """GANITE"""

    params_GANITE_IHDP_a = {'dataset_name': "ihdp_a", 'num': 100, 'lr': 1e-3, 'patience': 40, 'defaults': True,
                            'batch_size_g': 64, 'batch_size_i': 128, 'reg_l2': .01, 'activation': 'linear',
                            'binary': False, 'epochs_g': 1000, 'verbose': 0, 'epochs_i': 500, 'val_split': 0.0,
                            'kernel_init': 'RandomNormal'}

    params_GANITE_IHDP_b = {'dataset_name': "ihdp_b", 'num': 100, 'lr': 1e-3, 'patience': 40, 'batch_size_g': 64,
                            'batch_size_i': 256, 'reg_l2': .01, 'activation': 'linear', 'binary': False,
                            'epochs_g': 1000, 'verbose': 0, 'epochs_i': 500, 'val_split': 0.0,
                            'kernel_init': 'GlorotNormal'}
    
    params_GANITE_ACIC = {'dataset_name': "acic", 'num': 77, 'lr': 1e-4, 'patience': 40,'batch_size_g': 128,
                          'batch_size_i': 256, 'reg_l2': .01, 'activation': 'linear', 'binary': False, 'epochs_g': 1000,
                          'verbose': 0, 'epochs_i': 500, 'val_split': 0.0, 'kernel_init': 'RandomNormal'}

    params_GANITE_JOBS = {'dataset_name': "jobs", 'num': 100, 'lr': 1e-3, 'patience': 40, 'batch_size_g': 64,
                          'batch_size_i': 256, 'reg_l2': .01, 'activation': 'linear', 'binary': True, 'epochs_g': 1000,
                          'verbose': 0, 'epochs_i': 500, 'val_split': 0.0, 'kernel_init': 'RandomNormal'}

    """DKLITE"""

    params_DKLITE_IHDP_a = {'dataset_name': "ihdp_a", 'num': 100, 'lr': 1e-3, 'patience': 40, 'batch_size': 1024,
                            'max_trials': 10, 'tuner_batch_size': 32, 'reg_l2': .01,  'activation': 'linear',
                            'dim_z': 80, 'epochs': 150, 'binary': False, 'reg_var': 1.0, 'reg_rec': 0.7, 'defaults': True,
                            'n_fc_encoder': 2, 'hidden_phi_encoder': 50, 'n_fc_decoder': 2, 'hidden_phi_decoder': 50,
                            'verbose': 0, 'kernel_init': 'RandomNormal', 'x_size': 25}

    params_DKLITE_IHDP_b = {'dataset_name': "ihdp_a", 'num': 100, 'lr': 1e-3, 'patience': 40, 'batch_size': 1024,
                            'max_trials': 15, 'tuner_batch_size': 1024, 'reg_l2': .01,  'activation': 'linear',
                            'dim_z': 80, 'epochs': 300, 'binary': False, 'reg_var': 1.0, 'reg_rec': 0.7,
                            'defaults': False, 'n_fc_encoder': 2, 'hidden_phi_encoder': 50, 'n_fc_decoder': 2,
                            'hidden_phi_decoder': 50, 'verbose': 0, 'kernel_init': 'GlorotNormal', 'x_size': 25}

    params_DKLITE_ACIC = {'dataset_name': "acic", 'num': 77, 'lr': 1e-3, 'patience': 40, 'batch_size': 32,
                          'max_trials': 20, 'tuner_batch_size': 32, 'reg_l2': .01, 'activation': 'linear',
                          'dim_z': 50, 'epochs': 300, 'binary': False, 'reg_var': 1.0, 'reg_rec': 0.7,
                          'verbose': 0,  'kernel_init': 'RandomNormal', 'x_size': 55}

    params_DKLITE_JOBS = {'dataset_name': "jobs", 'num': 100, 'lr': 1e-4, 'patience': 40, 'batch_size': 512,
                          'max_trials': 10, 'tuner_batch_size': 512, 'reg_l2': .01, 'activation': 'sigmoid',
                          'dim_z': 80, 'epochs': 30, 'binary': True, 'reg_var': 1.0, 'reg_rec': 0.7,
                          'verbose': 0, 'kernel_init': 'GlorotNormal', 'x_size': 17}


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

    """-------------------------------------------------------------"""

    params = {'TARnet': params_TARnet, 'CEVAE': params_CEVAE, 'TEDVAE': params_TEDVAE, 'DKLITE': params_DKLITE,
              'DragonNet': params_DragonNet, 'TLearner': params_TLearner, 'SLearner': params_SLearner,
              'RLearner': params_RLearner, 'XLearner': params_XLearner, 'CFRNet': params_CFRNet,
              'GANITE': params_GANITE}

    return params[model_name][dataset_name]
