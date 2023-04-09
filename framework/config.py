# -*- coding: utf-8 -*-

import numpy as np
import os
class DefaultConfig():
    model='DeepRM_TC'
    num_fea = 1
    dataset = 'total'
    k_fold = 2

    # -------------base config-----------------------#
    use_gpu = True
    gpu_id = 1

    seed = 2019
    num_epochs = 30
    num_workers = 0

    optimizer = 'Adam'
    output = 1e-3  # optimizer rameteri
    lr = 2e-3
    loss_method = 'mse'
    drop_out = 0.2

    use_word_embedding = True

    id_emb_size = 32
    query_mlp_size = 128
    fc_dim = 32

    doc_len = 500
    filters_num = 100
    kernel_size = 3
    num_heads = 2

    use_review = True
    use_doc = True
    self_att = False
    point_num_heads = 1

    r_id_merge = 'cat'  # review and ID feature
    ui_merge = 'cat'  # cat/add/dot
    output = 'gumbel'  # 'fm', 'lfm', 'other: sum the ui_feature'

    fine_step = False  # save mode in step level, defualt in epoch

    prediction_results_path = f'results/prediction_results'
    if not os.path.exists(prediction_results_path):
        os.makedirs(prediction_results_path, exist_ok=True)
    pth_path = f'results/prediction_results/{model}_{loss_method}'

    picture_save_path =  f'results/pictures'
    if not os.path.exists(picture_save_path):
        os.makedirs(picture_save_path, exist_ok=True)

    batch_size = 32
    print_step = 100
    weight_decay = 0.001
    ## 读取

    def parse(self,dataset,kwargs):
        f = open(f'pro_dataset/{dataset}/para_dict_rating.txt', 'r')
        a = f.read()
        para_dict = eval(a)

        self.vocab_size = para_dict['vocab_size']
        self.r_max_len = para_dict['r_max_len']
        self.u_max_r = para_dict['u_max_r']
        self.i_max_r = para_dict['i_max_r']
        # print('self.i_max_r',self.i_max_r)

        self.train_data_size = para_dict['train_data_size']
        self.test_data_size = para_dict['test_data_size']
        self.val_data_size = para_dict['val_data_size']

        self.user_num = para_dict['user_num']+1
        self.item_num = para_dict['item_num']+1
        self.type_num = para_dict['type_num']+1
        self.month_num = para_dict['month_num']+1

        self.word_dim = 300
        self.lstm_hidden_size = 100
        self.bidirectional = False
        self.num_layers = 1

        self.data_root = f'./pro_dataset/{dataset}'
        prefix = f'{self.data_root}/train'
        # print("data_root", self.data_root)

        self.user_list_path = f'{prefix}/userReview2Index.npy'
        self.item_list_path = f'{prefix}/itemReview2Index.npy'

        self.user_item_id_path = f'{prefix}/user_item2id.npy'
        self.item_user_id_path = f'{prefix}/item_user2id.npy'

        self.user_item_type_path = f'{prefix}/user_item2type.npy'
        self.item_user_type_path = f'{prefix}/item_user2type.npy'

        self.user_item_month_path = f'{prefix}/user_item2month.npy'
        self.item_user_month_path = f'{prefix}/item_user2month.npy'

        self.user_item_ratio_path = f'{prefix}/user_item2ratio.npy'
        self.item_user_ratio_path = f'{prefix}/item_user2ratio.npy'

        self.user_doc_path = f'{prefix}/userDoc2Index.npy'
        self.item_doc_path = f'{prefix}/itemDoc2Index.npy'

        self.w2v_path = f'{prefix}/w2v_'+str(self.word_dim)+'.npy'
        # self.w2v_path = f'{prefix}/w2v.npy'
        '''
        user can update the default hyperparamter
        '''
        # print("load npy from dist...")
        self.users_review_list = np.load(self.user_list_path, encoding='bytes')
        self.items_review_list = np.load(self.item_list_path, encoding='bytes')
        self.user_item_id_list = np.load(self.user_item_id_path, encoding='bytes')
        self.item_user_id_list = np.load(self.item_user_id_path, encoding='bytes')
        # self.user_item_type_list = np.load(self.user_item_type_path, encoding='bytes')
        # self.item_user_type_list = np.load(self.item_user_type_path, encoding='bytes')
        # self.user_item_month_list = np.load(self.user_item_month_path, encoding='bytes')
        # self.item_user_month_list = np.load(self.item_user_month_path, encoding='bytes')

        self.user_item_ratio_list = np.load(self.user_item_ratio_path, encoding='bytes')
        self.item_user_ratio_list = np.load(self.item_user_ratio_path, encoding='bytes')
        self.user_doc = np.load(self.user_doc_path, encoding='bytes')
        self.item_doc = np.load(self.item_doc_path, encoding='bytes')

        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise Exception('opt has No key: {}'.format(k))
            setattr(self, k, v)

        # print('*************************************************')
        # print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__') and k != 'user_list' and k != 'item_list':
                ...
                # print("{} => {}".format(k, getattr(self, k)))
        # print('*************************************************')

