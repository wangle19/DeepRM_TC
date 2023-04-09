# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import time

from .prediction import PredictionLayer
from .fusion import FusionLayer


class Model(nn.Module):

    def __init__(self, opt, Net):
        super(Model, self).__init__()
        self.opt = opt
        self.model_name = self.opt.model
        self.net = Net(opt)

        if self.opt.ui_merge == 'cat':
            if self.opt.r_id_merge == 'cat':
                feature_dim = self.opt.id_emb_size * self.opt.num_fea * 2
            else:
                feature_dim = self.opt.id_emb_size * 2
        else:
            if self.opt.r_id_merge == 'cat':
                feature_dim = self.opt.id_emb_size * self.opt.num_fea
            else:
                feature_dim = self.opt.id_emb_size

        self.opt.feature_dim = feature_dim
        self.fusion_net = FusionLayer(opt)
        self.predict_net = PredictionLayer(opt)
        self.dropout = nn.Dropout(self.opt.drop_out)

    def forward(self, datas):

        user_reviews, item_reviews, uids, iids,user_item2id, item_user2id,user_item_ratio,item_user_ratio,user_doc,item_doc,type,month = datas
        #, user_item_type, user_item_month, item_user_type, item_user_month

        user_feature, item_feature = self.net(datas)

        ui_feature = self.fusion_net(user_feature, item_feature)
        ui_feature = self.dropout(ui_feature)

        output = self.predict_net(ui_feature, uids, iids)  #,user_item_ratio,item_user_ratio
        return output

    def load(self, path):
        '''
        加载指定模型
        '''
        self.load_state_dict(torch.load(path))

    def save(self, path):
        '''
        保存模型
        '''
        prefix = 'checkpoints/'
        torch.save(self.state_dict(), path)
        return path
