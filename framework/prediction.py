# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictionLayer(nn.Module):
    '''
        Rating Prediciton Methods
        - LFM: Latent Factor Model
        - (N)FM: (Neural) Factorization Machine
        - MLP
        - SUM
    '''
    def __init__(self, opt):
        super(PredictionLayer, self).__init__()
        self.output = opt.output
        if opt.output == "fm":
            self.model = FM(opt.feature_dim, opt.user_num, opt.item_num)
        elif opt.output == "lfm":
            self.model = LFM(opt.feature_dim, opt.user_num, opt.item_num)
        elif opt.output == 'mlp':
            self.model = MLP(opt.feature_dim)
        elif opt.output == 'nfm':
            self.model = NFM(opt.feature_dim)
        elif opt.output =='gumbel':
            self.model = Gumbel(opt.feature_dim)
        else:
            self.model = torch.sum

    def forward(self, feature, uid, iid):  #,user_item_ratio,item_user_ratio
        if self.output =='gumbel':
            preds_output = self.model(feature, uid, iid)  #, user_item_ratio, item_user_ratio
        elif self.output == "lfm" or "fm" or "nfm":
            preds_output = self.model(feature, uid, iid)
        else:
            preds_output = self.model(feature, 1, keepdim=True)

        return preds_output

class Gumbel(nn.Module):

    def __init__(self,dim):

        super(Gumbel, self).__init__()
        self.fc = nn.Linear(dim, 5)
        self.conv1d = nn.Conv1d(5, 5, kernel_size=3)
        self.mean_fun = nn.Linear(5, 5)
        self.var_fun = nn.Linear(5, 5)
        self.mlp1 = nn.Linear(5, 5)
        self.mlp2 = nn.Linear(5,5)


    def forward(self,feature,uid,iid,u_rating_ratio,i_rating_ratio):

        feature = self.fc(feature)
        feature = feature.unsqueeze(1)
        e = 2.71828182845

        rating = torch.FloatTensor([1.,2.,3.,4.,5.]).cuda()

        umean = self.mean_fun(u_rating_ratio.float())
        uvar = self.var_fun(u_rating_ratio.float())
        uvar = (uvar)**2 + 0.2
        l1 = torch.pow(e,((rating-umean))/(uvar))
        l2 = torch.pow(e,-l1)
        out1 = 1/uvar * l1 * l2  #(32,1,5)

        imean = self.mean_fun(i_rating_ratio.float())
        ivar = self.var_fun(i_rating_ratio.float())
        ivar = (ivar)**2 + 0.2
        l1 = torch.pow(e, ((rating - imean)) / ivar)
        l2 = torch.pow(e, -l1)
        out2 = 1 / ivar * l1 * l2

        out = torch.cat((out1.unsqueeze(1), out2.unsqueeze(1), feature), dim=1)
        out = self.conv1d(out.permute(0, 2, 1)).squeeze(2)
        user_rating_ratio = self.mlp1(u_rating_ratio.float()) # (32, 5)
        item_rating_ratio = self.mlp2(i_rating_ratio.float()) # (32, 5)

        out = torch.sum(out*user_rating_ratio*item_rating_ratio, dim=-1)

        return out


class LFM(nn.Module):

    def __init__(self, dim, user_num, item_num):
        super(LFM, self).__init__()

        # ---------------------------fc_linear------------------------------
        self.fc = nn.Linear(dim, 1)
        # -------------------------LFM-user/item-bias-----------------------
        self.b_users = nn.Parameter(torch.randn(user_num, 1))
        self.b_items = nn.Parameter(torch.randn(item_num, 1))

        self.init_weight()

    def init_weight(self):
        nn.init.uniform_(self.fc.weight, a=-0.1, b=0.1)
        nn.init.uniform_(self.fc.bias, a=0.5, b=1.5)
        nn.init.uniform_(self.b_users, a=0.5, b=1.5)
        nn.init.uniform_(self.b_users, a=0.5, b=1.5)

    def rescale_sigmoid(self, score, a, b):
        return a + torch.sigmoid(score) * (b - a)

    def forward(self, feature, user_id, item_id):
        # return self.rescale_sigmoid(self.fc(feature), 1.0, 5.0) + self.b_users[user_id] + self.b_items[item_id]
        out = self.fc(feature) + self.b_users[user_id] + self.b_items[item_id]
        return out.squeeze(1)


class NFM(nn.Module):
    '''
    Neural FM
    '''
    def __init__(self, dim):
        super(NFM, self).__init__()
        self.dim = dim
        # ---------------------------fc_linear------------------------------
        self.fc = nn.Linear(dim, 1)
        # ------------------------------FM----------------------------------
        self.fm_V = nn.Parameter(torch.randn(16, dim))
        self.mlp = nn.Linear(16, 16)
        self.h = nn.Linear(16, 1, bias=False)
        self.drop_out = nn.Dropout(0.5)
        self.init_weight()

    def init_weight(self):
        nn.init.uniform_(self.fc.weight, -0.1, 0.1)
        nn.init.constant_(self.fc.bias, 0.1)
        nn.init.uniform_(self.fm_V, -0.1, 0.1)
        nn.init.uniform_(self.h.weight, -0.1, 0.1)

    def forward(self, input_vec, *args):
        fm_linear_part = self.fc(input_vec)
        fm_interactions_1 = torch.mm(input_vec, self.fm_V.t())
        fm_interactions_1 = torch.pow(fm_interactions_1, 2)

        fm_interactions_2 = torch.mm(torch.pow(input_vec, 2), torch.pow(self.fm_V, 2).t())
        bilinear = 0.5 * (fm_interactions_1 - fm_interactions_2)

        out = F.relu(self.mlp(bilinear))
        out = self.drop_out(out)
        out = self.h(out) + fm_linear_part
        return out.squeeze(1)


class FM(nn.Module):

    def __init__(self, dim, user_num, item_num):
        super(FM, self).__init__()
        self.dim = dim
        # ---------------------------fc_linear------------------------------
        self.fc = nn.Linear(dim, 1)
        # ------------------------------FM----------------------------------
        self.fm_V = nn.Parameter(torch.randn(dim, 10))
        self.b_users = nn.Parameter(torch.randn(user_num, 1))
        self.b_items = nn.Parameter(torch.randn(item_num, 1))

        self.init_weight()

    def init_weight(self):
        nn.init.uniform_(self.fc.weight, -0.05, 0.05)
        nn.init.constant_(self.fc.bias, 0.0)
        # nn.init.uniform_(self.b_users, a=0, b=0.1)
        # nn.init.uniform_(self.b_items, a=0, b=0.1)
        nn.init.uniform_(self.fm_V, -0.05, 0.05)

    def build_fm(self, input_vec):
        '''
        y = w_0 + \sum {w_ix_i} + \sum_{i=1}\sum_{j=i+1}<v_i, v_j>x_ix_j
        factorization machine layer
        refer: https://github.com/vanzytay/KDD2018_MPCN/blob/master/tylib/lib
                      /compose_op.py#L13
        '''
        # linear part: first two items
        fm_linear_part = self.fc(input_vec)

        fm_interactions_1 = torch.mm(input_vec, self.fm_V)
        fm_interactions_1 = torch.pow(fm_interactions_1, 2)

        fm_interactions_2 = torch.mm(torch.pow(input_vec, 2),
                                     torch.pow(self.fm_V, 2))
        fm_output = 0.5 * torch.sum(fm_interactions_1 - fm_interactions_2, 1, keepdim=True) + fm_linear_part
        return fm_output

    def forward(self, feature, uids, iids):

        fm_out = self.build_fm(feature)
        out = fm_out + self.b_users[uids] + self.b_items[iids]
        return out.squeeze(1)


class MLP(nn.Module):

    def __init__(self, dim):
        super(MLP, self).__init__()
        self.dim = dim
        # ---------------------------fc_linear------------------------------
        self.fc = nn.Linear(dim, 1)
        self.init_weight()

    def init_weight(self):
        nn.init.uniform_(self.fc.weight, 0.1, 0.1)
        nn.init.uniform_(self.fc.bias, a=0, b=0.2)

    def forward(self, feature, *args, **kwargs):
        out = F.relu(self.fc(feature)[0])
        return out.squeeze(1)
