# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.language import *
from models.attention import *


class DeepRM_TC_No_Attention(nn.Module):
    '''
    NARRE: WWW 2018
    '''
    def __init__(self, opt):
        super(DeepRM_TC_No_Attention, self).__init__()
        self.opt = opt
        self.num_fea = 3  # ID + Review +context
        self.head = opt.point_num_heads
        # id embedding
        self.user_id_embedding = nn.Embedding(self.opt.user_num, self.opt.id_emb_size)  # user/item num * 32
        self.item_id_embedding = nn.Embedding(self.opt.item_num, self.opt.id_emb_size)  # user/item num * 32
        self.type_embedding = nn.Embedding(self.opt.type_num, self.opt.id_emb_size)
        self.month_embedding = nn.Embedding(self.opt.month_num, self.opt.id_emb_size)

        self.user_word_embs = nn.Embedding(self.opt.vocab_size, self.opt.word_dim)  # vocab_size * 300
        self.item_word_embs = nn.Embedding(self.opt.vocab_size, self.opt.word_dim)  # vocab_size * 300

        self.review_hidden_size = self.opt.word_dim

        self.relu = nn.ReLU()
        self.user_fc_layer = nn.Linear(self.review_hidden_size, self.opt.id_emb_size)
        self.item_fc_layer = nn.Linear(self.review_hidden_size, self.opt.id_emb_size)

        # final fc
        self.u_fc = nn.Linear(self.opt.id_emb_size * self.head, self.opt.id_emb_size)
        self.i_fc = nn.Linear(self.opt.id_emb_size * self.head, self.opt.id_emb_size)

        self.dropout = nn.Dropout(self.opt.drop_out)

        self.reset_para()

    def forward(self, datas):
        user_reviews, item_reviews, uids, iids, user_item2id, item_user2id,user_item_ratio,item_user_ratio, user_doc, item_doc, \
        type, month = datas   #,user_item_type,user_item_month,item_user_type,item_user_month
        batch = user_reviews.size(0)

        # --------------- word embedding ----------------------------------
        user_reviews = self.user_word_embs(user_reviews)  # (20,1,82,300)
        item_reviews = self.item_word_embs(item_reviews)  # (20,25,82,300)
        user_word_embedding = user_reviews.view(batch, -1, self.review_hidden_size)
        item_word_embedding = item_reviews.view(batch, -1, self.review_hidden_size)

        user_word_embedding = user_word_embedding.view(batch, self.opt.u_max_r, self.opt.r_max_len,
                                                               self.review_hidden_size)
        item_word_embedding = item_word_embedding.view(batch, self.opt.i_max_r, self.opt.r_max_len,
                                                               self.review_hidden_size)
        # --------review_embedding --------------------
        user_review_embedding = self.dropout(
            self.relu(self.user_fc_layer(torch.sum(user_word_embedding, 2))))  # [batch, review_hidden_size]
        item_review_embedding = self.dropout(
            self.relu(self.item_fc_layer(torch.sum(item_word_embedding, 2))))  # [batch, review_hidden_size]

        # --------------- context embedding ----------------------------------
        user_id_emb = self.user_id_embedding(uids)
        item_id_emb = self.item_id_embedding(iids)
        type_emd = self.type_embedding(type)
        month_emd = self.month_embedding(month)


        user_fea = torch.sum(user_review_embedding, 1)  # [batch, review_hidden_size]
        item_fea = torch.sum(item_review_embedding, 1)  # [batch, review_hidden_size]

        user_fea = torch.stack([user_id_emb, type_emd, user_fea], 1)
        item_fea = torch.stack([item_id_emb, month_emd, item_fea], 1)
        # user_fea = torch.stack([user_id_emb, month_emd,user_fea], 1)
        # item_fea = torch.stack([item_id_emb, type_emd, item_fea], 1)

        return user_fea, item_fea

    def reset_para(self):
        for fc in [self.user_fc_layer, self.item_fc_layer]:
            nn.init.uniform_(fc.weight, -0.1, 0.1)
            nn.init.uniform_(fc.bias, -0.1, 0.1)

        if self.opt.use_word_embedding:
            w2v = torch.from_numpy(np.load(self.opt.w2v_path))
            if self.opt.use_gpu:
                self.user_word_embs.weight.data.copy_(w2v.cuda())
                self.item_word_embs.weight.data.copy_(w2v.cuda())
            else:
                self.user_word_embs.weight.data.copy_(w2v)
                self.item_word_embs.weight.data.copy_(w2v)
        else:
            nn.init.xavier_normal_(self.user_word_embs.weight)
            nn.init.xavier_normal_(self.item_word_embs.weight)

        nn.init.uniform_(self.user_id_embedding.weight, a=-0.1, b=0.1)
        nn.init.uniform_(self.item_id_embedding.weight, a=-0.1, b=0.1)


