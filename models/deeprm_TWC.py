# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.attention import *

class DeepRM_TC_TWC(nn.Module):
    '''
    Multi-Pointer Co-Attention Network for Recommendation
    WWW 2018
    '''
    def __init__(self, opt, head=3):
        '''
        head: the number of pointers
        '''
        super(DeepRM_TC_TWC, self).__init__()

        self.opt = opt
        self.num_fea = 3  # ID + DOC

        self.user_word_embs = nn.Embedding(opt.vocab_size, opt.word_dim)  # vocab_size * 300
        self.item_word_embs = nn.Embedding(opt.vocab_size, opt.word_dim)  # vocab_size * 300

        # id embedding
        self.user_id_embedding = nn.Embedding(self.opt.user_num, self.opt.id_emb_size)  # user/item num * 32
        self.item_id_embedding = nn.Embedding(self.opt.item_num, self.opt.id_emb_size)  # user/item num * 32
        self.type_embedding = nn.Embedding(self.opt.type_num, self.opt.id_emb_size)
        self.month_embedding = nn.Embedding(self.opt.month_num, self.opt.id_emb_size)

        # review gate
        self.fc_g1 = nn.Linear(opt.word_dim, opt.word_dim)
        self.fc_g2 = nn.Linear(opt.word_dim, opt.word_dim)

        # co-attention
        self.review_coatt =  new_Co_Attention(gumbel=True, pooling='max')
        self.word_coatt =  new_Co_Attention(gumbel=False, pooling='avg')
        self.fea_coatt = new_Co_Attention(gumbel=False, pooling='avg')

        self.user_fc_layer = nn.Linear(opt.word_dim, opt.id_emb_size)
        self.item_fc_layer = nn.Linear(opt.word_dim, opt.id_emb_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(opt.drop_out)

        self.reset_para()

    def forward(self, datas):

        user_reviews, item_reviews, uids, iids, user_item2id, item_user2id,user_item_ratio,item_user_ratio, user_doc, item_doc, \
        type, month= datas  #,user_item_type,user_item_month,item_user_type,item_user_month
        # --------------- id embedding ----------------------------------
        user_id_emb = self.user_id_embedding(uids)
        item_id_emb = self.item_id_embedding(iids)
        type_emd = self.type_embedding(type)
        month_emd = self.month_embedding(month)
        feature_embedding = torch.stack((user_id_emb, item_id_emb, type_emd, month_emd), 1)

        # ------------------review-level co-attention ---------------------------------
        u_word_embs = self.user_word_embs(user_reviews)
        i_word_embs = self.item_word_embs(item_reviews)
        u_reviews = self.review_gate(u_word_embs)
        i_reviews = self.review_gate(i_word_embs)

        # ------------------review-level co-attention ---------------------------------
        p_u, p_i = self.review_coatt(u_reviews, i_reviews)             # B * L1/2 * 1
        # ------------------word-level co-attention ---------------------------------
        u_r_words = user_reviews.permute(0, 2, 1).float().bmm(p_u)   # (B * N * L1) X (B * L1 * 1)
        i_r_words = item_reviews.permute(0, 2, 1).float().bmm(p_i)   # (B * N * L2) X (B * L2 * 1)
        u_words = self.user_word_embs(u_r_words.squeeze(2).long())  # B * N * d
        i_words = self.item_word_embs(i_r_words.squeeze(2).long())  # B * N * d
        p_u, p_i = self.word_coatt(u_words, i_words)                 # B * N * 1
        u_w_fea = u_words.permute(0, 2, 1).bmm(p_u).squeeze(2)
        i_w_fea = u_words.permute(0, 2, 1).bmm(p_i).squeeze(2)

        user_review_embedding = self.dropout(self.relu(self.user_fc_layer(u_w_fea)))  # [batch, review_hidden_size]
        item_review_embedding = self.dropout(self.relu(self.item_fc_layer(i_w_fea)))
        review_embedding = torch.stack((user_review_embedding, item_review_embedding), 1)
        # --------review_feature_co-attention --------------------
        review_att, fea_att = self.fea_coatt(review_embedding, feature_embedding)
        fea_att_embedding = feature_embedding * fea_att
        user_id_emb, item_id_emb, type_emd, month_emd = torch.split(fea_att_embedding, 1, 1)
        user_id_emb = self.dropout(torch.sum(user_id_emb, 1))  # [batch, review_hidden_size]
        item_id_emb = self.dropout(torch.sum(item_id_emb, 1))  # [batch, review_hidden_size]
        type_emd = self.dropout(torch.sum(type_emd, 1))  # [batch, review_hidden_size]
        month_emd = self.dropout(torch.sum(month_emd, 1))  # [batch, review_hidden_size]

        review_att_embedding = review_embedding * review_att
        user_2att_review_embedding, item_2att_review_embedding = torch.split(review_att_embedding,[1, 1],1)

        u_fea = user_2att_review_embedding.squeeze(1)  # [batch, review_hidden_size]
        i_fea = user_2att_review_embedding.squeeze(1)   # [batch, review_hidden_size]

        user_fea = torch.stack([user_id_emb, type_emd, u_fea], 1)
        item_fea = torch.stack([item_id_emb, month_emd, i_fea], 1)

        return user_fea, item_fea

    def review_gate(self, reviews):
        # Eq 1
        reviews = reviews.sum(2)
        return torch.sigmoid(self.fc_g1(reviews)) * torch.tanh(self.fc_g2(reviews))

    def reset_para(self):
        for fc in [self.fc_g1, self.fc_g2, self.user_fc_layer, self.item_fc_layer]:
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
            nn.init.uniform_(self.user_word_embs.weight, -0.1, 0.1)
            nn.init.uniform_(self.item_word_embs.weight, -0.1, 0.1)

