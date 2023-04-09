# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.language import ReviewLSTMSA
from models.attention import Attention, SoftAttention

class HANCI(nn.Module):
    '''
    NARRE: WWW 2018
    '''
    def __init__(self, opt):
        super(HANCI, self).__init__()
        self.opt = opt
        self.num_fea = 2  # ID + Review

        self.user_net = Net(opt, 'user')
        self.item_net = Net(opt, 'item')

    def forward(self, datas):
        user_reviews, item_reviews, uids, iids, user_item2id, item_user2id,user_item_ratio,item_user_ratio, user_doc, item_doc, \
        type, month = datas  #,user_item_type,user_item_month,item_user_type,item_user_month
        u_fea = self.user_net(user_reviews, uids, user_item2id)
        i_fea = self.item_net(item_reviews, iids, item_user2id)
        return u_fea, i_fea


class Net(nn.Module):
    def __init__(self, opt, uori='user'):
        super(Net, self).__init__()
        self.opt = opt

        if uori == 'user':
            id_num = self.opt.user_num
            ui_id_num = self.opt.item_num
        else:
            id_num = self.opt.item_num
            ui_id_num = self.opt.user_num

        bidirectional = False

        self.id_embedding = nn.Embedding(id_num, self.opt.id_emb_size)  # user/item num * 32
        self.word_embs = nn.Embedding(self.opt.vocab_size, self.opt.word_dim)  # vocab_size * 300
        self.u_i_id_embedding = nn.Embedding(ui_id_num, self.opt.id_emb_size)

        self.user_review_emb = ReviewLSTMSA(embedding_size=self.opt.word_dim, dropout=0.,
                                            hidden_size=opt.lstm_hidden_size, num_layers=1, bidirectional=bidirectional,
                                            da=100, r=10)
        review_hidden_size = opt.lstm_hidden_size * (1 + int(bidirectional))
        self.user_reviews_att = Attention(review_hidden_size, self.opt.id_emb_size, self.opt.id_emb_size)

        self.relu = nn.ReLU()
        self.fc_layer = nn.Linear(self.opt.filters_num, self.opt.id_emb_size)
        self.dropout = nn.Dropout(self.opt.drop_out)
        self.reset_para()

    def forward(self, reviews, ids, ids_list):
        # --------------- word embedding ----------------------------------
        reviews = self.word_embs(reviews)  # [32,5,5,300]
        id_emb = self.id_embedding(ids)    #[32,32]
        u_i_id_emb = self.u_i_id_embedding(ids_list)   #[32,5,32]

        # --------LSTM for review--------------------
        # 2.review_embedding, word-level attention
        user_reviews_emb, user_word_att = self.user_review_emb(reviews)  # [32,5,100] [batch, review_num, review_hidden_size], # [batch, review_num, review_len]

        # 4.review-level attention
        user_reviews_att = self.user_reviews_att(user_reviews_emb, u_i_id_emb)  # [batch, user_review_num, 1]
                                       # 5.add_reviews
        r_fea = self.dropout(torch.sum(user_reviews_emb * user_reviews_att, 1))  #[32,100] [batch, review_hidden_size]

        return torch.stack([id_emb, self.fc_layer(r_fea)], 1)


    def reset_para(self):
        nn.init.uniform_(self.fc_layer.weight, -0.1, 0.1)
        nn.init.constant_(self.fc_layer.bias, 0.1)
        if self.opt.use_word_embedding:
            w2v = torch.from_numpy(np.load(self.opt.w2v_path))
            if self.opt.use_gpu:
                self.word_embs.weight.data.copy_(w2v.cuda())
            else:
                self.word_embs.weight.data.copy_(w2v)
        else:
            nn.init.xavier_normal_(self.word_embs.weight)
        nn.init.uniform_(self.id_embedding.weight, a=-0.1, b=0.1)
        nn.init.uniform_(self.u_i_id_embedding.weight, a=-0.1, b=0.1)
