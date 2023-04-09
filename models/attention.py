"""
Created on 2019/3/2 15:58

@author: zhouweixin
@note: 注意力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_layers, id_embedding_size, attention_size):
        super(Attention, self).__init__()

        self.id_embedding_size = id_embedding_size
        self.attention_size = attention_size

        self.review_linear = nn.Linear(hidden_layers, attention_size, bias=True)
        self.id_linear = nn.Linear(id_embedding_size, attention_size, bias=False)
        self.att_linear = nn.Linear(attention_size, 1)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

    def forward(self, reviews_embed, ids_embed):
        """
        :param reviews_embed: [batch, review_num, hidden_layers]   [batch, user_review_num, hidden_size]
        :param ids_embed: [batch, user_review_num, id_embedding_size]
        :return:
        """
        review_linear_out = self.review_linear(reviews_embed) # [batch, review_num, attention_size]
        id_linear_out = self.id_linear(ids_embed) # [batch, review_num, attention_size]
        output = review_linear_out + id_linear_out
        output = self.relu(output)
        output = self.att_linear(output) # [batch, review_num, 1]
        output = self.softmax(output)
        return output


class SelfAttention(nn.Module):
    def __init__(self, review_hidden_size, da=100, r=10):
        super(SelfAttention, self).__init__()

        self.review_hidden_size = review_hidden_size
        self.da = da
        self.r = r
        self.linear1 = nn.Linear(review_hidden_size, da)
        self.linear2 = nn.Linear(da, r)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, review_embedding):
        # review_embedding: [batch*review_num, review_len, review_hidden_size]
        linear_output = self.linear1(review_embedding)  # [batch*review_num, review_len, da]
        attention = self.linear2(linear_output)  # [batch*review_num, review_len, r]

        # attention = self.softmax(attention) TODO 实验测试不要softmax效果更好
        attention = attention.transpose(1, 2)  # [batch*review_num, r, review_len]

        return attention


class SoftAttention(nn.Module):
    def __init__(self, review_num_filters, id_embedding_size, soa_size=100):
        super(SoftAttention, self).__init__()

        self.linear1 = nn.Linear(id_embedding_size, soa_size)
        self.linear2 = nn.Linear(soa_size, review_num_filters)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, review_embedding, id_embedding):
        # review_embedding: [batch, review_num_filters]
        # id_embedding: [batch, id_embedding_size]

        soft_att = self.linear1(id_embedding) # [batch, soa_size]
        soft_att = self.linear2(soft_att) # [batch, review_num_filters]
        # soft_att = self.softmax(soft_att)

        output = soft_att * review_embedding # [batch, review_num_filters]
        return output

class Co_Attention(nn.Module):
    '''
    review-level and word-level co-attention module
    Eq (2,3, 10,11)
    '''
    def __init__(self, dim, gumbel, pooling):
        super(Co_Attention, self).__init__()
        self.gumbel = gumbel
        self.pooling = pooling
        self.M = nn.Parameter(torch.randn(dim, dim))
        # self.fc_u = nn.Linear(dim, dim)
        # self.fc_i = nn.Linear(dim, dim)

        # self.reset_para()

    # def reset_para(self):
    #     nn.init.xavier_uniform_(self.M, gain=1)
    #     nn.init.uniform_(self.fc_u.weight, -0.1, 0.1)
    #     nn.init.uniform_(self.fc_u.bias, -0.1, 0.1)
    #     nn.init.uniform_(self.fc_i.weight, -0.1, 0.1)
    #     nn.init.uniform_(self.fc_i.bias, -0.1, 0.1)

    def forward(self, u_fea, i_fea):
        S = u_fea.matmul(self.M).matmul(i_fea.permute(0,2,1))  # B * L1 * L2 Eq(2/10), we transport item instead user
        if self.pooling == 'max':
            u_score = S.max(2)[0]  # B * L1
            i_score = S.max(1)[0]  # B * L2
        else:
            u_score = S.mean(2)  # B * L1
            i_score = S.mean(1)  # B * L2
        if self.gumbel:
            p_u = F.gumbel_softmax(u_score, hard=True, dim=1)
            p_i = F.gumbel_softmax(i_score, hard=True, dim=1)
        else:
            p_u = F.softmax(u_score, dim=1)
            p_i = F.softmax(i_score, dim=1)
        return p_u.unsqueeze(2), p_i.unsqueeze(2)

class new_Co_Attention(nn.Module):
    '''
    review-level and word-level co-attention module
    Eq (2,3, 10,11)
    '''
    def __init__(self, gumbel, pooling):
        super(new_Co_Attention, self).__init__()
        self.gumbel = gumbel
        self.pooling = pooling

    def forward(self, u_fea, i_fea):

        # S = u_fea.matmul(self.M).matmul(i_fea.permute(0,2,1))  # B * L1 * L2 Eq(2/10), we transport item instead user
        S = torch.matmul(u_fea, torch.transpose(i_fea, 1, 2))
        if self.pooling == 'max':
            u_score = S.max(2)[0]  # B * L1
            i_score = S.max(1)[0]  # B * L2
        else:
            u_score = S.mean(2)  # B * L1
            i_score = S.mean(1)  # B * L2
        if self.gumbel:
            p_u = F.gumbel_softmax(u_score, hard=True, dim=1)
            p_i = F.gumbel_softmax(i_score, hard=True, dim=1)
        else:
            p_u = F.softmax(u_score, dim=1)
            p_i = F.softmax(i_score, dim=1)
        return p_u.unsqueeze(2), p_i.unsqueeze(2)


class word_Co_Attention(nn.Module):
    '''
    review-level and word-level co-attention module
    Eq (2,3, 10,11)
    '''
    def __init__(self, dim, gumbel, pooling):
        super(word_Co_Attention, self).__init__()
        self.gumbel = gumbel
        self.pooling = pooling
        self.M = nn.Parameter(torch.randn(dim, dim))
        self.fc_u = nn.Linear(dim, dim)
        self.fc_i = nn.Linear(dim, dim)

        self.reset_para()

    def reset_para(self):
        nn.init.xavier_uniform_(self.M, gain=1)
        nn.init.uniform_(self.fc_u.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc_u.bias, -0.1, 0.1)
        nn.init.uniform_(self.fc_i.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc_i.bias, -0.1, 0.1)

    def forward(self, u_fea, i_fea):
        '''
        u_fea: B * L1 * d
        i_fea: B * L2 * d
        return:
        B * L1 * 1
        B * L2 * 1
        '''
        # u_fea = self.fc_u(u_fea)
        # i_fea = self.fc_i(i_fea)

        S = u_fea.matmul(self.M).matmul(i_fea.permute(0,1,3,2))  # B * L1 * L2 Eq(2/10), we transport item instead user

        if self.pooling == 'max':
            u_score = S.max(3)[0]  # B * L1
            i_score = S.max(2)[0]  # B * L2
        else:
            u_score = S.mean(3)  # B * L1
            i_score = S.mean(2)  # B * L2
        if self.gumbel:
            p_u = F.gumbel_softmax(u_score, hard=True, dim=2)
            p_i = F.gumbel_softmax(i_score, hard=True, dim=2)
        else:
            p_u = F.softmax(u_score, dim=2)
            p_i = F.softmax(i_score, dim=2)
        return p_u.unsqueeze(3), p_i.unsqueeze(3)
