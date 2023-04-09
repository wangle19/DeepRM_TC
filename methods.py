# -*- encoding: utf-8 -*-
import time
import random
import math
import fire

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from framework import ReviewData
from framework import Model
import models
from framework import config
import datetime
import os
import threading
from sklearn.metrics import *
import matplotlib.pyplot as plt
import math
import math
def now():
    return datetime.datetime.now()


def collate_fn(batch):
    data, label = zip(*batch)
    return data, label

def train(opt,each_pth_folder,based_on_path):
    # opt = getattr(config, 'DefaultConfig')()
    # opt.parse(dataset,kwargs)

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if opt.use_gpu:
        torch.cuda.manual_seed_all(opt.seed)

    if opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)

    model = Model(opt, getattr(models, opt.model))
    if based_on_path!=None:
        print(based_on_path)
        model.load(based_on_path)
    if opt.use_gpu:
        model.cuda()

    if model.net.num_fea != opt.num_fea:
        raise ValueError(f"the num_fea of {opt.model} is error, please specific --num_fea={model.net.num_fea}")

    # 3 data
    train_data = ReviewData(opt.data_root, mode="Train")
    train_data_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn)
    val_data = ReviewData(opt.data_root, mode="Val")
    val_data_loader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)
    # print(f'train data: {len(train_data)}; test data: {len(val_data)}')

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    # training
    # print("start training....")
    min_loss = 1e+10
    best_res = 1e+10
    mse_func = nn.MSELoss()
    mae_func = nn.L1Loss()
    smooth_mae_func = nn.SmoothL1Loss()
    binary_cross_entropy_func = nn.BCELoss()

    for epoch in range(opt.num_epochs):
        epoch_start_times=now()
        total_loss = 0.0
        total_maeloss = 0.0
        model.train()
        # print(f"{now()}  Epoch {epoch}...")
        for idx, (train_datas, scores) in enumerate(train_data_loader):
            if opt.use_gpu:
                scores = torch.FloatTensor(scores).cuda()
            else:
                scores = torch.FloatTensor(scores)
            train_datas = unpack_input(opt, train_datas)

            optimizer.zero_grad()
            output = model(train_datas)

            if opt.loss_method == 'mse':
                mse_loss = mse_func(output, scores)
                total_loss += mse_loss.item() * len(scores)
                loss = mse_loss
            if opt.loss_method == 'mae':
                mae_loss = mae_func(output, scores)
                total_maeloss += mae_loss.item()
                loss = mae_loss
            if opt.loss_method == 'smooth_mae':
                smooth_mae_loss = smooth_mae_func(output, scores)
                total_maeloss += smooth_mae_loss.item()
                loss = smooth_mae_loss
            loss.backward()
            optimizer.step()
            if opt.fine_step:
                if idx % opt.print_step == 0 and idx > 0:
                    # print("\t{}, {} step finised;".format(now(), idx))
                    val_loss, val_mse, val_mae, val_rmse = predict(model, val_data_loader, opt)
                    if val_loss < min_loss:
                        model.save(opt.pth_path)
                        min_loss = val_loss
                        # print("\tmodel save")
                    if val_loss > min_loss:
                        best_res = min_loss

        scheduler.step()
        average_loss = total_loss * 1.0 / len(train_data)
        # print(f"\ttrain data: loss:{total_loss:.4f}, average_loss: {average_loss:.4f};")

        val_loss, val_mse, val_mae = predict(model, val_data_loader, opt)

        if (epoch + 1) % 10 == 0:
            model.save( f'{each_pth_folder}/{opt.model}_{epoch}.pth')

        if val_loss < min_loss:
            model.save(opt.pth_path)
            min_loss = val_loss
            # print("model save")
        if val_mse < best_res:
            best_res = val_mse
        # print("*"*30)
    # print(f"{opt.model} {dataset} best_res:{best_res}")
    # print("----" * 20)

    return best_res


def predict(model, data_loader, opt):
    total_loss = 0.0
    total_maeloss = 0.0
    total_rmseloss = 0.0
    model.eval()
    with torch.no_grad():
        for idx, (test_data, scores) in enumerate(data_loader):
            if opt.use_gpu:
                scores = torch.FloatTensor(scores).cuda()
            else:
                scores = torch.FloatTensor(scores)
            test_data = unpack_input(opt, test_data)

            output = model(test_data)

            mse_loss = torch.sum((output-scores)**2)
            total_loss += mse_loss.item()

            mae_loss = torch.sum(abs(output-scores))
            total_maeloss += mae_loss.item()

    data_len = len(data_loader.dataset)
    mse = total_loss * 1.0 / data_len
    mae = total_maeloss * 1.0 / data_len

    model.train()
    return total_loss, mse,mae

def predict_test(opt):
    # print('test_1')
    # opt = getattr(config, 'DefaultConfig')()
    # opt.parse(dataset, kwargs)

    assert(len(opt.pth_path) > 0)
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if opt.use_gpu:
        torch.cuda.manual_seed_all(opt.seed)

    if len(opt.gpu_ids) == 0 and opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)

    model = Model(opt, getattr(models, opt.model))
    if opt.use_gpu:
        model.cuda()
        if len(opt.gpu_ids) > 0:
            model = nn.DataParallel(model, device_ids=opt.gpu_ids)
    if model.net.num_fea != opt.num_fea:
        raise ValueError(f"the num_fea of {opt.model} is error, please specific --num_fea={model.net.num_fea}")

    model.load(opt.pth_path)
    # print(f"load model: {opt.pth_path}")e
    test_data = ReviewData(opt.data_root, mode="Test")
    test_data_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)
    # print(f"{now()}: test in the test datset")
    metric_df(model, test_data_loader, opt)
    return


def metric_df(model, data_loader, opt):
    pre_y_all = []
    real_y_all  = []
    model.eval()
    with torch.no_grad():
        for idx, (test_data, scores) in enumerate(data_loader):
            if opt.use_gpu:
                scores = torch.FloatTensor(scores).cuda()
            else:
                scores = torch.FloatTensor(scores)
            test_data = unpack_input(opt, test_data)
            output = model(test_data)
            pre_y_all.extend(output.cpu().numpy())
            real_y_all.extend(scores.cpu().numpy())
    pre_real_df = pd.DataFrame(columns = ['real','pre'])
    pre_real_df['real'] = real_y_all
    pre_real_df['pre'] = pre_y_all

    prediction_results_path = f'results/prediction_results'
    if not os.path.exists(prediction_results_path):
        os.makedirs(prediction_results_path, exist_ok=True)

    try:
        final_df = pd.read_excel(f'{prediction_results_path}/{opt.dataset}_final_df.xlsx')
    except:
        final_df = pd.DataFrame(columns=['dataset','model','output','ui_merge','lr','MSE', 'MAE','RMSE'])

    index_num = len(final_df)
    average_mse = mean_squared_error(real_y_all, pre_y_all)
    average_mae = mean_absolute_error(real_y_all, pre_y_all)
    average_rmse = math.sqrt(average_mse)


    final_df.loc[index_num, ['dataset', 'model','output','ui_merge','lr','MSE', 'MAE','RMSE']] = [opt.dataset, opt.model, opt.output,opt.ui_merge,opt.lr,average_mse, average_mae,average_rmse]

    for score, score_df in pre_real_df.groupby('real'):
        MSE = f'{int(score)}_mse'
        MAE = f'{int(score)}_mae'
        final_df.loc[index_num,MSE] = mean_squared_error(score_df['real'], score_df['pre'])
        final_df.loc[index_num, MAE] = mean_absolute_error(score_df['real'], score_df['pre'])

    final_df.to_excel(f'{prediction_results_path}/{opt.dataset}_final_df.xlsx', index=False)
    print(
        f"{opt.model}_evaluation reslut: test_mse: {average_mse:.4f}; test_mae: {average_mae:.4f}")
    return

def unpack_input(opt, x):
    uids, iids,type,month = list(zip(*x))
    uids = list(uids)
    iids = list(iids)
    type = list(type)
    month= list(month)

    # a = list(map(int, a))

    user_reviews = opt.users_review_list[uids]
    user_item_id = opt.user_item_id_list[uids]  # 检索出该user对应的item id
    # user_item_type = opt.user_item_type_list[uids]  # 检索出该user对应的type
    # user_item_month = opt.user_item_month_list[uids]  # 检索出该user对应的month
    user_item_ratio = opt.user_item_ratio_list[uids]
    user_doc = opt.user_doc[uids]

    item_reviews = opt.items_review_list[iids]
    item_user_id = opt.item_user_id_list[iids]  # 检索出该item对应的user id
    # item_user_type = opt.item_user_type_list[iids]
    # item_user_month = opt.item_user_month_list[iids]
    item_user_ratio = opt.item_user_ratio_list[iids]
    item_doc = opt.item_doc[iids]

    data = [user_reviews, item_reviews, uids, iids,user_item_id, item_user_id,user_item_ratio,item_user_ratio, user_doc,item_doc,
            type,month]

    ##,user_item_type,user_item_month,item_user_type,item_user_month
    if opt.use_gpu:
        data = list(map(lambda x: torch.LongTensor(x).cuda(), data))
    else:
        data = list(map(lambda x: torch.LongTensor(x), data))
    return data

def model_run(model_name,num_fea,gpu_id,output,ui_merge,**kwargs):
    opt = getattr(config, 'DefaultConfig')()
    opt.model = model_name
    opt.num_fea = num_fea
    opt.gpu_id = gpu_id
    opt.output = output
    opt.ui_merge = ui_merge

    for data in ['total']:

        for k_fold in range(1,opt.k_fold+1): #
            opt.dataset = f'{data}_{k_fold}'
            opt.parse(opt.dataset, kwargs)

            pth_folder = f'results/checkpoints/{opt.dataset}'
            if not os.path.exists(pth_folder):
                os.makedirs(pth_folder, exist_ok=True)
            each_pth_folder = f'results/checkpoints/each/{opt.dataset}'
            if not os.path.exists(each_pth_folder):
                os.makedirs(each_pth_folder, exist_ok=True)

            opt.pth_path = f'{pth_folder}/{model_name}.pth'
            based_on_path = None
            best_res = train(opt, each_pth_folder, based_on_path)
            predict_test(opt)
            print(opt.dataset, opt.model)
    return

"""重新定义带返回值的线程类"""
class MyThread(threading.Thread):
    def __init__(self, func, args):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None

if __name__ == "__main__":

    # run single model
    model_run('DeepCoNN', 1,0,'fm','cat')
    # model_run('D_ATTN', 1, 0,'fm','dot')
    # model_run('NARRE', 2,0,'lfm','dot')
    # model_run('MPCN', 1, 0,'fm','cat')
    # model_run('HANCI', 2, 0,'fm','dot')
    # model_run('DeepRM_TC', 3,0,'fm','cat')

    ####Multithreaded running
    # t1 = MyThread(model_run, args=('DeepCoNN', 1,0,'fm','cat'))
    # t2 = MyThread(model_run, args=('DeepRM_TC', 3,0,'fm','cat'))
    # t3 = MyThread(model_run, args=('NARRE', 2,7,'lfm','dot'))
    # t4 = MyThread(model_run, args=('D_ATTN', 1, 2,'fm','dot'))
    # t5 = MyThread(model_run, args=('MPCN', 1, 7,'fm','cat'))
    # t6 = MyThread(model_run, args=('HANCI', 2, 5,'fm','dot'))
    #
    # #
    # t1.start()
    # t2.start()
    # t3.start()
    # t4.start()
    # t5.start()
    # t6.start()
    #
    #
    # #
    # t1.join()
    # t2.join()
    # t3.join()
    # t4.join()
    # t5.join()
    # t6.join()
