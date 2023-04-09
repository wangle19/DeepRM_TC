# -*- coding: utf-8 -*-
import json
import pandas as pd
import re
import sys
import os
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from operator import itemgetter
import gensim
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
P_REVIEW = 0.5
MAX_DF = 0.7
MAX_VOCAB = 50000
DOC_LEN = 500
from collections import Counter
import string
from framework import config
opt = getattr(config, 'DefaultConfig')()

def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))


def get_count(data, id):
    ids = set(data[id].tolist())
    return ids


def numerize(data):
    uid = list(map(lambda x: user2id[x], data['user_id']))
    iid = list(map(lambda x: item2id[x], data['item_id']))
    type=list(map(lambda x: type2id[x], data['type']))
    month = list(map(lambda x: month2id[x], data['month']))
    data['user_id'] = uid
    data['item_id'] = iid
    data['type'] = type
    data['month'] = month
    return data

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"sssss ", " ", string)
    return string.strip().lower()


def bulid_vocbulary(xDict):
    rawReviews = []
    for (id, text) in xDict.items():
        rawReviews.append(' '.join(text))
    return rawReviews


def build_doc(u_reviews_dict, i_reviews_dict):
    '''
    1. extract the vocab
    2. fiter the reviews and documents of users and items
    '''
    u_reviews = []
    for ind in range(len(u_reviews_dict)):
        u_reviews.append(' <SEP> '.join(u_reviews_dict[ind]))

    i_reviews = []
    for ind in range(len(i_reviews_dict)):
        i_reviews.append('<SEP>'.join(i_reviews_dict[ind]))

    vectorizer = TfidfVectorizer(max_df=MAX_DF, max_features=MAX_VOCAB)
    vectorizer.fit(u_reviews)
    vocab = vectorizer.vocabulary_
    vocab[MAX_VOCAB] = '<SEP>'

    def clean_review(rDict):
        new_dict = {}
        for k, text in rDict.items():
            new_reviews = []
            for r in text:
                words = ' '.join([w for w in r.split() if w in vocab])
                new_reviews.append(words)
            new_dict[k] = new_reviews
        return new_dict

    def clean_doc(raw):
        new_raw = []
        for line in raw:
            review = [word for word in line.split() if word in vocab]
            if len(review) > DOC_LEN:
                review = review[:DOC_LEN]
            new_raw.append(review)
        return new_raw

    u_reviews_dict = clean_review(u_reviews_dict)
    i_reviews_dict = clean_review(i_reviews_dict)

    u_doc = clean_doc(u_reviews)
    i_doc = clean_doc(i_reviews)

    return vocab, u_doc, i_doc, u_reviews_dict, i_reviews_dict


def countNum(xDict):
    minNum = 100
    maxNum = 0
    sumNum = 0
    maxSent = 0
    minSent = 3000
    # pSentLen = 0
    ReviewLenList = []
    SentLenList = []
    for (i, text) in xDict.items():
        sumNum = sumNum + len(text)
        if len(text) < minNum:
            minNum = len(text)
        if len(text) > maxNum:
            maxNum = len(text)
        ReviewLenList.append(len(text))
        for sent in text:
            # SentLenList.append(len(sent))
            if sent != "":
                wordTokens = sent.split()
            if len(wordTokens) > maxSent:
                maxSent = len(wordTokens)
            if len(wordTokens) < minSent:
                minSent = len(wordTokens)
            SentLenList.append(len(wordTokens))
    averageNum = sumNum // (len(xDict))

    x = np.sort(SentLenList)
    xLen = len(x)
    pSentLen = x[int(P_REVIEW * xLen) - 1]
    x = np.sort(ReviewLenList)
    xLen = len(x)
    pReviewLen = x[int(P_REVIEW * xLen) - 1]
    # pReviewLen = averageNum
    return minNum, maxNum, averageNum, maxSent, minSent, pReviewLen, pSentLen


def dateset_generation(data_train, data_test, dataset_name, cv,typeNum_all,monthNum_all):
    save_folder = f'pro_dataset/{dataset_name}_{cv}'
    print(f"数据集名称：{save_folder}")

    if not os.path.exists(save_folder + '/train'):
        os.makedirs(save_folder + '/train')
    if not os.path.exists(save_folder + '/val'):
        os.makedirs(save_folder + '/val')
    if not os.path.exists(save_folder + '/test'):
        os.makedirs(save_folder + '/test')

    uids_train, iids_train = get_count(data_train, 'user_id'), get_count(data_train, 'item_id')
    userNum = len(uids_train)
    itemNum = len(iids_train)
    print("===============Start: no-preprocess: trainData size======================")
    print("dataNum: {}".format(data_train.shape[0]))
    print("userNum: {}".format(userNum))
    print("itemNum: {}".format(itemNum))
    print("===============End: no-preprocess: trainData size========================")

    uidMiss = []
    iidMiss = []
    if userNum != userNum_all or itemNum != itemNum_all:
        for uid in range(userNum_all):
            if uid not in uids_train:
                uidMiss.append(uid)
        for iid in range(itemNum_all):
            if iid not in iids_train:
                iidMiss.append(iid)
    uid_index = []
    for uid in uidMiss:
        index = data_test.index[data_test['user_id'] == uid].tolist()
        uid_index.extend(index)
    data_train = pd.concat([data_train, data_test.loc[uid_index]])

    iid_index = []
    for iid in iidMiss:
        index = data_test.index[data_test['item_id'] == iid].tolist()
        iid_index.extend(index)
    data_train = pd.concat([data_train, data_test.loc[iid_index]])

    all_index = list(set().union(uid_index, iid_index))
    data_test = data_test.drop(all_index)

    # split validate set aand test set
    data_test, data_val = train_test_split(data_test, test_size=0.5, random_state=1234)
    uidList_train, iidList_train = get_count(data_train, 'user_id'), get_count(data_train, 'item_id')
    userNum = len(uidList_train)
    itemNum = len(iidList_train)
    print("===============Start--process finished: trainData size======================")
    print("dataNum: {}".format(data_train.shape[0]))
    print("userNum: {}".format(userNum))
    print("itemNum: {}".format(itemNum))
    print("===============End-process finished: trainData size========================")

    def extract(data_dict):
        x = []
        y = []
        for i in data_dict.values:
            uid = i[0]
            iid = i[1]
            type = i[4]
            month = i[5]
            x.append([uid, iid, type, month])
            y.append(float(i[2]))
        return x, y

    x_train, y_train = extract(data_train)
    x_val, y_val = extract(data_val)
    x_test, y_test = extract(data_test)

    np.save(f"{save_folder}/train/Train.npy", x_train)
    np.save(f"{save_folder}/train/Train_Score.npy", y_train)
    np.save(f"{save_folder}/val/Val.npy", x_val)
    np.save(f"{save_folder}/val/Val_Score.npy", y_val)
    np.save(f"{save_folder}/test/Test.npy", x_test)
    np.save(f"{save_folder}/test/Test_Score.npy", y_test)

    print(now())
    print(f"Train data size: {len(x_train)}")
    print(f"Val data size: {len(x_val)}")
    print(f"Test data size: {len(x_test)}")

    print(f"-" * 60)
    print(f"{now()} Step3: Construct the vocab and user/item reviews from training set.")
    # 2: build vocabulary only with train dataset
    user_reviews_dict = {}
    user_iid_dict = {}
    user_itype_dict = {}
    user_imonth_dict = {}
    user_iscore_dict = {}

    item_reviews_dict = {}
    item_uid_dict = {}
    item_utype_dict = {}
    item_umonth_dict = {}
    item_uscore_dict = {}


    for i in data_train.values:
        str_review = clean_str(i[3].encode('ascii', 'ignore').decode('ascii'))

        if len(str_review.strip()) == 0:
            str_review = "<unk>"

        if i[0] in user_reviews_dict:
            user_reviews_dict[i[0]].append(str_review)
            user_iid_dict[i[0]].append(i[1])
            user_itype_dict[i[0]].append(i[4])
            user_imonth_dict[i[0]].append(i[5])
            user_iscore_dict[i[0]].append(i[2])

        else:
            user_reviews_dict[i[0]] = [str_review]
            user_iid_dict[i[0]] = [i[1]]
            user_itype_dict[i[0]] = [i[4]]
            user_imonth_dict[i[0]] = [i[5]]
            user_iscore_dict[i[0]] = [i[2]]

        if i[1] in item_reviews_dict:
            item_reviews_dict[i[1]].append(str_review)
            item_uid_dict[i[1]].append(i[0])
            item_utype_dict[i[1]].append(i[4])
            item_umonth_dict[i[1]].append(i[5])
            item_uscore_dict[i[1]].append(i[2])
        else:
            item_reviews_dict[i[1]] = [str_review]
            item_uid_dict[i[1]] = [i[0]]
            item_utype_dict[i[1]] = [i[4]]
            item_umonth_dict[i[1]] = [i[5]]
            item_uscore_dict[i[1]] = [i[2]]

    vocab, user_review2doc, item_review2doc, user_reviews_dict, item_reviews_dict = build_doc(user_reviews_dict,
                                                                                              item_reviews_dict)
    word_index = {}
    word_index['<unk>'] = 0
    for i, w in enumerate(vocab.keys(), 1):
        word_index[w] = i

    f = open(f'{save_folder}/train/key2index.txt', 'w')
    f.write(str(word_index))
    f.close()

    print(f"The vocab size: {len(word_index)}")
    print(f"Average user document length: {sum([len(i) for i in user_review2doc]) / len(user_review2doc)}")
    print(f"Average item document length: {sum([len(i) for i in item_review2doc]) / len(item_review2doc)}")

    print(now())
    u_minNum, u_maxNum, u_averageNum, u_maxSent, u_minSent, u_pReviewLen, u_pSentLen = countNum(
        user_reviews_dict)
    print("用户最少有{}个评论,最多有{}个评论，平均有{}个评论, " \
          "句子最大长度{},句子的最短长度{}，" \
          "设定用户评论个数为{}： 设定句子最大长度为{}".format(u_minNum, u_maxNum, u_averageNum, u_maxSent, u_minSent,
                                            u_pReviewLen, u_pSentLen))
    i_minNum, i_maxNum, i_averageNum, i_maxSent, i_minSent, i_pReviewLen, i_pSentLen = countNum(
        item_reviews_dict)
    print("商品最少有{}个评论,最多有{}个评论，平均有{}个评论," \
          "句子最大长度{},句子的最短长度{}," \
          ",设定商品评论数目{}, 设定句子最大长度为{}".format(i_minNum, i_maxNum, i_averageNum, u_maxSent, i_minSent,
                                            i_pReviewLen, i_pSentLen))
    print("最终设定句子最大长度为(取最大值)：{}".format(max(u_pSentLen, i_pSentLen)))
    # ########################################################################################################
    maxSentLen = max(u_pSentLen, i_pSentLen)
    minSentlen = 1

    userReview2Index = []
    userDoc2Index = []
    user_iid_list = []
    user_itype_list = []
    user_imonth_list = []
    user_iratio_list = []

    print(f"-" * 60)
    print(f"{now()} Step4: padding all the text and id lists and save into npy.")

    def padding_text(textList, num):
        new_textList = []
        if len(textList) >= num:
            new_textList = textList[:num]
        else:
            padding = [[0] * len(textList[0]) for _ in range(num - len(textList))]
            new_textList = textList + padding
        return new_textList

    def padding_ids(iids, num, pad_id):
        if len(iids) >= num:
            new_iids = iids[:num]
        else:
            new_iids = iids + [pad_id] * (num - len(iids))
        return new_iids

    def score_ratio(u_iscores):
        rating_percent = [10, 10, 10, 10, 10]
        result1 = Counter(u_iscores)
        for i in result1:
            a = int(float(i))
            # a = int(i)
            rating_percent[a - 1] = (result1[i]) / len(u_iscores)
        return rating_percent

    def padding_doc(doc):
        pDocLen = DOC_LEN
        new_doc = []
        for d in doc:
            if len(d) < pDocLen:
                d = d + [0] * (pDocLen - len(d))
            else:
                d = d[:pDocLen]
            new_doc.append(d)

        return new_doc, pDocLen

    for i in range(userNum):
        textList = user_reviews_dict[i]
        u_iids = user_iid_dict[i]
        u_itypes = user_itype_dict[i]
        u_imonths = user_imonth_dict[i]
        u_iscores = user_iscore_dict[i]
        u_reviewList = []

        user_iid_list.append(padding_ids(u_iids, u_pReviewLen, itemNum))
        user_itype_list.append(padding_ids(u_itypes, u_pReviewLen, typeNum_all))
        user_imonth_list.append(padding_ids(u_imonths, u_pReviewLen, monthNum_all))
        user_iratio_list.append(score_ratio(u_iscores))
        doc2index = [word_index[w] for w in user_review2doc[i]]

        for text in textList:
            wordTokens = text.strip().split()
            if len(wordTokens) == 0:
                wordTokens = ['<unk>']
            text2index = [word_index[w] for w in wordTokens]
            if len(text2index) < maxSentLen:
                text2index = text2index + [0] * (maxSentLen - len(text2index))
            else:
                text2index = text2index[:maxSentLen]
            u_reviewList.append(text2index)

        userReview2Index.append(padding_text(u_reviewList, u_pReviewLen))
        userDoc2Index.append(doc2index)


    userDoc2Index, userDocLen = padding_doc(userDoc2Index)
    print(f"user document length: {userDocLen}")

    itemReview2Index = []
    itemDoc2Index = []
    item_uid_list = []
    item_utype_list = []
    item_umonth_list =[]
    item_uratio_list = []
    for i in range(itemNum):
        textList = item_reviews_dict[i]
        i_uids = item_uid_dict[i]
        i_utypes = item_utype_dict[i]
        i_umonths = item_umonth_dict[i]
        i_uscores = item_uscore_dict[i]
        i_reviewList = []  # 待添加
        item_uid_list.append(padding_ids(i_uids, i_pReviewLen, userNum))
        item_utype_list.append(padding_ids(i_utypes, i_pReviewLen, typeNum_all))
        item_umonth_list.append(padding_ids(i_umonths, i_pReviewLen, monthNum_all))
        item_uratio_list.append(score_ratio(i_uscores))
        doc2index = [word_index[w] for w in item_review2doc[i]]

        for text in textList:
            text2index = []
            wordTokens = text.strip().split()
            if len(wordTokens) == 0:
                wordTokens = ['<unk>']
            text2index = [word_index[w] for w in wordTokens]
            if len(text2index) < maxSentLen:
                text2index = text2index + [0] * (maxSentLen - len(text2index))
            else:
                text2index = text2index[:maxSentLen]
            i_reviewList.append(text2index)
        itemReview2Index.append(padding_text(i_reviewList, i_pReviewLen))
        itemDoc2Index.append(doc2index)

    itemDoc2Index, itemDocLen = padding_doc(itemDoc2Index)
    print(f"item document length: {itemDocLen}")

    # item_reviews = itemDoc2Index[data_train['item_id'].tolist()]
    # data_train.to_csv(f"{save_folder}/Train_dataframe.csv", header=True, index=False)
    # data_test.to_csv(f'{save_folder}/Test_dataframe.csv', header=True, index=False)
    # data_val.to_csv(f'{save_folder}/Val_dataframe.csv', header=True, index=False)

    print("-" * 60)
    print(f"{now()} start writing npy...")

    np.save(f"{save_folder}/train/userReview2Index.npy", userReview2Index)
    np.save(f"{save_folder}/train/user_item2id.npy", user_iid_list)
    np.save(f"{save_folder}/train/user_item2type.npy", user_itype_list)
    np.save(f"{save_folder}/train/user_item2month.npy", user_imonth_list)
    np.save(f"{save_folder}/train/user_item2ratio.npy", user_iratio_list)
    np.save(f"{save_folder}/train/userDoc2Index.npy", userDoc2Index)

    np.save(f"{save_folder}/train/itemReview2Index.npy", itemReview2Index)
    np.save(f"{save_folder}/train/item_user2id.npy", item_uid_list)
    np.save(f"{save_folder}/train/item_user2type.npy", item_utype_list)
    np.save(f"{save_folder}/train/item_user2month.npy", item_umonth_list)
    np.save(f"{save_folder}/train/item_user2ratio.npy", item_uratio_list)
    np.save(f"{save_folder}/train/itemDoc2Index.npy", itemDoc2Index)

    print(f"{now()} write finised")

    # #####################################################3,产生w2v############################################

    # PRE_W2V_BIN_PATH = 'embedding_models/' + dataset_name + '_embedding_model.bin'  # the pre-trained word2vec files
    PRE_W2V_BIN_PATH = ''
    if len(PRE_W2V_BIN_PATH) == 0:
        print("Warning: the word embedding file is not provided, will be initialized randomly")

    f = open(f'{save_folder}/train/key2index.txt', 'r')
    word_index = eval(f.read())  # 读取的str转换为字典
    print("-" * 60)
    print(f"{now()} Step5: start word embedding mapping...")
    vocab_item = sorted(word_index.items(), key=itemgetter(1))
    w2v = []
    out = 0

    if PRE_W2V_BIN_PATH:
        pre_word2v = gensim.models.KeyedVectors.load_word2vec_format(PRE_W2V_BIN_PATH, binary=True)
    else:
        pre_word2v = {}
        print(f"{now()} 开始提取embedding")
        for word, key in vocab_item:
            if word in pre_word2v:
                w2v.append(pre_word2v[word])
            else:
                out += 1
                w2v.append(np.random.uniform(-1.0, 1.0, (word_dim,)))
    print("############################")
    print(f"out of vocab: {out}")
    # print w2v[1000]
    print(f"w2v size: {len(w2v)}")
    print("############################")
    w2vArray = np.array(w2v)
    print(w2vArray.shape)
    np.save(f"{save_folder}/train/w2v_" + str(word_dim) + ".npy", w2v)
    print(f"{save_folder}/train/w2v_" + str(word_dim) + ".npy")
    end_time = time.time()
    print(f"{now()} all steps finised, cost time: {end_time - start_time:.4f}s")

    para_dic = {}
    para_dic['train_data_size'] = len(x_train)
    para_dic['test_data_size'] = len(x_test)
    para_dic['val_data_size'] = len(x_val)

    para_dic['vocab_size'] = len(word_index)
    para_dic['r_max_len'] = max(u_pSentLen, i_pSentLen)
    para_dic['u_max_r'] = u_pReviewLen
    para_dic['i_max_r'] = i_pReviewLen

    para_dic['user_num'] = userNum_all
    para_dic['item_num'] = itemNum_all
    para_dic['type_num'] = typeNum_all
    para_dic['month_num'] = monthNum_all

    # # 保存
    # f = open(f'results/config/{dataset_name}_{cv}_para_dict_rating.txt', 'w')
    f = open(f'{save_folder}/para_dict_rating.txt', 'w')
    f.write(str(para_dic))
    f.close()
    #


if __name__ == '__main__':

    start_time = time.time()

    for data_file in ['Software_5.json']:
    # for data_file in ['Gift_Cards_5.json']:
        word_dim = 300
        if data_file[-1] == 'v':
            data = pd.read_csv('given_data/'+data_file)
            print(data.shape)
            print(data.columns)
            dataset_name = data_file[:-4]
            print(dataset_name)

            #删除表中任何含有NaN的行
            data = data.dropna(axis=0, how='any')
            if len(data.columns) == 9:
                data.columns = ['user_id', 'item_id', 'rating', 'reviews', 'review_time', 'helpful', 'month','year', 'type']
            else:
                data.columns = ['g', 'd', 'r', 'attraction_city','user_id', 'item_id', 'rating', 'reviews', 'review_time', 'helpful', 'month','year', 'type']

            data = data.astype(str)
            type_num = data['type'].unique()
            print("type_num",type_num)

        else:
            f = open('given_data/type_name_dict.txt', 'r')
            a = f.read()
            type_name_dict = eval(a)
            type_name = type_name_dict[data_file]
            file = open(f'given_data/{data_file}', errors='ignore')
            dataset_name = data_file[:-5]
            print(dataset_name)

            users_id = []
            items_id = []
            ratings = []
            reviews = []
            review_times = []
            types = []

            for line in file:
                js = json.loads(line)
                if str(js['reviewerID']) == 'unknown':
                    print("unknown user id")
                    continue
                if str(js['asin']) == 'unknown':
                    print("unkown item id")
                    continue
                try:
                    reviews.append(js['reviewText'])
                    users_id.append(str(js['reviewerID']))
                    items_id.append(str(js['asin']))
                    ratings.append(str(js['overall']))
                    review_times.append(str(js['reviewTime']))
                    types.append(js['style'][type_name])
                except:
                    continue
            data_frame = {'user_id': pd.Series(users_id), 'item_id': pd.Series(items_id),
                          'rating': pd.Series(ratings), 'reviews': pd.Series(reviews),
                          'review_times': pd.Series(review_times), 'type': pd.Series(types)}
            data = pd.DataFrame(data_frame)  # [['user_id', 'item_id', 'ratings', 'reviews']]
            data.drop_duplicates(subset=None, keep='first', inplace=True, ignore_index=True)
            data['review_times'] = data['review_times'].apply(lambda x: x.replace(',', ' '))
            data[['month', 'day', 'year']] = data.review_times.str.split(expand=True)
            print(data.dtypes)

        data = data[['user_id', 'item_id', 'rating', 'reviews', 'type', 'month', 'year']]
        uidList, iidList = get_count(data, 'user_id'), get_count(data, 'item_id')
        type_List, month_List= get_count(data, 'type'),get_count(data, 'month')
        userNum_all = len(uidList)
        itemNum_all = len(iidList)
        typeNum_all = len(type_List)
        monthNum_all = len(month_List)
        print("===============Start:all  rawData size======================")
        print(f"dataNum: {data.shape[0]}")
        print(f"userNum: {userNum_all}")
        print(f"itemNum: {itemNum_all}")
        print(f"typeNum: {typeNum_all}")
        print(f"monthNum: {monthNum_all}")
        print(f"data densiy: {data.shape[0]/float(userNum_all * itemNum_all):.4f}")
        print("===============End: rawData size========================")

        user2id = dict((uid, i) for(i, uid) in enumerate(uidList))
        item2id = dict((iid, i) for(i, iid) in enumerate(iidList))
        type2id = dict((type, i) for (i, type) in enumerate(type_List))
        month2id = dict((month, i) for (i, month) in enumerate(month_List))
        data = numerize(data)

        print(f"-"*60)
        print(f"{now()} Step2: split datsets into train/val/test, save into npy data")

        if opt.k_fold ==1:
            data_train, data_test = train_test_split(data, test_size=0.2,shuffle=False)
            dateset_generation(data_train, data_test, dataset_name, opt.k_fold,typeNum_all,monthNum_all)
        else:
            kf = KFold(n_splits=opt.k_fold, shuffle=True, random_state=1)
            cv = 0
            for train_index, test_index in kf.split(data):
                cv+=1
                data_train, data_test = data.iloc[train_index], data.iloc[test_index]
                dateset_generation(data_train, data_test, dataset_name, cv,typeNum_all,monthNum_all)