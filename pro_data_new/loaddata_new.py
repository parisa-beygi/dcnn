'''
Data pre process

@author:
Parisa

'''
import gzip
import os
import json
import sys
import time

import pandas as pd
import pickle
import numpy as np
import pymongo

config = json.loads( open(sys.argv[1]).read() )
query_fit_max_len = 400

# TPS_DIR = '../data/patio'
TPS_DIR = config['data_dir']
general_dir = os.path.join(TPS_DIR, config['data_processed_file'])

TP_file = os.path.join(TPS_DIR, config['data_file_name'])
compressed_TP_file = os.path.join(TPS_DIR, config['compressed_data_file_name'])

shuffle_turn = config['shuffle_turn']


def get_count(tp, id):
    playcount_groupbyid = tp[[id, 'ratings']].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count

if not shuffle_turn:
    print "generating new train,valid,test sets\n"

    def parse(path):
      g = gzip.open(path, 'r')
      for l in g:
        yield eval(l)

    # Todo: original
    f= open(TP_file)
    users_id=[]
    items_id=[]
    ratings=[]
    reviews=[]
    np.random.seed(2017)

    # mygen = parse(compressed_TP_file)
    # for js in mygen:
    #     if str(js['reviewerID'])=='unknown':
    #         print "unknown"
    #         continue
    #     if str(js['asin'])=='unknown':
    #         print "unknown2"
    #         continue
    #     reviews.append(js['reviewText'])
    #     users_id.append(str(js['reviewerID'])+',')
    #     items_id.append(str(js['asin'])+',')
    #     ratings.append(str(js['overall']))

    # Todo: original
    for line in f:
        js=json.loads(line)
        if str(js['reviewerID'])=='unknown':
            print ("unknown")
            continue
        if str(js['asin'])=='unknown':
            print ("unknown2")
            continue
        reviews.append(js['reviewText'])
        users_id.append(str(js['reviewerID'])+',')
        items_id.append(str(js['asin'])+',')
        ratings.append(str(js['overall']))
    data=pd.DataFrame({'user_id':pd.Series(users_id),
                       'item_id':pd.Series(items_id),
                       'ratings':pd.Series(ratings),
                       'reviews':pd.Series(reviews)})[['user_id','item_id','ratings','reviews']]

    print ('line 49')
    # def get_count(tp, id):
    #     playcount_groupbyid = tp[[id, 'ratings']].groupby(id, as_index=False)
    #     count = playcount_groupbyid.size()
    #     return count
    usercount, itemcount = get_count(data, 'user_id'), get_count(data, 'item_id')
    print ('user_count:{}, item_count:{}'.format(usercount, itemcount))
    unique_uid = usercount.index
    unique_sid = itemcount.index
    print ('unique_user_count: {}, unique_item_count: {}'.format(unique_uid, unique_sid))
    item2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
    user2id = dict((uid, i) for (i, uid) in enumerate(unique_uid))
    print ('user count ::: {} item_count ::: {}'.format(len(user2id), len(item2id)))
    def numerize(tp):
        uid = map(lambda x: user2id[x], tp['user_id'])
        sid = map(lambda x: item2id[x], tp['item_id'])
        tp['user_id'] = uid
        tp['item_id'] = sid
        return tp

    data=numerize(data)
    tp_rating=data[['user_id','item_id','ratings']]
    print ('line 68')


    n_ratings = tp_rating.shape[0]
    test = np.random.choice(n_ratings, size=int(0.20 * n_ratings), replace=False)
    test_idx = np.zeros(n_ratings, dtype=bool)
    test_idx[test] = True
    print ('line 75')

    tp_1 = tp_rating[test_idx]
    tp_train= tp_rating[~test_idx]
    print ('len(tp_train)')
    print (len(tp_train))

    data2=data[test_idx]
    data=data[~test_idx]


    n_ratings = tp_1.shape[0]
    test = np.random.choice(n_ratings, size=int(0.50 * n_ratings), replace=False)

    test_idx = np.zeros(n_ratings, dtype=bool)
    test_idx[test] = True

    tp_test = tp_1[test_idx]
    tp_valid = tp_1[~test_idx]
    print ('len(tp_valid)')
    print (len(tp_valid))
    print ('len(tp_test)')
    print (len(tp_test))

    tp_train.to_csv(os.path.join(TPS_DIR, '{}_train.csv'.format(config['data_name'])), index=False,header=None)
    tp_valid.to_csv(os.path.join(TPS_DIR, '{}_valid.csv'.format(config['data_name'])), index=False,header=None)
    tp_test.to_csv(os.path.join(TPS_DIR, '{}_test.csv'.format(config['data_name'])), index=False,header=None)

    # MongoDB part
    # myclient = pymongo.MongoClient("mongodb://admin:admin123@127.0.0.1")
    # mydb = myclient["dcnn_database_{}".format(config['data_name'])]
    # print ("database dcnn_database_{} loaded!".format(config['data_name']))
    # user_reviews_coll = mydb["user_reviews"]
    # user_reviews_coll.remove()
    # print ('user_reviews coll: {}'.format(user_reviews_coll.find().count()))
    # item_reviews_coll = mydb["item_reviews"]
    # item_reviews_coll.remove()
    # print ('item_reviews coll: {}'.format(item_reviews_coll.find().count()))



    # Saving snapshot of data and data2
    output = open(os.path.join(TPS_DIR, 'data.dt'), 'wb')
    pickle.dump(data, output)

    output = open(os.path.join(TPS_DIR, 'data2.dt'), 'wb')
    pickle.dump(data2, output)

else:
    print 'Loaded saved data and data2'
    # Loading data and data2
    data_path = os.path.join(TPS_DIR, 'data.dt')
    data_file = open(data_path, 'rb')
    data = pickle.load(data_file)

    data2_path = os.path.join(TPS_DIR, 'data2.dt')
    data2_file = open(data2_path, 'rb')
    data2 = pickle.load(data2_file)


user_reviews={}
item_reviews={}
user_rid={}
item_rid={}

print (len(data))
print (type(data))
print (type(data.values))
print (data.values)
print (data.values.shape)
print (len(data2))
total = len(data)
total2 = len(data2)

log_start_time = time.time()
counter_1 = 0
main_start_time = time.time()

for i in data.values:
    # start = time.time()
    # counter_1 += 1
    # user_query = {"user_id" : i[0]}
    # user_doc = user_reviews_coll.find_one(user_query)
    # if user_doc is not None:
    #     # print ('user doc {} len: {}'.format(i[0], len(user_doc)))
    #     l = user_doc['review']
    #     # print (l)
    #     l.append(i[3])
    #     newvalues = {"$set": {"review": l}}
    #     user_reviews_coll.update_one(user_query, newvalues)
    #
    #     user_rid[i[0]].append(i[1])
    # else:
    #     user_rid[i[0]]=[i[1]]
    #
    #     d = {"user_id": i[0], "review": [i[3]]}
    #     y = user_reviews_coll.insert_one(d)
    #
    #
    # item_query = {"item_id": i[1]}
    # item_doc = item_reviews_coll.find_one(item_query)
    # if item_doc is not None:
    #     # print ('item doc {} len: {}'.format(i[1], type(item_doc)))
    #     l = item_doc['review']
    #     # print (l)
    #     l.append(i[3])
    #     newvalues = {"$set": {"review": l}}
    #     item_reviews_coll.update_one(item_query, newvalues)
    #
    #     item_rid[i[1]].append(i[0])
    # else:
    #     d = {"item_id": i[1], "review": [i[3]]}
    #     y = item_reviews_coll.insert_one(d)
    #
    #     item_rid[i[1]]=[i[0]]
    # end = time.time()
    # print('counter :{}/{} -- users {} minutes'.format(counter_1, total, (end - start)/60))
    #
    # if (time.time() - main_start_time)/60 > 5:
    #     print ('----------- {} minutes elapsed -----------!'.format((time.time() - log_start_time)/60))
    #     print ('<<<<<<<<<<< 5 minutes >>>>>>>>>>>>\n')
    #     main_start_time = time.time()



    if user_reviews.has_key(i[0]):
        user_reviews[i[0]].append(i[3])
        user_rid[i[0]].append(i[1])
    else:
        user_rid[i[0]]=[i[1]]
        user_reviews[i[0]]=[i[3]]
    if item_reviews.has_key(i[1]):
        item_reviews[i[1]].append(i[3])
        item_rid[i[1]].append(i[0])
    else:
        item_reviews[i[1]] = [i[3]]
        item_rid[i[1]]=[i[0]]
# log_start_time = time.time()
# main_start_time = time.time()
# counter = 0
for i in data2.values:
    # start = time.time()
    # counter += 1
    #
    # user_query = {"user_id" : i[0]}
    # user_doc = user_reviews_coll.find_one(user_query)
    # if user_doc is not None:
    #     l =1
    # else:
    #     user_rid[i[0]]=[0]
    #     d = {"user_id": i[0], "review": ['0']}
    #     y = user_reviews_coll.insert_one(d)
    #
    #
    # item_query = {"item_id" : i[1]}
    # item_doc = item_reviews_coll.find_one(item_query)
    # if item_doc is not None:
    #     l=1
    # else:
    #     item_rid[i[1]]=[0]
    #     d = {"item_id": i[1], "review": ['0']}
    #     y = item_reviews_coll.insert_one(d)
    #
    # end = time.time()
    # print('counter :{}/{} -- items {} minutes'.format(counter, total2, (end - start)/60))
    #
    # if (time.time() - main_start_time)/60 > 5:
    #     print ('----------- {} minutes elapsed -----------!'.format((time.time() - log_start_time)/60))
    #     print ('<<<<<<<<<<< 5 minutes >>>>>>>>>>>>\n')
    #     main_start_time = time.time()



    if user_reviews.has_key(i[0]):
        l=1
    else:
        user_rid[i[0]]=[0]
        user_reviews[i[0]]=['0']
    if item_reviews.has_key(i[1]):
        l=1
    else:
        item_reviews[i[1]] = ['0']
        item_rid[i[1]]=[0]

print (item_reviews[11])
print ('size of user_reviews: %s (bytes)' % (sys.getsizeof(user_reviews)))
print ('size of item_reviews: %s (bytes)' % (sys.getsizeof(item_reviews)))

print ("num of users, >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print ('num of user_reviews: %s' % (len(user_reviews)))
print ('num of item_reviews: %s' % (len(item_reviews)))

def get_sparse_ids(reviews):
    dict = {}
    for u in reviews:
        l = len(reviews[u])
        if l < 6:
            if not dict.has_key(l):
                dict[l] = []
            dict[l].append(int(u))
    return dict

user_dict = get_sparse_ids(user_reviews)
item_dict = get_sparse_ids(item_reviews)

output = open(os.path.join(general_dir, 'user_dict.sparse'), 'wb')
pickle.dump(user_dict, output)

output = open(os.path.join(general_dir, 'item_dict.sparse'), 'wb')
pickle.dump(item_dict, output)

# print ("Sparse users>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", len(user_dict[5]))
# for i in user_dict[5]:
#     print i

import random
def shuffle_first_reviews(user_reviews):
    for u in user_reviews:
        u_words_num = 0
        for rev_id in range(len(user_reviews[u])):
            if u_words_num < query_fit_max_len:
                u_words_num += len(user_reviews[u][rev_id].split())
            else:
                temp = user_reviews[u][0:rev_id+1]
                random.shuffle(temp)
                user_reviews[u][0:rev_id+1] = temp


if shuffle_turn:
    shuffle_first_reviews(user_reviews)
    shuffle_first_reviews(item_reviews)


pickle.dump(user_reviews, open(os.path.join(TPS_DIR, 'user_review'), 'wb'))
pickle.dump(item_reviews, open(os.path.join(TPS_DIR, 'item_review'), 'wb'))
pickle.dump(user_rid, open(os.path.join(TPS_DIR, 'user_rid'), 'wb'))
pickle.dump(item_rid, open(os.path.join(TPS_DIR, 'item_rid'), 'wb'))

usercount, itemcount = get_count(data, 'user_id'), get_count(data, 'item_id')


print (np.sort(np.array(usercount.values)))

print (np.sort(np.array(itemcount.values)))

# user_docs = user_reviews_coll.find()
# item_docs = item_reviews_coll.find()
# print (user_docs.count(), item_docs.count())
