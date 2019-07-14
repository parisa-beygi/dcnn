'''
Data pre process part2
@author:
Chong Chen (cstchenc@163.com)

@ created:
27/8/2017
'''
import codecs
import json
import math
import sys

import numpy as np
import re
import itertools
from collections import Counter

import pymongo
import tensorflow as tf
import csv
import os
import pickle

config = json.loads( open(sys.argv[1]).read() )
# MongoDB part
myclient = pymongo.MongoClient("mongodb://admin:admin123@127.0.0.1")
mydb = myclient["new_{}".format(config['data_name'])]


# TPS_DIR = '../data/patio'
TPS_DIR = config['data_dir']
general_dir = os.path.join(TPS_DIR, config['data_processed_file'])

valid_file = os.path.join(TPS_DIR, '{}_valid.csv'.format(config['data_name']))
test_file = os.path.join(TPS_DIR, '{}_test.csv'.format(config['data_name']))
train_file = os.path.join(TPS_DIR, '{}_train.csv'.format(config['data_name']))

user_review_file = os.path.join(TPS_DIR, 'user_review')
item_review_file = os.path.join(TPS_DIR, 'item_review')


tf.flags.DEFINE_string("valid_data",valid_file, " Data for validation")
tf.flags.DEFINE_string("test_data", test_file, "Data for testing")
tf.flags.DEFINE_string("train_data", train_file, "Data for training")
tf.flags.DEFINE_string("user_review", user_review_file, "User's reviews")
tf.flags.DEFINE_string("item_review", item_review_file, "Item's reviews")

# def clean_str(string):
#     regex = re.compile('[,\.!?"]')
#     string = regex.sub(' ', string)
#     return string.strip()

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z]", " ", string)
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
    return string.strip().lower()

def pad_sentences(u_text,u_len,padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length=u_len
    u_text2={}
    print len(u_text)
    for i in u_text.keys():
        #print i
        sentence = u_text[i]
        if sequence_length>len(sentence):

            num_padding = sequence_length - len(sentence)
            new_sentence = sentence + [padding_word] * num_padding
            u_text2[i]=new_sentence
        else:
            new_sentence = sentence[:sequence_length]
            u_text2[i] = new_sentence

    return u_text2

def build_vocab(sentences1):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts1 = Counter(itertools.chain(*sentences1))
    # Mapping from index to word
    vocabulary_inv1 = [x[0] for x in word_counts1.most_common()]
    vocabulary_inv1 = list(sorted(vocabulary_inv1))
    # Mapping from word to index
    vocabulary1 = {x: i for i, x in enumerate(vocabulary_inv1)}

    return [vocabulary1, vocabulary_inv1]

def build_input_data(u_text, vocabulary_u):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    # l = len(u_text)
    u_text2 = {}
    # encoded_user_text_coll = mydb["encoded_user_text"]
    # encoded_user_text_coll.remove()

    # user_text_coll = mydb["user_text"]
    # u_text = user_text_coll.find()
    # for doc in u_text:
    #     i = doc['user_id']
    #     u_reviews = doc['review']
    #     u = [vocabulary_u[word] for word  in u_reviews]
    #     u_text2[i] = u
    #     # d = {"user_id": i, "review": u}
    #     # y = encoded_user_text_coll.insert_one(d)

    for i in u_text.keys():
        u_reviews = u_text[i]
        u = np.array([vocabulary_u[word] for word  in u_reviews])
        u_text2[i] = u



    # # l = len(i_text)
    # # i_text2 = {}
    # encoded_item_text_coll = mydb["encoded_item_text"]
    # encoded_item_text_coll.remove()
    #
    # item_text_coll = mydb["item_text"]
    # i_text = item_text_coll.find()
    # for doc in i_text:
    #     j = doc['item_id']
    #     i_reviews = doc['review']
    #     i = [vocabulary_i[word] for word  in i_reviews]
    #     # i_text2[j] = i
    #     d = {"item_id": j, "review": i}
    #     y = encoded_item_text_coll.insert_one(d)
    #
    # # for j in i_text.keys():
    # #     i_reviews = i_text[j]
    # #     i = np.array([vocabulary_i[word] for word in i_reviews])
    # #     i_text2[j] = i
    # print 'built input data! (encoded_user_text, encoded_item_text)'
    return u_text2


def load_data(user_review, train_data,valid_data,test_data):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    print '2------------------->Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # u_text, y_train,y_valid,y_test,u_len,uid_train, iid_train, uid_valid,iid_valid, uid_test,iid_test,user_num
    u_text, y_train, y_valid,y_test,u_len,uid_train, iid_train, uid_valid, iid_valid, uid_test, iid_test, user_num=\
        load_data_and_labels(user_review, train_data,valid_data,test_data)

    if not config['save_padded_text']:
        print ('Saving Simple Text')
        user_voc_primary = [x for x in u_text.itervalues()]
        # item_voc_primary = [x for x in i_text.itervalues()]

        vocabulary_user_primary, vocabulary_inv_user_primary = build_vocab(user_voc_primary)
        print len(vocabulary_user_primary)
        # print len(vocabulary_item_primary)
        u_text2 = build_input_data(u_text,vocabulary_user_primary)

        save_text(u_text2, vocabulary_user_primary, vocabulary_inv_user_primary)




    print "load data done"
    u_text = pad_sentences(u_text,u_len)
    print "pad user done"
    # item_voc = pad_sentences('item',i_len)
    # print "pad item done"

    user_voc = [x for x in u_text.itervalues()]
    # item_voc = [x for x in i_text.itervalues()]

    vocabulary_user, vocabulary_inv_user = build_vocab(user_voc)
    print len(vocabulary_user)
    # print len(vocabulary_item)
    u_text = build_input_data(u_text,vocabulary_user)
    y_train = np.array(y_train)
    y_valid = np.array(y_valid)
    y_test = np.array(y_test)
    uid_train = np.array(uid_train)
    uid_valid = np.array(uid_valid)
    uid_test = np.array(uid_test)
    iid_train = np.array(iid_train)
    iid_valid = np.array(iid_valid)
    iid_test = np.array(iid_test)


    return [u_text, u_len, y_train,y_valid,y_test, vocabulary_user, vocabulary_inv_user,
            uid_train, iid_train, uid_valid, iid_valid, uid_test, iid_test, user_num]

def parse(path):
  g = open(path, 'r')
  for l in g:
    yield eval(l)

import resource
import time




def load_data_and_labels(user_review, train_data,valid_data,test_data):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # user_reviews_coll = mydb["user_reviews"]
    # print ('user_reviews_coll num of records: {}'.format(user_reviews_coll.find().count()))
    # item_reviews_coll = mydb["item_reviews"]
    # print ('item_reviews_coll num of records: {}'.format(item_reviews_coll.find().count()))

    # user_voc_primary = []
    # item_voc_primary = []

    u_lens = []
    # i_lens = []

    # mongo part
    # user_text_coll = mydb["user_text"]
    # user_text_coll.remove()

    # item_text_coll = mydb["item_text"]
    # item_text_coll.remove()


    # Load data from files
    f_train_op = open(train_data, "r")
    f_train = parse(train_data)
    f1 = open(user_review)
    # f2 = open(item_review)

    user_reviews = pickle.load(f1)
    # item_reviews = pickle.load(f2)
    print ('user_reviews len: {}'.format(len(user_reviews)))
    # print ('item_reviews len: {}'.format(len(item_reviews)))
    uid_train = []
    iid_train = []
    y_train = []
    u_text = {}
    # i_text = {}
    i = 0
    print('????????????????????/')
    t = 3685638
    print t
    for line in f_train:
        i = i + 1
        # line = line.split(',')
        uid_train.append(int(line[0]))
        iid_train.append(int(line[1]))

        if u_text.has_key(int(line[0])):
            a = 1
        else:
            u_text[int(line[0])] = '<PAD/>'
            for s in user_reviews[int(line[0])]:
                u_text[int(line[0])] = u_text[int(line[0])] + ' ' + s.strip()
            u_text[int(line[0])] = clean_str(u_text[int(line[0])])
            u_text[int(line[0])] = u_text[int(line[0])].split(" ")


        # if i_text.has_key(int(line[1])):
        #     a = 1
        # else:
        #     # i_text[int(line[1])] = '<PAD/>'
        #     # for s in item_reviews[int(line[1])]:
        #     #     i_text[int(line[1])] = i_text[int(line[1])] + ' ' + s.strip()
        #     # i_text[int(line[1])] = clean_str(i_text[int(line[1])])
        #     # i_text[int(line[1])] = i_text[int(line[1])].split(" ")

        y_train.append(float(line[2]))

    print "valid"

    uid_valid = []
    iid_valid = []
    y_valid=[]
    # Todo: original
    # f_valid=open(valid_data)
    f_valid = parse(valid_data)

    for line in f_valid:
        # Todo: original
        # line=line.split(',')
        uid_valid.append(int(line[0]))
        iid_valid.append(int(line[1]))

        if u_text.has_key(int(line[0])):
            a=1
        else:
            u_text[int(line[0])] = '<PAD/>'
            u_text[int(line[0])]=clean_str(u_text[int(line[0])])
            u_text[int(line[0])]=u_text[int(line[0])].split(" ")

        # if i_text.has_key(int(line[1])):
        #     a=1
        # else:
        #     # i_text[int(line[1])] = '<PAD/>'
        #     # i_text[int(line[1])]=clean_str(i_text[int(line[1])])
        #     # i_text[int(line[1])]=i_text[int(line[1])].split(" ")

        y_valid.append(float(line[2]))

    print ('test')
    uid_test = []
    iid_test = []
    y_test=[]
    # Todo: original
    # f_test=open(test_data)
    f_test = parse(test_data)
    for line in f_test:
        # Todo: original
        # line=line.split(',')
        uid_test.append(int(line[0]))
        iid_test.append(int(line[1]))

        if u_text.has_key(int(line[0])):
            a=1
        else:
            u_text[int(line[0])] = '<PAD/>'
            u_text[int(line[0])]=clean_str(u_text[int(line[0])])
            u_text[int(line[0])]=u_text[int(line[0])].split(" ")

        # if i_text.has_key(int(line[1])):
        #     a=1
        # else:
        #     # i_text[int(line[1])] = '<PAD/>'
        #     # i_text[int(line[1])]=clean_str(i_text[int(line[1])])
        #     # i_text[int(line[1])]=i_text[int(line[1])].split(" ")

        y_test.append(float(line[2]))


    print "len"
    u = np.array([len(x) for x in u_text.itervalues()])
    x = np.sort(u)
    u_len = x[int(0.85* len(u)) - 1]



    # i = np.array(i_lens)
    # # i = np.array([len(x) for x in i_text.itervalues()])
    # y = np.sort(i)
    # i_len = y[int(0.85 * len(i)) - 1]
    print "u_len:",u_len
    # print "i_len:",i_len
    user_num = len(u_text)
    # item_num = len(i_text)
    print "user_num:", user_num
    # print "item_num:", item_num
    return [u_text, y_train,y_valid,y_test,u_len,uid_train, iid_train, uid_valid,iid_valid, uid_test,iid_test,user_num]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

# def save_data_file(userid, itemid, y, mode):
#     file_dir ='/media/hdd/parisa/dcnn/DeepCoNN/data/m_insts/general_data/res_{}.txt'.format(mode)
#     f = codecs.open(file_dir, 'w', encoding='utf8')
#     f.write('label q1 q2\n')
#     for u, i, l in zip(userid, itemid, y):
#         f.write(str(l) + ' ' + str(u) + ' ' + str(i) + '\n')

def save_text(u_text, vocabulary_user, vocabulary_inv_user):
    output = open(os.path.join(general_dir, 'user.text'), 'wb')
    pickle.dump(u_text, output)
    #
    # output = open(os.path.join(general_dir, 'item.text'), 'wb')
    # pickle.dump(i_text, output)

    output = open(os.path.join(general_dir, 'user.vocab'), 'wb')
    pickle.dump(vocabulary_user, output)

    # output = open(os.path.join(general_dir, 'item.vocab'), 'wb')
    # pickle.dump(vocabulary_item, output)


    output = open(os.path.join(general_dir, 'user.invvocab'), 'wb')
    pickle.dump(vocabulary_inv_user, output)

    # output = open(os.path.join(general_dir, 'item.invvocab'), 'wb')
    # pickle.dump(vocabulary_inv_item, output)


if __name__ == '__main__':
    # TPS_DIR = '../data/patio'
    FLAGS = tf.flags.FLAGS
    # FLAGS._parse_flags()
    FLAGS(sys.argv)

    print ('FLAGS.train_data, FLAGS.valid_data, FLAGS.test_data')
    print (FLAGS.train_data, FLAGS.valid_data, FLAGS.test_data)
    print ('FLAGS.user_review, FLAGS.item_review')
    print (FLAGS.user_review, FLAGS.item_review)
    print '1-------------->Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # u_text, u_len, y_train, y_valid, y_test, vocabulary_user, vocabulary_inv_user,
    # uid_train, iid_train, uid_valid, iid_valid, uid_test, iid_test, user_num
    u_text, u_len, y_train, y_valid, y_test,vocabulary_user, vocabulary_inv_user, \
    uid_train, iid_train, uid_valid, iid_valid, uid_test, iid_test, user_num = \
        load_data(FLAGS.user_review, FLAGS.train_data, FLAGS.valid_data, FLAGS.test_data)

    if config['save_padded_text']:
        print ('Saving Padded Text')
        save_text(u_text, vocabulary_user, vocabulary_inv_user)

    # general_dir = os.path.join(TPS_DIR, config['data_processed_file'])
    # output = open(os.path.join(general_dir, 'user.text'), 'wb')
    # pickle.dump(u_text, output)
    #
    # output = open(os.path.join(general_dir, 'item.text'), 'wb')
    # pickle.dump(i_text, output)
    #
    # output = open(os.path.join(general_dir, 'user.vocab'), 'wb')
    # pickle.dump(vocabulary_user, output)
    #
    # output = open(os.path.join(general_dir, 'item.vocab'), 'wb')
    # pickle.dump(vocabulary_item, output)
    #
    #
    # output = open(os.path.join(general_dir, 'user.invvocab'), 'wb')
    # pickle.dump(vocabulary_inv_user, output)
    #
    # output = open(os.path.join(general_dir, 'item.invvocab'), 'wb')
    # pickle.dump(vocabulary_inv_item, output)


    print ('len(y_train), len(y_valid), len(y_test)')
    print (len(y_train), len(y_valid), len(y_test))
    np.random.seed(2017)

    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    print ('len(y_train)')
    print (len(y_train))

    userid_train = uid_train[shuffle_indices]
    itemid_train = iid_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    # save_data_file(userid_train, itemid_train, y_train, 'train')
    # save_data_file(uid_valid, iid_valid, y_valid, 'valid')
    # save_data_file(uid_test, iid_test, y_test, 'test')

    data_image = {}
    data_image['train'] = [userid_train, itemid_train, y_train]
    data_image['valid'] = [uid_valid, iid_valid, y_valid]
    data_image['test'] = [uid_test, iid_test, y_test]

    output = open(os.path.join(general_dir, '{}_data.image'.format(config['data_name'])), 'wb')

    pickle.dump(data_image, output)


    y_train = y_train[:, np.newaxis]
    y_valid = y_valid[:, np.newaxis]
    y_test = y_test[:, np.newaxis]

    userid_train = userid_train[:, np.newaxis]
    print ('LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL')
    print userid_train
    print userid_train.shape

    itemid_train = itemid_train[:, np.newaxis]
    userid_valid = uid_valid[:, np.newaxis]
    itemid_valid = iid_valid[:, np.newaxis]

    userid_test = uid_test[:, np.newaxis]
    itemid_test = iid_test[:, np.newaxis]


    batches_train=list(zip( userid_train, itemid_train, y_train))
    batches_valid=list(zip(userid_valid,itemid_valid,y_valid))
    batches_test=list(zip(userid_test,itemid_test,y_test))
    output = open(os.path.join(TPS_DIR, '{}.train'.format(config['data_name'])), 'wb')
    pickle.dump(batches_train,output)
    output = open(os.path.join(TPS_DIR, '{}.valid'.format(config['data_name'])), 'wb')
    pickle.dump(batches_valid,output)
    output = open(os.path.join(TPS_DIR, '{}.test'.format(config['data_name'])), 'wb')
    pickle.dump(batches_test,output)

    para={}
    para['user_num']=user_num
    # para['item_num']=item_num
    print ('******************************')
    print (user_num)
    print (u_len)
    print (u_text[0].shape[0])
    para['user_length']=u_text[0].shape[0]
    # para['item_length'] = i_text[0].shape[0]
    para['user_vocab'] = vocabulary_user
    # para['item_vocab'] = vocabulary_item
    para['train_length']=len(y_train)
    para['valid_length']=len(y_valid)
    para['test_length']=len(y_test)
    para['u_text'] = u_text
    # para['i_text'] = i_text
    output = open(os.path.join(TPS_DIR, '{}.upara'.format(config['data_name'])), 'wb')

    pickle.dump(para, output)










