'''
model_train
@author:
Chong Chen (cstchenc@163.com)

@ created:
27/8/2017
@references:
Lei Zheng, Vahid Noroozi, and Philip S Yu. 2017. Joint deep modeling of users and items using reviews for recommendation.
In WSDM. ACM, 425-434.
'''
import json
import os
import sys
import time

import numpy as np
import pymongo
import tensorflow as tf
import math
from tensorflow.contrib import learn
import datetime

import pickle
import DeepCoNN

config = json.loads( open(sys.argv[1]).read() )
# MongoDB part
# myclient = pymongo.MongoClient("mongodb://admin:admin123@127.0.0.1")
# mydb = myclient["dcnn_database_{}".format(config['data_name'])]


TPS_DIR = config['data_dir']
valid_file = os.path.join(TPS_DIR, '{}.valid'.format(config['data_name']))
user_para_file = os.path.join(TPS_DIR, '{}.upara'.format(config['data_name']))
item_para_file = os.path.join(TPS_DIR, '{}.ipara'.format(config['data_name']))
train_file = os.path.join(TPS_DIR, '{}.train'.format(config['data_name']))
test_file = os.path.join(TPS_DIR, '{}.test'.format(config['data_name']))

tf.flags.DEFINE_string("word2vec", "./data/google.bin", "Word2vec file with pre-trained embeddings (default: None)")
tf.flags.DEFINE_string("valid_data",valid_file, " Data for validation")
tf.flags.DEFINE_string("user_para_data", user_para_file, "Data parameters")
tf.flags.DEFINE_string("item_para_data", item_para_file, "Data parameters")
tf.flags.DEFINE_string("train_data", train_file, "Data for training")
tf.flags.DEFINE_string('test_data', test_file, "Data for testing")

# ==================================================

# Model Hyperparameters
#tf.flags.DEFINE_string("word2vec", "./data/rt-polaritydata/google.bin", "Word2vec file with pre-trained embeddings (default: None)")
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding ")
tf.flags.DEFINE_string("filter_sizes", "3", "Comma-separated filter sizes ")
tf.flags.DEFINE_integer("num_filters", 100, "Number of filters per filter size")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability ")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda")
tf.flags.DEFINE_float("l2_reg_V", 0, "L2 regularizaion V")
# Training parameters
tf.flags.DEFINE_integer("batch_size",100, "Batch Size ")
tf.flags.DEFINE_integer("num_epochs", 40, "Number of training epochs ")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps ")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps ")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


def train_step(u_batch, i_batch, uid, iid, y_batch, batch_num):
    """
    A single training step
    """
    feed_dict = {
        deep.input_u: u_batch,
        deep.input_i: i_batch,
        deep.input_y: y_batch,
        deep.input_uid: uid,
        deep.input_iid: iid,
        deep.dropout_keep_prob: FLAGS.dropout_keep_prob
    }
    _, step, loss, accuracy, mae, mse = sess.run(
        [train_op, global_step, deep.loss, deep.accuracy, deep.mae, deep.mse],
        feed_dict)
    time_str = datetime.datetime.now().isoformat()

    # print("{}: step {}, loss {:g}, rmse {:g},mae {:g}".format(time_str, batch_num, loss, accuracy, mae))
    return accuracy, mae, mse


def dev_step(u_batch, i_batch, uid, iid, y_batch, writer=None):
    """
    Evaluates model on a dev set

    """
    feed_dict = {
        deep.input_u: u_batch,
        deep.input_i: i_batch,
        deep.input_y: y_batch,
        deep.input_uid: uid,
        deep.input_iid: iid,
        deep.dropout_keep_prob: 1.0
    }
    step, loss, accuracy, mae, mse = sess.run(
        [global_step, deep.loss, deep.accuracy, deep.mae, deep.mse],
        feed_dict)
    time_str = datetime.datetime.now().isoformat()
    # print("{}: step{}, loss {:g}, rmse {:g},mae {:g}".format(time_str, step, loss, accuracy, mae))

    return [loss, accuracy, mae, mse]

def test_step(u_batch, i_batch, uid, iid, y_batch, writer=None):
    """
    Tests model on a test set

    """
    feed_dict = {
        deep.input_u: u_batch,
        deep.input_i: i_batch,
        deep.input_y: y_batch,
        deep.input_uid: uid,
        deep.input_iid: iid,
        deep.dropout_keep_prob: 1.0
    }
    preds, input_y, step, loss, accuracy, mae, mse = sess.run(
        [deep.predictions, deep.input_y, global_step, deep.loss, deep.accuracy, deep.mae, deep.mse],
        feed_dict)
    time_str = datetime.datetime.now().isoformat()
    # print("{}: step{}, loss {:g}, rmse {:g},mae {:g}".format(time_str, step, loss, accuracy, mae))

    return [preds, input_y, loss, accuracy, mae, mse]


general_dir = os.path.join(TPS_DIR, config['data_processed_file'])
sparse_user_dict_path = os.path.join(general_dir, 'user_dict.sparse')
sparse_user_dict_file = open(sparse_user_dict_path, 'rb')
sparse_user_dict = pickle.load(sparse_user_dict_file)

sparse_item_dict_path = os.path.join(general_dir, 'item_dict.sparse')
sparse_item_dict_file = open(sparse_item_dict_path, 'rb')
sparse_item_dict = pickle.load(sparse_item_dict_file)


#
#
# def set_sparse_end(userid_test):
#     for i in range(len(userid_test)):
#         if int(userid_test[i][0]) in sparse_user_dict[]


def get_sparse_test_data(sparse_dict, t_data, index):
    # print ('YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY')
    # print (type(test_data), test_data, test_data[0][0][0], test_data[0][1][0], test_data[0][2][0])

    t_data_out_1 =np.empty((0,3,1), float)
    t_data_out_2 =np.empty((0,3,1), float)
    t_data_out_3 =np.empty((0,3,1), float)
    t_data_out_4 =np.empty((0,3,1), float)
    t_data_out_5 =np.empty((0,3,1), float)

    for i in t_data:
        x =int(i[index][0])
        if x in sparse_dict[1]:
            t_data_out_1 = np.append(t_data_out_1, [i], axis=0)
        elif x in sparse_dict[2]:
            t_data_out_2 = np.append(t_data_out_2, [i], axis=0)
        elif x in sparse_dict[3]:
            t_data_out_3 = np.append(t_data_out_3, [i], axis=0)
        elif x in sparse_dict[4]:
            t_data_out_4 = np.append(t_data_out_4, [i], axis=0)
        elif x in sparse_dict[5]:
            t_data_out_5 = np.append(t_data_out_5, [i], axis=0)
        print ('sss', type(i), i)
    print ('injaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
    print (type(t_data_out_1), type(t_data_out_1[0]), t_data_out_1[0])
    return t_data_out_1, t_data_out_2, t_data_out_3, t_data_out_4, t_data_out_5


def test_proc(t_data, batch_size, log):
    test_length = len(t_data)
    loss_s = 0
    accuracy_s = 0
    mae_s = 0
    mse_s = 0
    ll_test = int(len(t_data) / batch_size) + 1
    for batch_num in range(ll_test):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size_test)
        data_test = t_data[start_index:end_index]

        # print ('************************************')
        # print (len(data_test))
        if len(data_test) == 0:
            continue
        userid_test, itemid_test, y_test = zip(*data_test)
        # set_sparse_end(userid_test)
        u_test = []
        i_test = []
        for i in range(len(userid_test)):
            # user_query = {"user_id": userid_test[i][0]}
            # user_doc = encoded_user_text_coll.find_one(user_query)
            # u_test.append(user_doc['review'])

            # item_query = {"item_id": itemid_test[i][0]}
            # item_doc = encoded_item_text_coll.find_one(item_query)
            # i_test.append(item_doc['review'])
            u_test.append(u_text[userid_test[i][0]])
            i_test.append(i_text[itemid_test[i][0]])
        u_test = np.array(u_test)
        i_test = np.array(i_test)

        # print ('&&&&&&&&&&&&&&&&&&&')
        # print u_test

        # print ('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        # print (type(userid_test), len(userid_test))
        # print ('*****************??????????????????')
        # for x in userid_test:
        #     print (type(x), type(x[0]), len(x), x, int(x[0]))
        # print (type(itemid_test), len(itemid_test))

        pred, input_y, loss, accuracy, mae, mse = test_step(u_test, i_test, userid_test, itemid_test, y_test)
        # print ('pred', type(pred), pred.shape)
        # print (pred)
        # print ('input_y', type(input_y), input_y.shape)
        # print (input_y)

        loss_s = loss_s + len(u_test) * loss
        accuracy_s = accuracy_s + len(u_test) * np.square(accuracy)
        mae_s = mae_s + len(u_test) * mae
        mse_s = mse_s + len(u_test) * mse
    print ('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@', log, '\n')
    print (
        "loss_test {:g}, rmse_test {:g}, mae_test {:g}, mse_test {:g}, accuracy_test {:g}".format(loss_s / test_length,
                                                                                                  np.sqrt(
                                                                                                      accuracy_s / test_length),
                                                                                                  mae_s / test_length,
                                                                                                  mse_s / test_length, (
                                                                                                              accuracy_s / test_length)))
    # rmse = np.sqrt(accuracy_s / test_length)
    # mae = mae_s / test_length
    # mse = mse_s / test_length

def get_greater_count(pred):
    sum = 0
    for i in range(len(pred)):
        if pred[i][0] > 5:
            sum += 1
    return sum


if __name__ == '__main__':
    # flags = tf.app.flags
    FLAGS = tf.app.flags.FLAGS

    # FLAGS = tf.flags.FLAGS
    # FLAGS._parse_flags()
    FLAGS(sys.argv)

    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")

    print("Loading data...")

    user_pkl_file = open(FLAGS.user_para_data, 'rb')
    item_pkl_file = open(FLAGS.item_para_data, 'rb')

    user_para = pickle.load(user_pkl_file)
    item_para = pickle.load(item_pkl_file)

    user_num = user_para['user_num']
    # item_num = user_para['item_num']
    item_num = item_para['item_num']
    user_length = user_para['user_length']
    # item_length = user_para['item_length']
    item_length = item_para['item_length']
    vocabulary_user = user_para['user_vocab']
    # vocabulary_item = user_para['item_vocab']
    vocabulary_item = item_para['item_vocab']
    train_length = user_para['train_length']
    valid_length = user_para['valid_length']
    test_length = user_para['test_length']
    u_text = user_para['u_text']
    # i_text = user_para['i_text']
    i_text = item_para['i_text']

    np.random.seed(2017)
    random_seed = 2017

    with tf.Graph().as_default():

        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            deep = DeepCoNN.DeepCoNN(
                user_num=user_num,
                item_num=item_num,
                user_length=user_length,
                item_length=item_length,
                num_classes=1,
                user_vocab_size=len(vocabulary_user),
                item_vocab_size=len(vocabulary_item),
                embedding_size=FLAGS.embedding_dim,
                fm_k=8,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                l2_reg_V=FLAGS.l2_reg_V,
                n_latent=32)
            tf.set_random_seed(random_seed)
            global_step = tf.Variable(0, name="global_step", trainable=False)

            # optimizer = tf.train.AdagradOptimizer(learning_rate=0.1, initial_accumulator_value=1e-8).minimize(deep.loss)

            optimizer = tf.train.AdamOptimizer(0.002, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(deep.loss)
            '''optimizer=tf.train.RMSPropOptimizer(0.002)f
            grads_and_vars = optimizer.compute_gradients(deep.loss)'''
            train_op = optimizer  # .apply_gradients(grads_and_vars, global_step=global_step)

            sess.run(tf.initialize_all_variables())

            if FLAGS.word2vec:
                # initial matrix with random uniform
                general_dir = os.path.join(TPS_DIR, config['data_processed_file'])

                u = 0
                initW = np.random.uniform(-1.0, 1.0, (len(vocabulary_user), FLAGS.embedding_dim))
                # load any vectors from the word2vec
                print("Load word2vec u file {}\n".format(FLAGS.word2vec))
                with open(FLAGS.word2vec, "rb") as f:
                    header = f.readline()
                    vocab_size, layer1_size = map(int, header.split())
                    binary_len = np.dtype('float32').itemsize * layer1_size
                    for line in xrange(vocab_size):
                        word = []
                        while True:
                            ch = f.read(1)
                            if ch == ' ':
                                word = ''.join(word)
                                break
                            if ch != '\n':
                                word.append(ch)
                        idx = 0

                        if word in vocabulary_user:
                            u = u + 1
                            idx = vocabulary_user[word]
                            initW[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                        else:
                            f.read(binary_len)

                output = open(os.path.join(general_dir, 'user_vocab.wv'), 'wb')
                pickle.dump(initW, output)

                sess.run(deep.W1.assign(initW))
                initW = np.random.uniform(-1.0, 1.0, (len(vocabulary_item), FLAGS.embedding_dim))
                # load any vectors from the word2vec
                print("Load word2vec i file {}\n".format(FLAGS.word2vec))

                item = 0
                with open(FLAGS.word2vec, "rb") as f:
                    header = f.readline()
                    vocab_size, layer1_size = map(int, header.split())
                    binary_len = np.dtype('float32').itemsize * layer1_size
                    for line in xrange(vocab_size):
                        word = []
                        while True:
                            ch = f.read(1)
                            if ch == ' ':
                                word = ''.join(word)
                                break
                            if ch != '\n':
                                word.append(ch)
                        idx = 0
                        if word in vocabulary_item:
                            item = item + 1
                            idx = vocabulary_item[word]
                            initW[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                        else:
                            f.read(binary_len)

                output = open(os.path.join(general_dir, 'item_vocab.wv'), 'wb')
                pickle.dump(initW, output)

                sess.run(deep.W2.assign(initW))

            l = (train_length / FLAGS.batch_size) + 1
            print l
            ll = 0
            epoch = 1
            best_mae = 5
            best_rmse = 5
            train_mae = 0
            train_rmse = 0
            train_mse = 0
            print ('FLAGS.train_data')
            print (FLAGS.train_data)
            pkl_file = open(FLAGS.train_data, 'rb')

            train_data = pickle.load(pkl_file)

            train_data = np.array(train_data)
            print ('train_data.shape')
            print (train_data.shape)
            pkl_file.close()

            # TODO
            # flags.DEFINE_string('test_data', '../data/patio/music.test', '../data/patio/music.test')
            # FLAGS.DEFINE_string('test_data', 'data', '../data/music/music.test')
            # FLAGS.test_data = '../data/music/music.test'
            print('FLAGS.test_data: ', FLAGS.test_data)
            pkl_file = open(FLAGS.test_data, 'rb')

            test_data = pickle.load(pkl_file)
            test_data = np.array(test_data)
            print ('len(t_data)', len(test_data), test_data.shape)
            print ('test_data.shape')
            print (test_data.shape)
            pkl_file.close()

            print('FLAGS.valid_data: ', FLAGS.valid_data)
            pkl_file = open(FLAGS.valid_data, 'rb')

            print ('*******************FLAGS.train_data, FLAGS.valid_data, FLAGS.test_data*******************')
            print (FLAGS.train_data, FLAGS.valid_data, FLAGS.test_data)

            valid_data = pickle.load(pkl_file)
            valid_data = np.array(valid_data)
            pkl_file.close()

            data_size_train = len(train_data)
            data_size_valid = len(valid_data)
            data_size_test = len(test_data)
            print ('data_size_train, data_size_valid, data_size_test')
            print (data_size_train, data_size_valid, data_size_test)
            batch_size = 100
            ll = int(len(train_data) / batch_size)

            # encoded_user_text_coll = mydb["encoded_user_text"]
            # encoded_item_text_coll = mydb["encoded_item_text"]

            start_train = time.time()
            for epoch in range(40):
                # Shuffle the data at each epoch
                shuffle_indices = np.random.permutation(np.arange(data_size_train))
                shuffled_data = train_data[shuffle_indices]
                for batch_num in range(ll):
                    start_index = batch_num * batch_size
                    end_index = min((batch_num + 1) * batch_size, data_size_train)
                    data_train = shuffled_data[start_index:end_index]

                    uid, iid, y_batch = zip(*data_train)

                    u_batch = []
                    i_batch = []
                    for i in range(len(uid)):
                        # print ('&&&&&&&&&&&&')
                        # print (type(uid))
                        # user_query = {"user_id": uid[i][0]}
                        # user_doc = encoded_user_text_coll.find_one(user_query)
                        # u_batch.append(user_doc['review'])

                        # item_query = {"item_id": iid[i][0]}
                        # item_doc = encoded_item_text_coll.find_one(item_query)
                        # i_batch.append(item_doc['review'])
                        u_batch.append(u_text[uid[i][0]])
                        i_batch.append(i_text[iid[i][0]])

                    u_batch = np.array(u_batch)
                    i_batch = np.array(i_batch)

                    t_rmse, t_mae, t_mse = train_step(u_batch, i_batch, uid, iid, y_batch, batch_num)
                    current_step = tf.train.global_step(sess, global_step)
                    train_rmse += t_rmse
                    train_mae += t_mae
                    train_mse += t_mse

                    if batch_num % 1000 == 0 and batch_num > 1:
                        print("\nEvaluation:")
                        print batch_num
                        loss_s = 0
                        accuracy_s = 0
                        mae_s = 0
                        mse_s = 0

                        ll_valid = int(len(valid_data) / batch_size) + 1
                        for batch_num2 in range(ll_valid):
                            start_index = batch_num2 * batch_size
                            end_index = min((batch_num2 + 1) * batch_size, data_size_valid)
                            data_valid = valid_data[start_index:end_index]

                            userid_valid, itemid_valid, y_valid = zip(*data_valid)

                            u_valid = []
                            i_valid = []
                            for i in range(len(userid_valid)):
                                # user_query = {"user_id": userid_valid[i][0]}
                                # user_doc = encoded_user_text_coll.find_one(user_query)
                                # u_valid.append(user_doc['review'])

                                # item_query = {"item_id": itemid_valid[i][0]}
                                # item_doc = encoded_item_text_coll.find_one(item_query)
                                # i_valid.append(item_doc['review'])
                                u_valid.append(u_text[userid_valid[i][0]])
                                i_valid.append(i_text[itemid_valid[i][0]])
                            u_valid = np.array(u_valid)
                            i_valid = np.array(i_valid)

                            loss, accuracy, mae, mse = dev_step(u_valid, i_valid, userid_valid, itemid_valid, y_valid)
                            loss_s = loss_s + len(u_valid) * loss
                            accuracy_s = accuracy_s + len(u_valid) * np.square(accuracy)
                            mae_s = mae_s + len(u_valid) * mae
                            mse_s = mse_s + len(u_valid) * mse
                        print ("loss_valid {:g}, rmse_valid {:g}, mae_valid {:g}, mse_valid {:g}".format(loss_s / valid_length,
                                                                                         np.sqrt(
                                                                                             accuracy_s / valid_length),
                                                                                         mae_s / valid_length, mse_s / valid_length))

                print str(epoch) + ':\n'
                print("\nEvaluation:")
                print "train:rmse,mae,mse:", train_rmse / ll, train_mae / ll, train_mse / ll
                train_rmse = 0
                train_mae = 0

                loss_s = 0
                accuracy_s = 0
                mae_s = 0
                mse_s = 0

                ll_valid = int(len(valid_data) / batch_size) + 1
                for batch_num in range(ll_valid):
                    start_index = batch_num * batch_size
                    end_index = min((batch_num + 1) * batch_size, data_size_valid)
                    data_valid = valid_data[start_index:end_index]

                    userid_valid, itemid_valid, y_valid = zip(*data_valid)
                    u_valid = []
                    i_valid = []
                    for i in range(len(userid_valid)):
                        # user_query = {"user_id": userid_valid[i][0]}
                        # user_doc = encoded_user_text_coll.find_one(user_query)
                        # u_valid.append(user_doc['review'])

                        # item_query = {"item_id": itemid_valid[i][0]}
                        # item_doc = encoded_item_text_coll.find_one(item_query)
                        # i_valid.append(item_doc['review'])
                        u_valid.append(u_text[userid_valid[i][0]])
                        i_valid.append(i_text[itemid_valid[i][0]])
                    u_valid = np.array(u_valid)
                    i_valid = np.array(i_valid)

                    loss, accuracy, mae, mse = dev_step(u_valid, i_valid, userid_valid, itemid_valid, y_valid)
                    loss_s = loss_s + len(u_valid) * loss
                    accuracy_s = accuracy_s + len(u_valid) * np.square(accuracy)
                    mae_s = mae_s + len(u_valid) * mae
                    mse_s = mse_s + len(u_valid) * mse
                print ("loss_valid {:g}, rmse_valid {:g}, mae_valid {:g}, mse_valid {:g}".format(loss_s / valid_length,
                                                                                 np.sqrt(accuracy_s / valid_length),
                                                                                 mae_s / valid_length, mse_s / valid_length))
                rmse = np.sqrt(accuracy_s / valid_length)
                mae = mae_s / valid_length
                if best_rmse > rmse:
                    best_rmse = rmse
                if best_mae > mae:
                    best_mae = mae
                print("")
            print 'best rmse:', best_rmse
            print 'best mae:', best_mae

            end_train = time.time()


            # Testing >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>...

            loss_s = 0
            accuracy_s = 0
            mae_s = 0
            mse_s = 0

            print ('YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY')
            print (type(test_data),type(test_data[0]), test_data, test_data[0][0][0], test_data[0][1][0], test_data[0][2][0])

            u_t_data_1, u_t_data_2, u_t_data_3, u_t_data_4, u_t_data_5 = get_sparse_test_data(sparse_user_dict, test_data, 0)
            i_t_data_1, i_t_data_2, i_t_data_3, i_t_data_4, i_t_data_5 = get_sparse_test_data(sparse_item_dict, test_data, 1)

            print ('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
            print (type(test_data), type(test_data[0]))
            test_proc(u_t_data_1, batch_size, "sparse USER with 1 review")
            test_proc(u_t_data_2, batch_size, "sparse USER with 2 reviews")
            test_proc(u_t_data_3, batch_size, "sparse USER with 3 reviews")
            test_proc(u_t_data_4, batch_size, "sparse USER with 4 reviews")
            test_proc(u_t_data_5, batch_size, "sparse USER with 5 reviews")

            test_proc(i_t_data_1, batch_size, "sparse ITEM with 1 review")
            test_proc(i_t_data_2, batch_size, "sparse ITEM with 2 reviews")
            test_proc(i_t_data_3, batch_size, "sparse ITEM with 3 reviews")
            test_proc(i_t_data_4, batch_size, "sparse ITEM with 4 reviews")
            test_proc(i_t_data_5, batch_size, "sparse ITEM with 5 reviews")

            test_start = time.time()
            greater_than_five_count = 0
            ll_test = int(len(test_data) / batch_size) + 1
            for batch_num in range(ll_test):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size_test)
                data_test = test_data[start_index:end_index]

                userid_test, itemid_test, y_test = zip(*data_test)
                # set_sparse_end(userid_test)
                u_test = []
                i_test = []
                for i in range(len(userid_test)):
                    # user_query = {"user_id": userid_test[i][0]}
                    # user_doc = encoded_user_text_coll.find_one(user_query)
                    # u_test.append(user_doc['review'])

                    # item_query = {"item_id": itemid_test[i][0]}
                    # item_doc = encoded_item_text_coll.find_one(item_query)
                    # i_test.append(item_doc['review'])
                    u_test.append(u_text[userid_test[i][0]])
                    i_test.append(i_text[itemid_test[i][0]])
                u_test = np.array(u_test)
                i_test = np.array(i_test)

                # print ('&&&&&&&&&&&&&&&&&&&')
                # print u_test

                # print ('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                # print (type(userid_test), len(userid_test))
                # print ('*****************??????????????????')
                # for x in userid_test:
                #     print (type(x), type(x[0]), len(x), x, int(x[0]))
                # print (type(itemid_test), len(itemid_test))

                pred, input_y, loss, accuracy, mae, mse = test_step(u_test, i_test, userid_test, itemid_test, y_test)
                print ('pred', type(pred), pred.shape)
                print (pred)
                greater_than_five_count += get_greater_count(pred)
                # print ('input_y', type(input_y), input_y.shape)
                # print (input_y)

                loss_s = loss_s + len(u_test) * loss
                accuracy_s = accuracy_s + len(u_test) * np.square(accuracy)
                mae_s = mae_s + len(u_test) * mae
                mse_s = mse_s + len(u_test) * mse
            test_end = time.time()
            print ("############################# TEST TIME FOR ONE SAMPLE ###################################")
            print ((test_end-test_start)/test_length)
            print ("############################# NUMBER OF SAMPLES GREATER THAN 5 ###################################")
            print (greater_than_five_count)
            print ("############################# Final Evaluation ###################################")
            print ("loss_test {:g}, rmse_test {:g}, mae_test {:g}, mse_test {:g}, accuracy_test {:g}".format(loss_s / test_length,
                                                                             np.sqrt(accuracy_s / test_length),
                                                                             mae_s / test_length, mse_s / test_length, (accuracy_s / test_length)))
            rmse = np.sqrt(accuracy_s / test_length)
            mae = mae_s / test_length
            mse = mse_s / test_length
            # if best_rmse > rmse:
            #     best_rmse = rmse
            # if best_mae > mae:
            #     best_mae = mae
            # print("")
        print 'test_rmse:', rmse
        print 'test_mae:', mae
        print 'test_mse:', mse

    print 'train took {} minutes.'.format(((end_train - start_train) / 60))

    print 'end'
