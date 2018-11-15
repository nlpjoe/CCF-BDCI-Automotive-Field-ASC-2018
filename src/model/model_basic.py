import tensorflow as tf
from keras.utils import np_utils
from tqdm import tqdm
import time
import datetime
import keras as keras
from keras import backend as K
from sklearn.model_selection import KFold
from keras.models import load_model
from keras.models import save_model
from sklearn.model_selection import StratifiedKFold
# from keras.utils.vis_utils import plot_model
#  import lightgbm as lgbm
from keras import optimizers
import numpy as np
import pandas as pd
import os
import pickle

from model.snapshot import SnapshotCallbackBuilder
from model.my_callbacks import JZTrainCategory

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score


class BasicModel(object):

    """Docstring for BasicModel. """

    def __init__(self):
        """TODO: to be defined1. """
        pass

    def create_model(self, kfold_X_train, y_train, kfold_X_test, y_test, test):
        pass

    # Generate batches
    def batch_iter(self, data, batch_size, num_epochs=1, shuffle=True):
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int((data_size-1)/batch_size) + 1
        for epoch in range(num_epochs):
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((1 + batch_num) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]

    def get_f1_score(self, x, y, verbose=False):
        tp = np.sum(np.logical_and(y > 0, x == y))
        fp = np.sum(np.logical_and(x > 0, y == 0)) + np.sum(np.logical_and(x * y > 0, y != x))  # 多判或者错判
        fn = np.sum(np.logical_and(y > 0, x == 0))  # 漏判

        P = float(tp) / (float(tp + fp) + 1e-8)
        R = float(tp) / (float(tp + fn) + 1e-8)
        F = 2 * P * R / (P + R + 1e-8)

        if verbose:
            print('P->', P)
            print('R->', R)
            print('F->', F)
        return F


class BasicDeepModel(BasicModel):

    """Docstring for BasicModel. """

    def __init__(self, n_folds=5, name='BasicModel', config=None):
        if config is None:
            exit('请传入数值')
        self.name = name
        self.config = config
        self.is_debug = config.is_debug
        self.n_classes = config.n_classes
        self.main_feature = config.main_feature
        self.n_epochs = config.n_epochs
        self.sentiment_embed = pickle.load(open(self.config.SENTIMENT_EMBED_PATH, 'rb'))

        if self.main_feature == 'char':
            self.embedding = config.char_embedding
            self.max_len = config.CHAR_MAXLEN
            self.max_features = len(config.char_embedding)
            self.embed_size = len(self.embedding[0])
            self.mask_value = self.max_features - 1
        elif self.main_feature == 'word':
            self.embedding = config.word_embedding
            self.max_len = config.WORD_MAXLEN
            self.max_features = len(config.word_embedding)
            self.embed_size = len(self.embedding[0])
            self.mask_value = self.max_features - 1
        elif self.main_feature == 'elmo_word':
            self.max_len = config.WORD_MAXLEN
            self.embed_size = 1024
        elif self.main_feature == 'elmo_char':
            self.max_len = config.CHAR_MAXLEN
            self.embed_size = 1024
        elif self.main_feature == 'elmo_qiuqiu':
            self.max_len = config.WORD_MAXLEN
            self.embed_size = 300

        else:
            exit('选择word或者char作为特征')

        self.batch_size = config.BATCH_SIZE

        self.n_folds = n_folds

        if self.config.data_type == 2 or self.config.data_type == 3:  # 多标签分类
            self.kf = KFold(n_splits=n_folds, shuffle=True, random_state=10)
        else:  # 单标签分类
            self.kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=10)

        print("[INFO] training with {} GPU...".format(config.gpu))

    def multi_train_predict(self, train, train_y, test, option=2):
        """
        we use KFold way to train our model and save the model
        :param train:
        :return:
        """
        gpu_options = tf.GPUOptions(visible_device_list=self.config.gpu, allow_growth=True)
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=True,
                gpu_options=gpu_options)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                initializer = tf.contrib.layers.xavier_initializer()
                with tf.variable_scope('', initializer=initializer):
                    model = self.create_model()

                # 定义训练流程
                train_op = tf.train.AdamOptimizer(1e-3).minimize(model.loss)

                def train_step(x_batch, y_batch, global_step):
                    feed_dict = {
                        model.input_x: x_batch,
                        model.input_y: y_batch,
                        model.dropout_keep_prob: 1.0
                    }
                    _, summaries, loss, accuracy, predictions = sess.run(
                        [train_op, train_summary_op, model.loss, model.accuracy, model.predictions],
                        feed_dict)
                    f1 = self.get_f1_score(predictions, np.argmax(y_batch,1))
                    global_step += 1
                    train_summary_writer.add_summary(summaries, global_step)
                    return loss, f1, global_step

                def dev_step(x_batch, y_batch, global_step, writer=None):
                    feed_dict = {
                        model.input_x: x_batch,
                        model.input_y: y_batch,
                        model.dropout_keep_prob: 1.0
                    }
                    summaries, loss, accuracy, predictions = sess.run(
                        [dev_summary_op, model.loss, model.accuracy, model.predictions],
                        feed_dict)
                    if writer is not None:
                        writer.add_summary(summaries, global_step)
                    return loss, predictions

                def test_step(batches):
                    all_prob = []
                    for x_batch in batches:
                        feed_dict = {
                            model.input_x: x_batch,
                            model.dropout_keep_prob: 1.0,
                        }
                        all_prob.extend(sess.run([model.prob], feed_dict))
                    return np.concatenate(all_prob)

                total_dev_probs = np.zeros((train['word'].shape[0], 10, self.n_classes))
                total_test_probs = np.zeros((test['word'].shape[0], 10, self.n_classes))
                # 把model和summary输出到文件夹
                timestamp = str(int(time.time()))
                out_dir_parent = os.path.abspath(os.path.join(os.path.curdir, 'runs'))
                out_dir = os.path.join(out_dir_parent, timestamp)
                print('Writing to ', out_dir)
                final_f1 = []
                final_loss = []
                for ith_fold, (train_index, dev_index) in enumerate(self.kf.split(train['word'])):
                    single_losses = []
                    single_f1_score = []
                    for j in tqdm(range(10)):
                        label = train_y[:, j]
                        label = np_utils.to_categorical(label, num_classes=self.n_classes)
                        # Initialize all varibles
                        global_step = 0
                        sess.run(tf.global_variables_initializer())

                        # loss和accuracy的summary
                        loss_summary = tf.summary.scalar('loss', model.loss)
                        acc_summary = tf.summary.scalar('accuracy', model.accuracy)

                        # Train summary
                        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
                        train_summary_dir = os.path.join(out_dir, self.name, 'train-{}'.format(ith_fold))
                        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

                        # Dev summary
                        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
                        dev_summary_dir = os.path.join(out_dir, self.name, 'dev-{}'.format(ith_fold))
                        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

                        # Checkpoint 文件夹
                        checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints-{}'.format(ith_fold)))
                        checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        saver = tf.train.Saver(max_to_keep=None)

                        kfold_X_train = {}
                        kfold_X_dev = {}
                        kfold_y_train, kfold_y_dev = label[train_index], label[dev_index]

                        #  for c in ['word', 'char', 'word_left', 'word_right', 'char_left', 'char_right', 'hann_word', 'hann_char']:
                            #  kfold_X_train[c] = train[c][train_index]
                            #  kfold_X_dev[c] = train[c][dev_index]

                        if 'han' not in self.name and 'rcnn' not in self.name:
                            kfold_X_train = train[self.main_feature.lower()][train_index]
                            kfold_X_dev = train[self.main_feature.lower()][dev_index]
                            test_data = test[self.main_feature.lower()]
                        else:
                            exit('测试textcnn先')

                        max_f1 = 0.0
                        min_loss = 10000.
                        early_stop = 0
                        for epoch in range(self.n_epochs):
                            if early_stop == 5:
                                break
                            print('epoch: %d' % epoch)
                            batches = self.batch_iter(list(zip(kfold_X_train, kfold_y_train)), self.batch_size)
                            for batch in batches:
                                if early_stop == 5:
                                    break
                                x_batch, y_batch = zip(*batch)  # zip(*) == unzip
                                loss, f1, global_step = train_step(x_batch, y_batch, global_step)
                                if global_step % 10 == 0:
                                    time_str = datetime.datetime.now().isoformat()
                                    print('{}: step {}, loss {:g}, f1 {:g}'.format(time_str, global_step, loss, f1))

                                if global_step % 50 == 0:
                                    dev_losses = []
                                    dev_pred = []
                                    dev_batches = self.batch_iter(list(zip(kfold_X_dev, kfold_y_dev)), self.batch_size, shuffle=False)
                                    for dev_batch in dev_batches:
                                        x_batch, y_batch = zip(*dev_batch)  # zip(*) == unzip
                                        loss, predictions = dev_step(x_batch, y_batch, global_step, writer=dev_summary_writer)
                                        dev_losses.append(loss * len(x_batch))
                                        dev_pred.append(predictions)

                                    dev_pred = np.concatenate(dev_pred)
                                    f1 = self.get_f1_score(dev_pred, np.argmax(kfold_y_dev, 1))

                                    loss = np.sum(dev_losses) / len(kfold_X_dev)

                                    if loss < min_loss:
                                        early_stop = 0
                                        min_loss = loss
                                        max_f1 = f1
                                        time_str = datetime.datetime.now().isoformat()
                                        print("\nEvaluation:")
                                        print('{}: step {}, loss {:g}, f1 {:g}'.format(time_str, global_step, loss, max_f1))
                                        print('saving model')
                                        path = saver.save(sess, checkpoint_prefix, global_step=0)
                                        print('have saved model to ', path, '\n')
                                    else:
                                        early_stop += 1

                        print('saving f1 {}, loss {}'.format(max_f1, min_loss))
                        single_losses.append(min_loss)
                        single_f1_score.append(max_f1)
                        path = checkpoint_prefix + '-0'
                        print('load model:', path)
                        try:
                            saver.restore(sess, path)
                        except:
                            exit()
                        test_batches = self.batch_iter(test_data, self.batch_size, shuffle=False)
                        dev_batches = self.batch_iter(kfold_X_dev, self.batch_size, shuffle=False)
                        total_dev_probs[dev_index, j] = test_step(dev_batches)
                        total_test_probs[:, j] += test_step(test_batches) / self.n_folds

                    subject = pickle.load(open('../data/sub_list.pkl', 'rb'))
                    for idx, sub in enumerate(subject):
                        print('{}->{}'.format(sub, single_f1_score[idx]))
                    final_f1.append(np.mean(single_f1_score))
                    final_loss.append(np.mean(single_losses))
                    if self.config.is_debug == True:
                        break

                mean_acc = np.mean(final_f1)
                mean_loss = np.mean(final_loss)
                print('final f1:\t{} -> {}'.format(final_f1, mean_acc))
                print('final loss:\t{} -> {}'.format(final_loss, mean_loss))
                os.system('mv {} {}'.format(out_dir, os.path.join(out_dir_parent, str(round(mean_loss, 5))+'_'+str(round(mean_acc, 5)))))
                os.makedirs('../data/result-dt{}-op{}-embed{}-debug{}'.format(self.config.data_type, self.config.option, self.config.EMBED_SIZE, self.config.is_debug), exist_ok=True)
                with open('../data/result-dt{}-op{}-embed{}-debug{}/{}_oof_l{:.5f}_f1{:.5f}_oe{}.pkl'.format(self.config.data_type, self.config.option, self.config.EMBED_SIZE,
                                                                                                self.config.is_debug, self.name,
                                                                                                np.mean(final_loss),
                                                                                                np.mean(final_f1), self.config.outer_embed), 'wb') as f:
                    pickle.dump(total_dev_probs, f)

                with open('../data/result-dt{}-op{}-embed{}-debug{}/{}_pre_l{:.5f}_f1{:.5f}_oe{}.pkl'.format(self.config.data_type, self.config.option, self.config.EMBED_SIZE,
                                                                                                self.config.is_debug, self.name,
                                                                                                np.mean(final_loss),
                                                                                                np.mean(final_f1), self.config.outer_embed), 'wb') as f:
                    pickle.dump(total_test_probs, f)

                print('done')

    def single_train_predict(self, train, train_y, test, option=2):
        """
        we use KFold way to train our model and save the model
        :param train:
        :return:
        """
        gpu_options = tf.GPUOptions(visible_device_list=self.config.gpu, allow_growth=True)
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=True,
                gpu_options=gpu_options)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                initializer = tf.contrib.layers.xavier_initializer()
                with tf.variable_scope('', initializer=initializer):
                    model = self.create_model()


                # 定义训练流程
                train_op = tf.train.AdamOptimizer().minimize(model.loss)

                def train_step(x_batch, y_batch, global_step):
                    feed_dict = {
                        model.input_x: x_batch,
                        model.input_y: y_batch,
                        model.dropout_keep_prob: 1.0,
                        model.output_keep_prob: 1.0
                    }
                    _, summaries, loss, accuracy = sess.run(
                        [train_op, train_summary_op, model.loss, model.accuracy],
                        feed_dict)
                    global_step += 1
                    train_summary_writer.add_summary(summaries, global_step)
                    return loss, accuracy, global_step

                def dev_step(x_batch, y_batch, global_step, writer=None):
                    feed_dict = {
                        model.input_x: x_batch,
                        model.input_y: y_batch,
                        model.dropout_keep_prob: 1.0,
                        model.output_keep_prob: 1.0
                    }
                    _, summaries, loss, accuracy = sess.run(
                        [train_op, dev_summary_op, model.loss, model.accuracy],
                        feed_dict)
                    if writer is not None:
                        writer.add_summary(summaries, global_step)
                    return loss, accuracy

                def test_step(batches):
                    all_prob = []
                    for x_batch in batches:
                        feed_dict = {
                            model.input_x: x_batch,
                            model.dropout_keep_prob: 1.0,
                            model.output_keep_prob: 1.0
                        }
                        all_prob.extend(sess.run([model.prob], feed_dict))
                    return np.concatenate(all_prob)

                total_dev_probs = np.zeros((train['word'].shape[0], self.n_classes))
                total_test_probs = np.zeros((test['word'].shape[0], self.n_classes))
                # 把model和summary输出到文件夹
                timestamp = str(int(time.time()))
                out_dir_parent = os.path.abspath(os.path.join(os.path.curdir, 'runs'))
                out_dir = os.path.join(out_dir_parent, timestamp)
                print('Writing to ', out_dir)
                final_acc = []
                final_loss = []
                for ith_fold, (train_index, dev_index) in enumerate(self.kf.split(train['word'], np.argmax(train_y, 1))):
                    # Initialize all varibles
                    global_step = 0
                    sess.run(tf.global_variables_initializer())

                    # loss和accuracy的summary
                    loss_summary = tf.summary.scalar('loss', model.loss)
                    acc_summary = tf.summary.scalar('accuracy', model.accuracy)

                    # Train summary
                    train_summary_op = tf.summary.merge([loss_summary, acc_summary])
                    train_summary_dir = os.path.join(out_dir, self.name, 'train-{}'.format(ith_fold))
                    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

                    # Dev summary
                    dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
                    dev_summary_dir = os.path.join(out_dir, self.name, 'dev-{}'.format(ith_fold))
                    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

                    # Checkpoint 文件夹
                    checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints-{}'.format(ith_fold)))
                    checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    saver = tf.train.Saver(max_to_keep=None)

                    kfold_X_train = {}
                    kfold_X_dev = {}
                    kfold_y_train, kfold_y_dev = train_y[train_index], train_y[dev_index]

                    #  for c in ['word', 'char', 'word_left', 'word_right', 'char_left', 'char_right', 'hann_word', 'hann_char']:
                        #  kfold_X_train[c] = train[c][train_index]
                        #  kfold_X_dev[c] = train[c][dev_index]

                    if 'han' not in self.name and 'rcnn' not in self.name:
                        kfold_X_train = train[self.main_feature.lower()][train_index]
                        kfold_X_dev = train[self.main_feature.lower()][dev_index]
                        test_data = test[self.main_feature.lower()]
                    else:
                        exit('测试textcnn先')

                    max_acc = 0.0
                    min_loss = 10000.
                    for epoch in range(self.n_epochs):
                        print('epoch: %d' % epoch)
                        batches = self.batch_iter(list(zip(kfold_X_train, kfold_y_train)), self.batch_size)
                        for batch in batches:

                            x_batch, y_batch = zip(*batch)  # zip(*) == unzip
                            loss, acc, global_step = train_step(x_batch, y_batch, global_step)
                            if global_step % 10 == 0:
                                time_str = datetime.datetime.now().isoformat()
                                print('{}: step {}, loss {:g}, acc {:g}'.format(time_str, global_step, loss, acc))
                            if global_step % 20 == 0:
                                print("\nEvaluation:")
                                dev_losses = []
                                dev_acc = []
                                dev_batches = self.batch_iter(list(zip(kfold_X_dev, kfold_y_dev)), self.batch_size, shuffle=False)
                                for dev_batch in dev_batches:
                                    x_batch, y_batch = zip(*dev_batch)  # zip(*) == unzip
                                    loss, acc = dev_step(x_batch, y_batch, global_step, writer=dev_summary_writer)
                                    dev_losses.append(loss * len(x_batch))
                                    dev_acc.append(acc * len(x_batch))

                                loss = np.sum(dev_losses) / len(kfold_X_dev)
                                acc = np.sum(dev_acc) / len(kfold_X_dev)

                                if loss < min_loss:
                                    min_loss = loss
                                    max_acc = acc
                                    time_str = datetime.datetime.now().isoformat()
                                    print('{}: step {}, loss {:g}, acc {:g}'.format(time_str, global_step, loss, acc))
                                    print('saving model')
                                    path = saver.save(sess, checkpoint_prefix, global_step=0)
                                    print('have saved model to ', path, '\n')

                    print('saving acc {}, loss {}'.format(max_acc, min_loss))
                    final_loss.append(min_loss)
                    final_acc.append(max_acc)
                    path = checkpoint_prefix + '-0'
                    print('load model:', path)
                    try:
                        saver.restore(sess, path)
                    except:
                        exit()
                    test_batches = self.batch_iter(test_data, self.batch_size, shuffle=False)
                    dev_batches = self.batch_iter(kfold_X_dev, self.batch_size, shuffle=False)
                    total_dev_probs[dev_index] = test_step(dev_batches)
                    total_test_probs += (test_step(test_batches) / self.n_folds)

                mean_acc = np.mean(final_acc)
                mean_loss = np.mean(final_loss)
                print('final accuracy:\t{} -> {}'.format(final_acc, mean_acc))
                print('final loss:\t{} -> {}'.format(final_loss, mean_loss))
                os.system('mv {} {}'.format(out_dir, os.path.join(out_dir_parent, str(round(mean_loss, 5))+'_'+str(round(mean_acc, 5)))))
                os.makedirs('../data/result-dt{}-op{}-embed{}-debug{}'.format(self.config.data_type, self.config.option, self.config.EMBED_SIZE, self.config.is_debug), exist_ok=True)
                with open('../data/result-dt{}-op{}-embed{}-debug{}/{}_oof_l{:.5f}_a{:.5f}_oe{}.pkl'.format(self.config.data_type, self.config.option, self.config.EMBED_SIZE,
                                                                                                self.config.is_debug, self.name,
                                                                                                np.mean(final_loss),
                                                                                                np.mean(final_acc), self.config.outer_embed), 'wb') as f:
                    pickle.dump(total_dev_probs, f)

                with open('../data/result-dt{}-op{}-embed{}-debug{}/{}_pre_l{:.5f}_a{:.5f}_oe{}.pkl'.format(self.config.data_type, self.config.option, self.config.EMBED_SIZE,
                                                                                                self.config.is_debug, self.name,
                                                                                                np.mean(final_loss),
                                                                                                np.mean(final_acc), self.config.outer_embed), 'wb') as f:
                    pickle.dump(total_test_probs, f)

                print('done')

    def four_classify_train_predict(self, train_dict, train_y, test, option=3):
        """
        we use KFold way to train our model and save the model
        :param train:
        :return:
        """
        train_y2 = np.reshape(np.argmax(pickle.load(open(self.config.Y_DISTILLATION, 'rb')), -1), [-1])
        train_y2 = np.reshape(np.eye(4)[train_y2], [-1, 10, 4])
        gpu_options = tf.GPUOptions(visible_device_list=self.config.gpu, allow_growth=True)
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=True,
                gpu_options=gpu_options)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                initializer = tf.contrib.layers.xavier_initializer()
                with tf.variable_scope('', initializer=initializer):
                    model = self.create_model()

                # 定义训练流程
                global_step = tf.Variable(0, name='global_step', trainable=False)
                # learning_rate = tf.train.exponential_decay(1e-3, global_step, num_batch, 0.98, True)
                #  learning_rate = tf.train.cosine_decay_restarts(1e-3, global_step, first_decay_steps=500, t_mul=1.0)
                learning_rate = tf.Variable(1e-3, name='lr', trainable=False)
                optimizer = tf.train.AdamOptimizer(learning_rate)                   # 定义优化器
                train_op = optimizer.minimize(model.loss, global_step=global_step)

                decay_ops = learning_rate.assign(learning_rate*0.5)

                def train_step(x_batch, x_token, x_mask, x_type, y_batch, y2_batch, global_step):
                    batch_len = len(x_batch)
                    if batch_len != self.batch_size:
                        n_copy = self.batch_size // batch_len + 1
                        x_copy = [x_batch for _ in range(n_copy)]
                        x_token_copy = [x_token for _ in range(n_copy)]
                        x_mask_copy = [x_mask for _ in range(n_copy)]
                        x_type_copy = [x_type for _ in range(n_copy)]
                        y_copy = [y_batch for _ in range(n_copy)]
                        y2_copy = [y2_batch for _ in range(n_copy)]
                        x_batch = np.concatenate(x_copy)[:self.batch_size]
                        x_token = np.concatenate(x_token_copy)[:self.batch_size]
                        x_type = np.concatenate(x_type_copy)[:self.batch_size]
                        x_mask = np.concatenate(x_mask_copy)[:self.batch_size]
                        y_batch = np.concatenate(y_copy)[:self.batch_size]
                        y2_batch = np.concatenate(y2_copy)[:self.batch_size]

                    feed_dict = {
                        model.input_x: x_batch,
                        model.input_ids: x_token,
                        model.type_ids: x_type,
                        model.mask_ids: x_mask,
                        model.input_y: y_batch,
                        model.input_y2: y2_batch,
                        model.dropout_keep_prob: 0.5,
                        #  model.dropout_keep_prob: 0.5,
                        model.output_keep_prob: 1.0,
                        model.is_training: True
                    }
                    _, loss, prediction, step = sess.run(
                        [train_op, model.loss, model.prediction, global_step],
                        feed_dict)
                    return loss, prediction[:batch_len], step

                def dev_step(x_batch, x_token, x_mask, x_type, y_batch, y2_batch, global_step, writer=None):
                    batch_len = len(x_batch)
                    if batch_len != self.batch_size:
                        n_copy = self.batch_size // batch_len + 1
                        x_copy = [x_batch for _ in range(n_copy)]
                        x_token_copy = [x_token for _ in range(n_copy)]
                        x_mask_copy = [x_mask for _ in range(n_copy)]
                        x_type_copy = [x_type for _ in range(n_copy)]
                        y_copy = [y_batch for _ in range(n_copy)]
                        y2_copy = [y2_batch for _ in range(n_copy)]
                        x_batch = np.concatenate(x_copy)[:self.batch_size]
                        x_token = np.concatenate(x_token_copy)[:self.batch_size]
                        x_type = np.concatenate(x_type_copy)[:self.batch_size]
                        x_mask = np.concatenate(x_mask_copy)[:self.batch_size]
                        y_batch = np.concatenate(y_copy)[:self.batch_size]
                        y2_batch = np.concatenate(y2_copy)[:self.batch_size]
                    feed_dict = {
                        model.input_x: x_batch,
                        model.input_ids: x_token,
                        model.type_ids: x_type,
                        model.mask_ids: x_mask,
                        model.input_y: y_batch,
                        model.input_y2: y2_batch,
                        model.dropout_keep_prob: 1.0,
                        #  model.dropout_keep_prob: 0.5,
                        model.output_keep_prob: 1.0,
                        model.is_training: False
                    }

                    loss, prediction = sess.run(
                        [model.loss, model.prediction],
                        feed_dict)
                    return loss, prediction[:batch_len]

                def test_step(batches):
                    all_prob = []
                    for x_batch in batches:
                        x_batch, x_token, x_mask, x_type = zip(*batch)  # zip(*) == unzip
                        batch_len = len(x_batch)
                        if batch_len != self.batch_size:
                            n_copy = self.batch_size // batch_len + 1
                            x_copy = [x_batch for _ in range(n_copy)]
                            x_token_copy = [x_token for _ in range(n_copy)]
                            x_mask_copy = [x_mask for _ in range(n_copy)]
                            x_type_copy = [x_type for _ in range(n_copy)]
                            x_batch = np.concatenate(x_copy)[:self.batch_size]
                            x_token = np.concatenate(x_token_copy)[:self.batch_size]
                            x_type = np.concatenate(x_type_copy)[:self.batch_size]
                            x_mask = np.concatenate(x_mask_copy)[:self.batch_size]

                        feed_dict = {
                            model.input_x: x_batch,
                            model.input_ids: x_token,
                            model.type_ids: x_type,
                            model.mask_ids: x_mask,
                            model.dropout_keep_prob: 1.0,
                            model.output_keep_prob: 1.0,
                            model.is_training: False
                        }

                        preds = sess.run([model.prob], feed_dict)[0][:batch_len]
                        all_prob.append(preds)
                    all_prob = np.concatenate(all_prob, axis=0)
                    return all_prob

                n_sent = len(train_dict[self.main_feature]) // 3

                # 训练集
                train = train_dict[self.main_feature][:n_sent]
                train_token_id = train_dict['token_id'][:n_sent]
                train_mask_id = train_dict['mask_id'][:n_sent]
                train_type_id = train_dict['type_id'][:n_sent]

                train_jp = train_dict[self.main_feature][n_sent:2*n_sent]
                train_en = train_dict[self.main_feature][2*n_sent:3*n_sent]

                # 测试集
                test_data = test[self.main_feature.lower()]
                test_token_id = test['token_id']
                test_mask_id = test['mask_id']
                test_type_id = test['type_id']

                # 结果
                total_dev_probs = np.zeros((len(train), 10, self.n_classes))
                total_test_probs = np.zeros((len(test_data), 10, self.n_classes))

                # 把model和summary输出到文件夹
                timestamp = str(int(time.time()))
                out_dir_parent = os.path.abspath(os.path.join(os.path.curdir, 'runs'))
                out_dir = os.path.join(out_dir_parent, timestamp)
                print('Writing to ', out_dir)
                final_f1 = []
                final_loss = []
                for ith_fold, (train_index, dev_index) in enumerate(self.kf.split(train)):
                    # Checkpoint 文件夹
                    checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints-{}'.format(ith_fold)))
                    checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    saver = tf.train.Saver(max_to_keep=None)

                    max_sub_f1 = 0.0
                    min_sub_loss = 10000.
                    for j in range(1):  # 3次取最好的结果
                        print('\n----第{}折{}----\n'.format(ith_fold, j))
                        # Initialize all varibles
                        sess.run(tf.global_variables_initializer())

                        #  kfold_X_train = np.concatenate((train[train_index], train_jp[train_index], train_en[train_index]))
                        #  kfold_y_train = np.concatenate((train_y[train_index], train_y[train_index], train_y[train_index]))
                        #  kfold_y_train_dill = np.concatenate((train_y2[train_index], train_y2[train_index], train_y2[train_index]))

                        kfold_X_train = train[train_index]
                        kfold_X_token_train = train_token_id[train_index]
                        kfold_X_mask_train = train_mask_id[train_index]
                        kfold_X_type_train = train_type_id[train_index]
                        kfold_y_train = train_y[train_index]
                        kfold_y_train_dill = train_y2[train_index]

                        kfold_X_dev = train[dev_index]
                        kfold_X_token_dev = train_token_id[dev_index]
                        kfold_X_mask_dev = train_mask_id[dev_index]
                        kfold_X_type_dev = train_type_id[dev_index]
                        kfold_y_dev = train_y[dev_index]
                        kfold_y_dev_dill = train_y2[dev_index]

                        max_f1 = -0.01
                        min_loss = 10000.
                        early_stop = 0
                        for epoch in range(self.n_epochs):
                            # if early_stop >= 20:
                                # break
                            print('epoch: %d' % epoch)
                            batches = self.batch_iter(list(zip(kfold_X_train, kfold_X_token_train, kfold_X_mask_train, kfold_X_type_train,\
                                                               kfold_y_train, kfold_y_train_dill)), self.batch_size, shuffle=True)
                            for batch in batches:
                                # if early_stop >= 20:
                                    # break
                                x_batch, x_token, x_mask, x_type, y_batch, y2_batch = zip(*batch)  # zip(*) == unzip
                                loss, train_pred, step = train_step(x_batch, x_token, x_mask, x_type, y_batch, y2_batch, global_step)
                                f1_score = self.get_f1_score(train_pred, np.argmax(y_batch, -1))
                                if step % 10 == 0:
                                    time_str = datetime.datetime.now().isoformat()
                                    print('{}: step {}, lr {:g}, loss {:g}, f1 {:g}'.format(time_str, step, sess.run(optimizer._lr), loss, f1_score))

                                if step % 50 == 0:
                                    dev_losses = []
                                    dev_preds = []
                                    dev_batches = self.batch_iter(list(zip(kfold_X_dev, kfold_X_token_dev, kfold_X_mask_dev, kfold_X_type_dev, kfold_y_dev, kfold_y_dev_dill)), self.batch_size, shuffle=False)
                                    for dev_batch in dev_batches:
                                        x_batch, x_token, x_mask, x_type, y_batch, y2_batch = zip(*dev_batch)  # zip(*) == unzip
                                        loss, dev_pred = dev_step(x_batch, x_token, x_mask, x_type, y_batch, y2_batch, step)
                                        dev_losses.append(loss)
                                        dev_preds.append(dev_pred)
                                    loss = np.mean(dev_losses)
                                    dev_preds = np.concatenate(dev_preds, axis=0)
                                    dev_f1 = self.get_f1_score(dev_preds, np.argmax(kfold_y_dev, -1), verbose=True)
                                    if dev_f1 > max_f1:
                                        early_stop = 0
                                        print("Evaluation:")
                                        print('{}: step {}, lr {:g}, loss {:g}, f1 {:g}'.format(time_str, step, sess.run(optimizer._lr), loss, dev_f1))
                                        min_loss = loss
                                        max_f1 = dev_f1
                                        time_str = datetime.datetime.now().isoformat()
                                        print('saving model')
                                        path = saver.save(sess, checkpoint_prefix, global_step=0)
                                        print('have saved model to ', path)
                                    else:
                                        early_stop += 1

                                    if early_stop >= 4:
                                        #  saver.restore(sess, path)
                                        # sess.run(decay_ops)
                                        early_stop = 0


                        print('best f1 {}, loss {}'.format(max_f1, min_loss))
                        path = checkpoint_prefix + '-0'
                        print('load model:', path)
                        try:
                            saver.restore(sess, path)
                        except:
                            exit()
                        # test_batches = self.batch_iter(test_data,  self.batch_size, shuffle=False)
                        # dev_batches = self.batch_iter(kfold_X_dev, self.batch_size, shuffle=False)
                        test_batches = self.batch_iter(list(zip(test_data, test_token_id, test_mask_id, test_type_id)), self.batch_size, shuffle=False)
                        dev_batches = self.batch_iter(list(zip(kfold_X_dev, kfold_X_token_dev, kfold_X_mask_dev, kfold_X_type_dev)), self.batch_size, shuffle=False)

                        if max_f1 > max_sub_f1:  # 该次最优
                            print('\n获取临时测试结果, f1:{}->{}'.format(max_sub_f1, max_f1))
                            max_sub_f1 = max_f1
                            min_sub_loss = min_loss
                            tmp_test = (test_step(test_batches) / self.n_folds)
                            tmp_dev = test_step(dev_batches)

                    print('saving f1 {}, loss {}'.format(max_sub_f1, min_sub_loss))
                    final_loss.append(min_sub_loss)
                    final_f1.append(max_sub_f1)
                    total_dev_probs[dev_index] = tmp_dev
                    total_test_probs += tmp_test
                    if self.is_debug:
                        break

                mean_f1 = np.mean(final_f1)
                mean_loss = np.mean(final_loss)
                print('final f1:\t{} -> {}'.format(final_f1, mean_f1))
                print('final loss:\t{} -> {}'.format(final_loss, mean_loss))
                os.system('mv {} {}'.format(out_dir, os.path.join(out_dir_parent, str(round(mean_loss, 5))+'_'+str(round(mean_f1, 5)))))
                os.makedirs('../data/result-dt{}-op{}-embed{}-debug{}-distillation'.format(self.config.data_type, self.config.option, self.config.EMBED_SIZE, self.config.is_debug), exist_ok=True)
                with open('../data/result-dt{}-op{}-embed{}-debug{}-distillation/{}_oof_l{:.5f}_f{:.5f}_oe{}_bal{}.pkl'.format(self.config.data_type, self.config.option, self.config.EMBED_SIZE,
                                                                                                self.config.is_debug, self.name,
                                                                                                mean_loss,
                                                                                                mean_f1, self.config.outer_embed, self.config.balance), 'wb') as f:
                    pickle.dump(total_dev_probs, f)

                with open('../data/result-dt{}-op{}-embed{}-debug{}-distillation/{}_pre_l{:.5f}_f{:.5f}_oe{}_bal{}.pkl'.format(self.config.data_type, self.config.option, self.config.EMBED_SIZE,
                                                                                                self.config.is_debug, self.name,
                                                                                                mean_loss,
                                                                                                mean_f1, self.config.outer_embed, self.config.balance), 'wb') as f:
                    pickle.dump(total_test_probs, f)
                print('done')


class BasicStaticModel(BasicModel):

    def __init__(self, config=None, n_folds=5, name='BasicStaticModel'):
        self.n_folds = n_folds
        self.name = name
        self.config = config
        self.kf = KFold(n_splits=n_folds, shuffle=True, random_state=10)

    def train_predict(self, train, train_y, test, option=None):
        name = self.name

        predict = np.zeros((test.shape[0], 10, 4))
        oof_predict = np.zeros((train.shape[0], 10, 4))
        scores_f1 = []

        for train_index, dev_index in self.kf.split(train):
            kfold_X_train, kfold_X_val = train[train_index], train[dev_index]
            y_train, y_dev = train_y[train_index], train_y[dev_index]

            model_dict = {}
            print('start train model:')
            for idx in tqdm(range(10)):
                label = y_train[:, idx]
                model = self.create_model()
                model.fit(kfold_X_train, label)
                model_dict[idx] = model
            print('complete train model')
            print('start validate model')
            f1_scores = []
            for idx in tqdm(range(10)):
                label_dev = y_dev[:, idx]
                model = model_dict[idx]
                dev_prob = model.predict_proba(kfold_X_val)
                test_prob = model.predict_proba(test)

                oof_predict[dev_index, idx] = dev_prob
                predict[:, idx] += test_prob / self.n_folds

                dev_predict = np.argmax(dev_prob, 1)
                f1_scores.append(self.get_f1_score(dev_predict, label_dev))
            f1_score = np.mean(f1_scores)
            scores_f1.append(f1_score)
            print('f1_scores-> ', f1_scores)
            print('f1_score: ', f1_score)
            if self.config.is_debug == True:
                break

        print('Total f1->', scores_f1)
        print("Total f1'mean is ", np.mean(scores_f1))

        # 保存结果
        os.makedirs('../data/result-ml', exist_ok=True)

        with open('../data/result-ml/{}_oof_f1_{}.pkl'.format(name, str(np.mean(scores_f1))), 'wb') as f:
            pickle.dump(oof_predict, f)

        with open('../data/result-ml/{}_pre_f1_{}.pkl'.format(name, str(np.mean(scores_f1))), 'wb') as f:
            pickle.dump(predict, f)

        print('done')



if __name__ == '__main__':
    bm = BasicModel()
    print(bm.cal_f_alpha([[0, 0, 1], [0, 1, 0], [0, 1, 0]], [2, 1, 0], alpha=1.0, n_out=3))

