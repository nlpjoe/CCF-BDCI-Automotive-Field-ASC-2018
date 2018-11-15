from model.model_basic import BasicDeepModel
from bilm.model import BidirectionalLanguageModel,dump_token_embeddings
from bilm.elmo import weight_layers
import tensorflow as tf
import numpy as np

from tensorflow.contrib import rnn
import tensorflow.contrib.layers as layers

filter_sizes = [1, 2, 3, 4]
n_filter = 128
hidden_size = 300
n_sub = 10
n_sent = 4


class RCNNModel(BasicDeepModel):
    def __init__(self, name='basicModel', n_folds=10, config=None):
        name = 'RCNN' + config.main_feature
        BasicDeepModel.__init__(self, name=name, n_folds=n_folds, config=config)

    def create_model(self, share_dense=True):
        self.input_y = tf.placeholder(dtype=tf.float32, shape=[None, n_sub, n_sent], name='input_y')
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, name='dropout_keep_prob')
        self.output_keep_prob = tf.placeholder(dtype=tf.float32, name='output_keep_prob')

        if self.main_feature.lower() in ['word', 'char']:
            self.input_x = tf.placeholder(dtype=tf.int32, shape=[None, self.max_len], name='input_x')
            self.word_embedding = tf.get_variable(initializer=self.embedding, name='word_embedding')
            self.word_encoding = tf.nn.embedding_lookup(self.embedding, self.input_x)
            self.word_encoding = tf.nn.dropout(self.word_encoding, self.dropout_keep_prob) # new

        elif self.main_feature.lower() in ['elmo_word', 'elmo_char', 'elmo_qiuqiu']:
            self.input_x = tf.placeholder(dtype=tf.int32, shape=[None,self.max_len+2], name='input_x')
            if self.main_feature == 'elmo_word':
                options_file = self.config.elmo_word_options_file
                weight_file = self.config.elmo_word_weight_file
                embed_file = self.config.elmo_word_embed_file
            elif self.main_feature == 'elmo_char':
                options_file = self.config.elmo_char_options_file
                weight_file = self.config.elmo_char_weight_file
                embed_file = self.config.elmo_char_embed_file
            elif self.main_feature == 'elmo_qiuqiu':
                options_file = self.config.elmo_qiuqiu_options_file
                weight_file = self.config.elmo_qiuqiu_weight_file
                embed_file = self.config.elmo_qiuqiu_embed_file

            self.bilm = BidirectionalLanguageModel(options_file,
                                                    weight_file,
                                                    use_character_inputs=False,
                                                    embedding_weight_file=embed_file,
                                                    max_batch_size=self.batch_size)
            bilm_embedding_op = self.bilm(self.input_x)
            bilm_embedding = weight_layers('output', bilm_embedding_op, l2_coef=0.0)
            self.word_encoding = bilm_embedding['weighted_op']
            self.word_encoding = tf.nn.dropout(self.word_encoding,self.dropout_keep_prob) # new

        else:
            exit('wrong feature')

        rcnn_outputs = []
        for i in range(n_sub):
            with tf.variable_scope('rcnn_output_%d' % i):
                output_bigru = self.bi_gru(self.word_encoding, hidden_size)
                output = self.textcnn(output_bigru, self.max_len)
                rcnn_outputs.append(output)

        n_filter_total = n_filter * len(filter_sizes)
        outputs = tf.reshape(tf.concat(rcnn_outputs, 1), (-1, n_sub, n_filter_total))

        if share_dense:
            cnn_outputs = tf.reshape(outputs, (-1, n_filter_total))
            W = tf.get_variable('W', shape=[n_filter_total, self.n_classes])
            b = tf.get_variable('b', initializer=tf.constant(0.1, shape=[self.n_classes]))
            self.logits = tf.nn.xw_plus_b(cnn_outputs, W, b, name='scores')
        else:
            cnn_outputs = tf.reshape(tf.concat(outputs, 1), (-1, n_sub, n_filter_total))
            W = tf.get_variable('W', shape=[self.batch_size, n_filter_total, self.n_classes])
            b = tf.get_variable('b', initializer=tf.constant(0.1, shape=[self.n_classes]))
            self.logits = tf.nn.xw_plus_b(cnn_outputs, W, b, name='scores')

        y_ = tf.nn.softmax(self.logits)
        self.prob = tf.reshape(y_, [-1, n_sub, 4])
        self.prediction = tf.argmax(self.prob, 2, name="prediction")

        if not self.config.balance:
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=tf.reshape(self.input_y, [-1,4])))
        else:
            #  class0_weight = 0.882 * self.n_classes  # 第0类的权重系数
            #  class1_weight = 0.019 * self.n_classes  # 第1类的权重系数
            #  class2_weight = 0.080 * self.n_classes  # 第2类的权重系数
            #  class3_weight = 0.019 * self.n_classes  # 第3类的权重系数
            class0_weight = 1  # 第0类的权重系数
            class1_weight = 3  # 第1类的权重系数
            class2_weight = 3  # 第2类的权重系数
            class3_weight = 3  # 第3类的权重系数
            #  coe = tf.constant([1., 1., 1., 1.])
            #  y = tf.reshape(self.input_y, [-1, 4]) * coe
            #  self.loss = -tf.reduce_mean(y * tf.log(y_))

            y = tf.reshape(self.input_y, [-1, 4])
            self.loss = tf.reduce_mean(-class0_weight * (y[:, 0]*tf.log(y_[:, 0]))
                                        -class1_weight * (y[:, 1]*tf.log(y_[:, 1]))
                                        -class2_weight * (y[:, 2]*tf.log(y_[:, 2]))
                                        -class3_weight * (y[:, 3]*tf.log(y_[:, 3])))
            #  tf.reduce_mean(-class1_weight*tf.reduce_sum(y_[:,0] * tf.log(y[:,0])-class2_weight*tf.reduce_sum(y_[:,1] * tf.log(y[:,1])-class3_weight*tf.reduce_sum(y_[:,2] * tf.log(y[:,2]))

        return self

    def textcnn(self, cnn_inputs, n_step):
        # cnn_inputs = [batch_size, n_step, -1]
        inputs = tf.expand_dims(cnn_inputs, -1)
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope('conv-maxpool-%s' % filter_size):
                filter_shape = [filter_size, hidden_size*2+self.embed_size, 1, n_filter]
                W_filter = tf.get_variable(initializer=tf.truncated_normal(filter_shape, stddev=0.1), name='W_filter')
                beta = tf.get_variable(initializer=tf.constant(0.1, shape=[n_filter]), name='beta')
                conv = tf.nn.conv2d(inputs, W_filter, strides=[1]*4, padding='VALID', name='conv')
                h = tf.nn.relu(tf.nn.bias_add(conv, beta), name='relu')
                pooled = tf.nn.max_pool(h, ksize=[1, n_step - filter_size + 1, 1, 1],
                                        strides=[1]*4, padding='VALID', name='pool')
                pooled_outputs.append(pooled)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, n_filter * len(filter_sizes)])
        h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)
        return h_drop

    def gru_cell(self, hidden_size):
        cell = rnn.GRUCell(hidden_size, reuse=tf.get_variable_scope().reuse)
        return rnn.DropoutWrapper(cell, output_keep_prob=self.output_keep_prob)

    def bi_gru(self, inputs, hidden_size, res_add=True):
        """build the bi-GRU network. Return the encoder represented vector.
        X_inputs: [batch_size, n_step]
        n_step: 句子的词数量；或者文档的句子数。
        outputs: [batch_size, n_step, hidden_size*2+embedding_size(if res_add)]
        """
        cells_fw = [self.gru_cell(hidden_size) for _ in range(1)]
        cells_bw = [self.gru_cell(hidden_size) for _ in range(1)]
        initial_states_fw = [cell_fw.zero_state(self.batch_size, tf.float32) for cell_fw in cells_fw]
        initial_states_bw = [cell_bw.zero_state(self.batch_size, tf.float32) for cell_bw in cells_bw]
        outputs, _, _ = rnn.stack_bidirectional_dynamic_rnn(cells_fw, cells_bw, inputs,
                                                            initial_states_fw=initial_states_fw,
                                                            initial_states_bw=initial_states_bw,
                                                            dtype=tf.float32)
        if res_add:
            outputs = tf.concat([outputs, inputs], axis=2)
        return outputs

    # def batchnorm(self, Ylogits, offset, convolutional=False):
        # exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, )

