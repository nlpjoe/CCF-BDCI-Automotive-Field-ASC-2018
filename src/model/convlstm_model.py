from model.model_basic import BasicDeepModel
import tensorflow as tf
from bilm.model import BidirectionalLanguageModel,dump_token_embeddings
from bilm.elmo import weight_layers

n_sub = 10

class ConvlstmModel(BasicDeepModel):
    def __init__(self, name='basicModel', n_folds=5, config=None):
        name = 'convlstm' + config.main_feature
        self.hidden_dim = 300
        BasicDeepModel.__init__(self, name=name, n_folds=n_folds, config=config)

    def LSTM(self, layers=1):
        lstms = []
        for num in range(layers):
            lstm = tf.contrib.rnn.BasicLSTMCell(self.hidden_dim, forget_bias=1.0)
            print(lstm.name)
            # lstm = tf.contrib.rnn.GRUCell(self.hidden_dim)
            lstm = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=self.output_keep_prob)
            lstms.append(lstm)
        lstms = tf.contrib.rnn.MultiRNNCell(lstms)
        return lstms

    def GRU(self, layers=1):
        lstms = []
        for num in range(layers):
            #  lstm = tf.contrib.rnn.BasicLSTMCell(self.hidden_dim, forget_bias=1.0)
            lstm = tf.contrib.rnn.GRUCell(self.hidden_dim)
            print(lstm.name)
            lstm = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=self.output_keep_prob)
            lstms.append(lstm)
        lstms = tf.contrib.rnn.MultiRNNCell(lstms)
        return lstms

    def create_model(self, share_dense=True, concat_sub=True):
        self.input_y = tf.placeholder(dtype=tf.float32, shape=[None,n_sub,4], name='input_y')
        self.input_y2 = tf.placeholder(dtype=tf.float32, shape=[None,n_sub,4], name='input_y2')
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, name='dropout_keep_prob')
        self.output_keep_prob = tf.placeholder(dtype=tf.float32, name='output_keep_prob')

        if self.main_feature.lower() in ['word', 'char']:
            self.input_x = tf.placeholder(dtype=tf.int32, shape=[None,self.max_len], name='input_x')
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
            bilm_embedding = weight_layers('output', bilm_embedding_op,l2_coef=0.0)
            self.word_encoding = bilm_embedding['weighted_op']
            self.word_encoding = tf.nn.dropout(self.word_encoding, self.dropout_keep_prob) # new

        else:
            exit('wrong feature')

        inputs_expanded = tf.expand_dims(self.word_encoding, -1)
        n_filters = 128
        filter_shape = [3, self.embed_size, 1, n_filters]
        W = tf.get_variable(initializer=tf.truncated_normal(filter_shape, stddev=0.1), name='W')
        b = tf.get_variable('b', initializer=tf.constant(0.1, shape=[n_filters]))
        conv = tf.nn.conv2d(inputs_expanded, W, strides=[1]*4, padding='VALID', name='conv2d')
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
        h = tf.reshape(h, [-1, self.max_len-3+1, n_filters])

        self.forward = self.LSTM()
        self.backward = self.LSTM()
        x, _ = tf.nn.bidirectional_dynamic_rnn(self.forward,self.backward, h, dtype=tf.float32)
        x = tf.concat(x, -1)
        output_sentence = tf.layers.dense(x, self.hidden_dim, activation=tf.nn.relu)

        x_reshape = tf.reshape(output_sentence, [-1, 1, self.max_len-3+1, self.hidden_dim])
        x_tile = tf.tile(x_reshape, [1, n_sub, 1, 1])  # 句子复制n_sub份

        sub_embedding = tf.get_variable(shape=[n_sub, self.hidden_dim], name='sub_embedding')
        sub_reshape = tf.reshape(sub_embedding, [1, n_sub, 1, self.hidden_dim])
        sub_tile = tf.tile(sub_reshape, [self.batch_size, 1, self.max_len-3+1, 1])

        embed_concat = tf.reshape(tf.concat((x_tile, sub_tile), -1), [-1, 2*self.hidden_dim])

        att_w = tf.get_variable(shape=[2*self.hidden_dim, self.hidden_dim], name='att_w')
        att_b = tf.get_variable(shape=[self.hidden_dim], name='att_b')
        att_v = tf.get_variable(shape=[self.hidden_dim, 1], name='att_v')

        score = tf.matmul(tf.nn.tanh(tf.matmul(embed_concat, att_w) + att_b), att_v)
        score_fit = tf.reshape(score, [-1, n_sub, self.max_len-3+1])
        alpha = tf.nn.softmax(score_fit)

        layer_sentence = tf.matmul(alpha, output_sentence)

        if concat_sub:
            # 是否拼接layer_sub信息
            layer_sub = tf.reshape(sub_embedding, [1, n_sub, self.hidden_dim])
            layer_sub_tile = tf.tile(layer_sub, [self.batch_size, 1, 1])

            layer_total = tf.concat((layer_sentence, layer_sub_tile), -1)
            outputs = tf.reshape(layer_total, [-1, 2*self.hidden_dim])
        else:
            outputs = tf.reshape(layer_sentence, [-1, self.hidden_dim])

        self.logits = tf.layers.dense(layer_sentence, 4, activation=None)
        y_ = tf.nn.softmax(self.logits)
        self.prob = tf.reshape(y_, [-1, 10, 4])
        self.prediction = tf.argmax(self.prob, 2, name="prediction")

        if not self.config.balance:
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=tf.reshape(self.input_y, [-1,4])))
            self.loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=tf.reshape(self.input_y2, [-1,4])))
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

