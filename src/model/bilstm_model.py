from model.model_basic import BasicDeepModel
from model import modeling
import tensorflow as tf
from bilm.model import BidirectionalLanguageModel,dump_token_embeddings
from bilm.elmo import weight_layers

n_sub = 10

class BilstmV0(BasicDeepModel):
    def __init__(self, name='basicModel', n_folds=5, config=None):
        name = 'qiuqiuv0' + config.main_feature
        self.hidden_dim = 300
        BasicDeepModel.__init__(self, name=name, n_folds=n_folds, config=config)

    def create_model(self):
        self.input_y = tf.placeholder(dtype=tf.float32, shape=[None,10,4], name='input_y')
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

        self.layer_embedding = tf.get_variable(shape=[10, self.hidden_dim], name='layer_embedding')
        # self.layer_embedding = tf.get_variable(initializer=self.sentiment_embed, name='layer_embedding')

        self.forward = self.LSTM()
        self.backwad = self.LSTM()
        # self.forward2 = self.LSTM()
        # self.backwad2 = self.LSTM()

        # add point
        self.forward2 = self.GRU()
        self.backwad2 = self.GRU()

        with tf.variable_scope('sentence_encode'):
            all_output_words, _ = tf.nn.bidirectional_dynamic_rnn(self.forward,self.backwad,self.word_encoding,dtype=tf.float32)
        # output_sentence = 0.5*(all_output_words[0] + all_output_words[1])
        output_sentence = tf.concat(axis=2, values=all_output_words)

        with tf.variable_scope('sentence_encode2'):
            all_output_words, _ = tf.nn.bidirectional_dynamic_rnn(self.forward2,self.backwad2,output_sentence,dtype=tf.float32)
        # output_sentence = 0.5*(all_output_words[0] + all_output_words[1])
        output_sentence = tf.concat(axis=2, values=all_output_words)
        output_sentence = tf.layers.dense(output_sentence, self.hidden_dim, activation=tf.nn.tanh)
        sentence_reshape = tf.reshape(output_sentence, [-1, 1, self.max_len, self.hidden_dim])
        sentence_reshape_tile = tf.tile(sentence_reshape, [1, 10, 1, 1])  # 句子复制10份

        layer_reshape = tf.reshape(self.layer_embedding, [1, 10, 1, self.hidden_dim])
        layer_reshape_tile = tf.tile(layer_reshape, [self.batch_size, 1, self.max_len, 1])

        embed_concat = tf.reshape(tf.concat(axis=3,values=[sentence_reshape_tile,layer_reshape_tile]),[-1,2*self.hidden_dim])

        self.att_w = tf.get_variable(shape=[2*self.hidden_dim,self.hidden_dim],name='att_w')
        self.att_b = tf.get_variable(shape=[self.hidden_dim],name='att_b')
        self.att_v = tf.get_variable(shape=[self.hidden_dim,1],name='att_v')

        score = tf.reshape(tf.matmul(tf.nn.tanh(tf.matmul(embed_concat,self.att_w) + self.att_b),self.att_v),[-1,10,self.max_len])
        alpah = tf.nn.softmax(score,axis=2)
        layer_sentence = tf.matmul(alpah,output_sentence)

        layer_reshape2 = tf.reshape(self.layer_embedding,[1,10,self.hidden_dim])
        layer_reshape2_tile = tf.tile(layer_reshape2,[self.batch_size,1,1])
        layer_sentence = tf.concat(axis=2,values=[layer_sentence,layer_reshape2_tile])
        layer_sentence = tf.reshape(layer_sentence,[-1,2*self.hidden_dim])

        layer_sentence = tf.layers.dense(layer_sentence,self.hidden_dim,activation=tf.nn.relu)

        # add point
        layer_sentence = tf.nn.dropout(layer_sentence, self.dropout_keep_prob)

        self.logits = tf.layers.dense(layer_sentence, 4, activation=None)
        y_ = tf.nn.softmax(self.logits, axis=1)
        self.prob = tf.reshape(y_, [-1, 10, 4])
        self.prediction = tf.argmax(self.prob, 2, name="prediction")

        if not self.config.balance:
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=tf.reshape(self.input_y, [-1,4])))
            #  self.loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=tf.reshape(self.input_y2, [-1,4])))
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


class BilstmV1(BasicDeepModel):
    def __init__(self, name='basicModel', n_folds=5, config=None):
        name = 'qiuqiuv1' + config.main_feature
        self.hidden_dim = 300
        BasicDeepModel.__init__(self, name=name, n_folds=n_folds, config=config)

    def create_model(self, concat_sub=True):
        self.input_y = tf.placeholder(dtype=tf.float32, shape=[None,10,4], name='input_y')
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

        self.layer_embedding = tf.get_variable(shape=[10, self.hidden_dim], name='layer_embedding')
        layer_reshape = tf.reshape(self.layer_embedding, [1, 10, 1, self.hidden_dim])
        layer_reshape_tile = tf.tile(layer_reshape, [self.batch_size, 1, self.max_len, 1])

        self.forward = self.LSTM()
        self.backwad = self.LSTM()
        self.forward2 = self.LSTM()
        self.backwad2 = self.LSTM()

        with tf.variable_scope('sentence_encode'):
            s1_out, _ = tf.nn.bidirectional_dynamic_rnn(self.forward,self.backwad,self.word_encoding,dtype=tf.float32)
        # output_sentence = 0.5*(all_output_words[0] + all_output_words[1])
        s1_out = tf.concat(axis=2, values=s1_out)
        s1_reshape = tf.reshape(s1_out, [-1, 1, self.max_len, 2*self.hidden_dim])
        s1_tile = tf.tile(s1_reshape, [1, 10, 1, 1])  # 第一层lstm复制10份

        s2_input = tf.reshape(tf.concat((s1_tile, layer_reshape_tile), -1), [-1, self.max_len, 3*self.hidden_dim])

        with tf.variable_scope('sentence_encode2'):
            s2_out, _ = tf.nn.bidirectional_dynamic_rnn(self.forward2,self.backwad2,s2_input,dtype=tf.float32)
        # output_sentence = 0.5*(all_output_words[0] + all_output_words[1])
        s2_out = tf.reshape(tf.concat(axis=-1, values=s2_out), [-1, 10, self.max_len, 2*self.hidden_dim])
        res_out = s2_out + s1_tile
        res_dense = tf.layers.dense(res_out, self.hidden_dim, activation=tf.nn.relu)

        res_layer_concat = tf.reshape(tf.concat((res_dense, layer_reshape_tile), -1), [-1, 2*self.hidden_dim])

        self.att_w = tf.get_variable(shape=[2*self.hidden_dim,self.hidden_dim],name='att_w')
        self.att_b = tf.get_variable(shape=[self.hidden_dim],name='att_b')
        self.att_v = tf.get_variable(shape=[self.hidden_dim,1],name='att_v')

        score = tf.reshape(tf.matmul(tf.nn.tanh(tf.matmul(res_layer_concat, self.att_w) + self.att_b),self.att_v),[-1,1,self.max_len])
        alpha = tf.nn.softmax(score)
        layer_sentence = tf.reshape(tf.matmul(alpha, tf.reshape(res_out, [-1, self.max_len, 2*self.hidden_dim])), [-1, n_sub, 2*self.hidden_dim])

        if concat_sub:
            # 是否拼接layer_sub信息
            layer_sub = tf.reshape(self.layer_embedding, [1, n_sub, self.hidden_dim])
            layer_sub_tile = tf.tile(layer_sub, [self.batch_size, 1, 1])

            layer_total = tf.concat((layer_sentence, layer_sub_tile), -1)
            outputs = tf.reshape(layer_total, [-1, 3*self.hidden_dim])
        else:
            outputs = tf.reshape(layer_sentence, [-1, 2*self.hidden_dim])

        self.logits = tf.layers.dense(outputs, 4, activation=None)
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


class BilstmV2(BasicDeepModel):
    def __init__(self, name='basicModel', n_folds=5, config=None):
        name = 'qiuqiuv2' + config.main_feature
        self.hidden_dim = 300
        BasicDeepModel.__init__(self, name=name, n_folds=n_folds, config=config)

    def create_model(self):
        self.input_y = tf.placeholder(dtype=tf.float32, shape=[None,10,4], name='input_y')
        self.input_y2 = tf.placeholder(dtype=tf.float32, shape=[None,n_sub,4], name='input_y2')
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, name='dropout_keep_prob')
        self.output_keep_prob = tf.placeholder(dtype=tf.float32, name='output_keep_prob')

        self.input_ids = tf.placeholder(dtype=tf.int32, shape=[None,190], name='input_ids')
        self.mask_ids = tf.placeholder(dtype=tf.int32, shape=[None,190], name='mask_ids')
        self.type_ids = tf.placeholder(dtype=tf.int32, shape=[None,190], name='type_ids')
        self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')

        #  bert_hidden_size = bert_output_layer.shape[-1].value
        #  hidden_size = output_layer.shape[-1].value

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

        self.layer_embedding = tf.get_variable(shape=[10, self.hidden_dim], name='layer_embedding')

        self.forward = self.LSTM()
        self.backwad = self.LSTM()
        # self.forward2 = self.LSTM()
        # self.backwad2 = self.LSTM()

        # add point
        self.forward2 = self.GRU()
        self.backwad2 = self.GRU()

        # bert使用
        bert_config = modeling.BertConfig.from_json_file(self.config.BERT_CONFIG_FILES)

        bert_model = modeling.BertModel(
            config=bert_config,
            is_training=self.is_training,
            input_ids=self.input_ids,
            input_mask=self.mask_ids,
            token_type_ids=self.type_ids
        )
        if self.is_training is not None:
           print('bert config hidden dropout -- ---', bert_config.hidden_dropout_prob)
           print('bert config hidden dropout -- ---', bert_config.attention_probs_dropout_prob)
        self.word_encoding = bert_model.get_sequence_output()
        all_layer_output = bert_model.get_all_encoder_layers()
        self.word_encoding = (all_layer_output[0] + all_layer_output[1] + all_layer_output[2] + all_layer_output[3]) / 4
        with tf.variable_scope('sentence_encode'):
            all_output_words, _ = tf.nn.bidirectional_dynamic_rnn(self.forward,self.backwad,self.word_encoding,dtype=tf.float32)
        # output_sentence = 0.5*(all_output_words[0] + all_output_words[1])
        output_sentence = tf.concat(axis=2, values=all_output_words)

        with tf.variable_scope('sentence_encode2'):
            all_output_words, _ = tf.nn.bidirectional_dynamic_rnn(self.forward2,self.backwad2,output_sentence,dtype=tf.float32)
        # output_sentence = 0.5*(all_output_words[0] + all_output_words[1])
        output_sentence = tf.concat(axis=2, values=all_output_words)
        output_sentence = tf.layers.dense(output_sentence, self.hidden_dim, activation=tf.nn.tanh)
        sentence_reshape = tf.reshape(output_sentence, [-1, 1, self.max_len, self.hidden_dim])
        sentence_reshape_tile = tf.tile(sentence_reshape, [1, 10, 1, 1])  # 句子复制10份

        layer_reshape = tf.reshape(self.layer_embedding, [1, 10, 1, self.hidden_dim])
        layer_reshape_tile = tf.tile(layer_reshape, [self.batch_size, 1, self.max_len, 1])

        embed_concat = tf.reshape(tf.concat(axis=3,values=[sentence_reshape_tile,layer_reshape_tile]),[-1,2*self.hidden_dim])

        self.att_w = tf.get_variable(shape=[2*self.hidden_dim,self.hidden_dim],name='att_w')
        self.att_b = tf.get_variable(shape=[self.hidden_dim],name='att_b')
        self.att_v = tf.get_variable(shape=[self.hidden_dim,1],name='att_v')

        score = tf.reshape(tf.matmul(tf.nn.tanh(tf.matmul(embed_concat,self.att_w) + self.att_b),self.att_v),[-1,10,self.max_len])
        alpah = tf.nn.softmax(score,axis=2)
        layer_sentence = tf.matmul(alpah,output_sentence)

        layer_reshape2 = tf.reshape(self.layer_embedding,[1,10,self.hidden_dim])
        layer_reshape2_tile = tf.tile(layer_reshape2,[self.batch_size,1,1])
        layer_sentence = tf.concat(axis=2,values=[layer_sentence,layer_reshape2_tile])
        layer_sentence = tf.reshape(layer_sentence,[-1,2*self.hidden_dim])

        layer_sentence = tf.layers.dense(layer_sentence,self.hidden_dim,activation=tf.nn.relu)

        # add point
        layer_sentence = tf.nn.dropout(layer_sentence, self.dropout_keep_prob)

        self.logits = tf.layers.dense(layer_sentence, 4, activation=None)
        y_ = tf.nn.softmax(self.logits, axis=1)
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

