from keras.models import *
from keras.layers import *
from model.model_basic import BasicDeepModel
from model.model_component import AttLayer
from model.model_component import AttentionWithContext


class HANModel(BasicDeepModel):
    def __init__(self, name='basicModel', num_flods=5, config=None):
        name = 'han' + config.main_feature
        BasicDeepModel.__init__(self, name=name, n_folds=num_flods, config=config)

    def create_model(self):

        if self.config.main_feature == 'word':
            input = Input(shape=(self.config.HANN_WORD_LEN,), dtype='int32')
        else:
            input = Input(shape=(self.config.HANN_CHAR_LEN,), dtype='int32')

        mask = Masking(mask_value=self.mask_value)(input)
        embedding = Embedding(self.max_features, self.embed_size, weights=[self.embedding], trainable=True, name='embedding')
        x = embedding(mask)
        x = SpatialDropout1D(0.5)(x)
        x = Bidirectional(GRU(128, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
        l_att = AttLayer(100)(x)
        # l_att = AttentionWithContext()(x)
        sentEncoder = Model(input, l_att)

        if self.config.main_feature == 'word':
            word_input = Input(shape=(self.config.HANN_SENT, self.config.HANN_WORD_LEN), name='hann_word')
            word_encoder = TimeDistributed(sentEncoder)(word_input)
            word_sent_lstm = Bidirectional(GRU(128, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(word_encoder)
            #  x = AttLayer(100)(word_sent_lstm)
            x = AttentionWithContext()(word_sent_lstm)
            x = Dropout(0.2)(x)
            if self.config.data_type == 3:
                dense = Dense(self.n_class, activation="sigmoid")(x)
            else:
                dense = Dense(self.n_class, activation="softmax")(x)
            model = Model(word_input, dense)
        else:
            char_input = Input(shape=(self.config.HANN_SENT, self.config.HANN_CHAR_LEN), name='hann_char')
            char_encoder = TimeDistributed(sentEncoder)(char_input)
            char_sent_lstm = Bidirectional(GRU(128, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(char_encoder)
            x = AttLayer(100)(char_sent_lstm)
            # x = AttentionWithContext()(char_sent_lstm)
            x = Dropout(0.2)(x)
            if self.config.data_type == 3:
                dense = Dense(self.n_class, activation="sigmoid")(x)
            else:
                dense = Dense(self.n_class, activation="softmax")(x)
            model = Model(char_input, dense)
        return model


