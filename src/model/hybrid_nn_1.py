from keras.models import *
from keras.layers import *
from keras import backend as K
from model.model_basic import BasicDeepModel
from model.model_component import AttLayer
from model.model_component import Capsule


class HybridNN1Model(BasicDeepModel):
    def __init__(self, name='basicModel', num_flods=5, config=None):
        name = 'hybridnn1' + config.main_feature
        BasicDeepModel.__init__(self, name=name, n_folds=num_flods, config=config)

    def create_model(self):
        if self.main_feature == 'char':
            input = Input(shape=(self.max_len,), name='char')
        else:
            input = Input(shape=(self.max_len,), name='word')

        embedding = Embedding(self.max_features, self.embed_size, weights=[self.embedding], trainable=True, name='embedding')
        x = Masking(mask_value=self.mask_value)(input)
        x = embedding(x)

        x = SpatialDropout1D(0.5)(x)
        x = GRU(256, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)(x) # ??
        capsule1 = Capsule(19, 17, 5)(x)
        capsule1 = Flatten()(capsule1)
        capsule2 = Capsule(19, 16, 5)(x)
        capsule2 = Flatten()(capsule2)
        output = concatenate([capsule1, capsule2])

        output = Dense(256)(output)
        output = BatchNormalization()(output)
        output = Activation('relu')(output)
        output = Dropout(0.2)(output)

        output = Dense(256)(output)
        output = BatchNormalization()(output)
        output = Activation('relu')(output)
        x = Dropout(0.2)(output)

        if self.config.data_type == 3:
            dense = Dense(self.n_class, activation="sigmoid")(x)
        else:
            dense = Dense(self.n_class, activation="softmax")(x)
        model = Model(inputs=[input], output=dense)

        return model


