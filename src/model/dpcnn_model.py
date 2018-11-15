from keras.models import *
from keras.layers import *
from model.model_basic import BasicDeepModel
from keras import regularizers


dp = 4
filter_nr = 64
filter_size = 3
max_pool_size = 3
max_pool_strides = 2
dense_nr = 128
spatial_dropout = 0.5
dense_dropout = 0.5


class DpcnnModel(BasicDeepModel):
    def __init__(self, name='basicModel', num_flods=5, config=None):
        name = 'dpcnn' + config.main_feature
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

        block1 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(x)
        block1 = BatchNormalization()(block1)
        block1 = PReLU()(block1)
        block1 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block1)
        block1 = BatchNormalization()(block1)
        block1 = PReLU()(block1)

        # we pass embedded comment through conv1d with filter size 1 because it needs to have the same shape as block output
        # if you choose filter_nr = embed_size (300 in this case) you don't have to do this part and can add emb_comment directly to block1_output
        resize_emb = Conv1D(filter_nr, kernel_size=1, padding='same', activation='linear')(x)
        resize_emb = PReLU()(resize_emb)

        block1_output = add([block1, resize_emb])
        x = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block1_output)

        for i in range(dp):
            block1 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(x)
            block1 = BatchNormalization()(block1)
            block1 = PReLU()(block1)
            block1 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block1)
            block1 = BatchNormalization()(block1)
            block1 = PReLU()(block1)

            block_output = add([block1, x])
            if i + 1 != dp:
                x = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block_output)

        x = GlobalMaxPooling1D()(block_output)
        output = Dense(dense_nr, activation='linear')(x)
        output = BatchNormalization()(output)
        x = PReLU()(output)

        # output = Dropout(dense_dropout)(output)
        if self.config.data_type == 3:
            dense = Dense(self.n_class, activation="sigmoid")(x)
        else:
            dense = Dense(self.n_class, activation="softmax")(x)
        res_model = Model(inputs=[input], outputs=dense)

        return res_model
