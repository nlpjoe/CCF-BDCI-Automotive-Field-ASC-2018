from keras.layers import *
from keras.models import *
from model.model_basic import BasicDeepModel
from model.model_component import Capsule
from keras import regularizers

class CapsuleModel(BasicDeepModel):
    def __init__(self, name='basicModel', num_flods=5, config=None):
        name = 'capsule' + config.main_feature
        BasicDeepModel.__init__(self, name=name, n_folds=num_flods, config=config)

    def create_model(self):
        Routings = 5
        Num_capsule = 10
        Dim_capsule = 16
        dropout_p = 0.25
        rate_drop_dense = 0.28
        gru_len = 128
        if self.main_feature == 'char':
            input = Input(shape=(self.max_len,), name='char')
        else:
            input = Input(shape=(self.max_len,), name='word')

        embedding = Embedding(self.max_features, self.embed_size, weights=[self.embedding], trainable=True, name='embedding')
        x = Masking(mask_value=self.mask_value)(input)
        x = embedding(x)

        x = SpatialDropout1D(rate_drop_dense)(x)

        x = Bidirectional(GRU(gru_len, activation='relu', dropout=dropout_p, recurrent_dropout=dropout_p, return_sequences=True))(x)
        # x = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)

        capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings,
                          share_weights=True)(x)

        capsule = Flatten()(capsule)
        capsule = Dropout(dropout_p)(capsule)
        dense = Dense(self.n_class, activation="softmax")(capsule)
        res_model = Model(inputs=[input], outputs=dense)

        return res_model
