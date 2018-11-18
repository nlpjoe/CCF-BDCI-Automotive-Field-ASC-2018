# from model.lightgbm_model import LightGbmModel
# from model.xgboost_model import XgboostModel
from model.textcnn_model import TextCNNModel
from model.dpcnn_model import DpcnnModel
from model.capsule_model import CapsuleModel
from model.rcnn_model import RCNNModel
from model.attention import AttentionModel
from model.convlstm_model import ConvlstmModel
from model.lstmconv_model import LstmconvModel
from model.lstmgru_model import LstmgruModel
from model.han_model import HANModel
from model.hybrid_nn_1 import HybridNN1Model
from model.ml_models import SVCClassifier
from model.ml_models import Fasttext
from model.bilstm_model import *


class Config(object):

    """Docstring for Config. """

    def __init__(self):
        """TODO: to be defined1. """
        self.model = {
            # 'xgboost': XgboostModel,
            # 'lightgbm': LightGbmModel,
            # 'svc': SVCClassifier,
            # 'fasttext': Fasttext,

            # dl model
            'aspv0': BilstmV0,
            'aspv1': BilstmV1,
            # 'aspv2': BilstmV2,
            'textcnn': TextCNNModel,
            'lstmgru': LstmgruModel,
            'attention': AttentionModel,
            'convlstm': ConvlstmModel,
            'lstmconv': LstmconvModel,
            # 'dpcnn': DpcnnModel,
            # 'rcnn': RCNNModel,
            # 'capsule': CapsuleModel,
            # 'han': HANModel,
            # 'hybridnn1': HybridNN1Model,
        }
        self.CHAR_MAXLEN = 190
        self.WORD_MAXLEN = 128

        self.HANN_SENT = 20
        self.HANN_WORD_LEN = 40
        self.HANN_CHAR_LEN = 70
        self.EMBED_SIZE = 300
        self.main_feature = 'word'
        self.is_debug = True
        # self.elmo_word_options_file = './bilm/dump/options.word.json'
        # self.elmo_word_weight_file = './bilm/dump/weights.word.hdf5'
        # self.elmo_word_embed_file = './bilm/dump/vocab_embedding.word.hdf5'
        # self.elmo_word_vocab_file = '../data/word2vec_models/word2vec.word.300d.vocab.txt'

        # self.elmo_char_options_file = './bilm/dump/options.char.json'
        # self.elmo_char_weight_file = './bilm/dump/weights.char.hdf5'
        # self.elmo_char_embed_file = './bilm/dump/vocab_embedding.char.hdf5'
        # self.elmo_char_vocab_file = '../data/word2vec_models/word2vec.char.300d.vocab.txt'

        # self.elmo_qiuqiu_options_file = './bilm/dump/tmp/options.json'
        # self.elmo_qiuqiu_weight_file = './bilm/dump/tmp/weight-11-4.hdf5'
        # self.elmo_qiuqiu_embed_file = './bilm/dump/tmp/word_embedding.after.elmo-11-4.hdf5'
        # self.elmo_qiuqiu_vocab_file = './bilm/dump/tmp/sa_elmo_vocabs.txt'

        self.loss_path = '../data/loss'
        self.TEST_X = '../data/csvs/test_public.csv'
        self.TRAIN_MULTI_X = '../data/csvs/train_multi.csv'
        self.TRAIN_JP = '../data/csvs/round2zh2jp.csv'
        self.TRAIN_EN = '../data/csvs/round2zh2en.csv'
        # self.SENTIMENT_EMBED_PATH = '../data/sentiment_embedding.pkl'

        # self.BERT_VOCAB_FILES = '../data/chinese_L-12_H-768_A-12/vocab.txt'
        # self.BERT_CONFIG_FILES = '../data/chinese_L-12_H-768_A-12/bert_config.json'

        # self.Y_DISTILLATION = '../data/result/oof.pkl'

    # property 等待调用到它时才计算，先加载embed size再加载对应词向量
    @property
    def char_stoi_file(self):
        if self.car:
            return '../data/char_item_to_id.cars-home.pkl'
        else:
            return '../data/char_item_to_id.pkl'

    @property
    def word_stoi_file(self):
        if self.car:
            return '../data/word_item_to_id.cars-home.pkl'
        else:
            return '../data/word_item_to_id.pkl'

    @property
    def char_w2v_file(self):
        if self.outer_embed:
            return '../data/word2vec_models/sgns.baidubaike.bigram-char'
        else:
            if not self.car:
                return '../data/word2vec_models/word2vec.char.{}d.model.txt'.format(self.EMBED_SIZE)
            else:
                return '../data/word2vec_models/word2vec.char.{}d.model.cars-home.txt'.format(self.EMBED_SIZE)


    @property
    def word_w2v_file(self):

        if self.outer_embed:
            return '../data/word2vec_models/sgns.baidubaike.bigram-char'
        else:
            if not self.car:
                return '../data/word2vec_models/word2vec.word.{}d.model.txt'.format(self.EMBED_SIZE)
            else:
                return '../data/word2vec_models/word2vec.word.{}d.model.cars-home.txt'.format(self.EMBED_SIZE)

    @property
    def TRAIN_X(self):
        if self.data_type == 0:
            return '../data/csvs/train_single_label.csv'
        elif self.data_type == 1:
            return '../data/csvs/train_single_label.csv'
        elif self.data_type == 2:
            return '../data/csvs/train_multi.csv'
        elif self.data_type == 3:
            return '../data/csvs/train_multi.csv'
        elif self.data_type == 4:
            return '../data/csvs/train.csv'
        elif self.data_type == 5:
            return '../data/csvs/multi_train.csv'

    @property
    def n_classes(self):
        if self.data_type == 0:
            return 10
        elif self.data_type == 1:
            return 3
        elif self.data_type == 2:
            return 4
        elif self.data_type == 3:
            return 4
        elif self.data_type == 4:
            return 3
        elif self.data_type == 5:
            return 30



