import os
import pandas as pd
import pickle
from config import Config
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
import logging
from gensim.models.word2vec import Word2Vec
from bilm import TokenBatcher
from scipy.sparse import hstack

import tokenization
from keras.preprocessing import sequence
from keras.utils import np_utils
import tensorflow as tf

#  np.random.seed(201)
#  tf.set_random_seed(201)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)


def deep_data_prepare(config):
    print('深度学习模型数据准备')
    train_df = pd.read_csv(config.TRAIN_X)
    train_jp = pd.read_csv(config.TRAIN_JP)
    train_en = pd.read_csv(config.TRAIN_EN)
    test_df = pd.read_csv(config.TEST_X)

    char_sw_list = pickle.load(open('../data/char_stopword.pkl', 'rb'))
    word_sw_list = pickle.load(open('../data/word_stopword.pkl', 'rb'))
    # 用词向量
    # 用字向量
    train_x_char = train_df['char']
    train_x_word = train_df['word']
    # train_x_sent_word = [w for w in open('../data/sentiment_word.txt')]
    # train_x_sent_char = [w for w in open('../data/sentiment_word.txt')]
    train_jp_char = train_jp['char']
    train_jp_word = train_jp['word']
    train_en_char = train_en['char']
    train_en_word = train_en['word']

    train_char = pd.concat((train_x_char, train_jp_char, train_en_char))
    train_word = pd.concat((train_x_word, train_jp_word, train_en_word))
    test_char = test_df['char']
    test_word = test_df['word']

    if config.data_type == 0:
        train_y = train_df['sub_numerical'].values
        train_y = np_utils.to_categorical(train_y, num_classes=config.n_classes)

    elif config.data_type == 1:
        train_y = train_df['sentiment_value'].values
        train_y = np_utils.to_categorical(train_y, num_classes=config.n_classes)

    elif config.data_type == 2:
        train_y = np.array(train_df.iloc[:, 6:].values)
    elif config.data_type == 3:
        train_y = train_df.iloc[:, 6:].values
        targets = train_y.reshape(-1)
        one_hot_targets = np.eye(config.n_classes)[targets]
        train_y = one_hot_targets.reshape(-1, 10, config.n_classes)
    elif config.data_type == 4:
        train_y = (train_df['sentiment_value']+1).values
        train_y = np_utils.to_categorical(train_y, num_classes=config.n_classes)
    elif config.data_type == 5:
        train_y = train_df.iloc[:, 4:].values
    else:
        exit('错误数据类别')

    UNK_CHAR = len(char_stoi)
    PAD_CHAR = len(char_stoi) + 1

    UNK_WORD = len(word_stoi)
    PAD_WORD = len(word_stoi) + 1

    def generate_hann_data(df):
        import re
        hann_train_word = np.full(shape=(len(df['word']), config.HANN_SENT, config.HANN_WORD_LEN), fill_value=PAD_WORD)
        hann_train_char = np.full(shape=(len(df['char']), config.HANN_SENT, config.HANN_CHAR_LEN), fill_value=PAD_CHAR)

        for i, sentences in enumerate(df['word']):
            sentences = re.split(r" 。 | ， ", sentences)
            for j, sent in enumerate(sentences):
                if j < config.HANN_SENT:
                    k = 0
                    word_tokens = sent.split()
                    for _, word in enumerate(word_tokens):
                        if k < config.HANN_WORD_LEN and word not in word_sw_list and word in word_stoi:
                            hann_train_word[i, j, k] = word_stoi[word]
                            k += 1

        for i, sentences in enumerate(df['char']):
            sentences = re.split(r" 。 | ， ", sentences)
            for j, sent in enumerate(sentences):
                if j < config.HANN_SENT:
                    k = 0
                    word_tokens = sent.split()
                    for _, word in enumerate(word_tokens):
                        if k < config.HANN_CHAR_LEN and word not in char_sw_list and word in char_stoi:
                            hann_train_char[i, j, k] = char_stoi[word]
                            k += 1
        return hann_train_word, hann_train_char

    hann_train_word, hann_train_char = generate_hann_data(train_df)
    hann_test_word, hann_test_char = generate_hann_data(test_df)

    def word2id(train_dialogs, type='char'):
        if type == 'char':
            stoi = char_stoi
            max_len = config.CHAR_MAXLEN
            UNK = UNK_CHAR
            sw_list = set(char_sw_list)
        elif type == 'word':
            stoi = word_stoi
            max_len = config.WORD_MAXLEN
            UNK = UNK_WORD
            sw_list = set(word_sw_list)
        else:
            exit('类型错误')

        train_x = []
        for d in tqdm(train_dialogs):
            d = str(d).split()
            line = []
            for token in d:
                if token in sw_list\
                        or token == ''\
                        or token == ' ':
                    continue
                if token in stoi:
                    line.append(stoi[token])
                else:
                    line.append(UNK)

            train_x.append(line[:max_len])
        return train_x

    # 普通模型数据
    train_x_word = word2id(train_word, type='word')
    train_x_char = word2id(train_char, type='char')
    test_x_char = word2id(test_char, type='char')
    test_x_word = word2id(test_word, type='word')

    # train_x_sent_word = word2id(train_x_sent_word, type='word')
    # train_x_sent_char = word2id(train_x_sent_char, type='char')
    # rcnn模型数据准备
    UNK_CHAR = PAD_CHAR
    UNK_WORD = PAD_WORD

    train_word_left = [[UNK_WORD] + w[:-1] for w in train_x_word]
    train_word_right = [w[1:] + [UNK_WORD] for w in train_x_word]
    train_char_left = [[UNK_CHAR] + w[:-1] for w in train_x_char]
    train_char_right = [w[1:] + [UNK_CHAR] for w in train_x_char]

    test_word_left = [[UNK_WORD] + w[:-1] for w in test_x_word]
    test_word_right = [w[1:] + [UNK_WORD] for w in test_x_word]
    test_char_left = [[UNK_CHAR] + w[:-1] for w in test_x_char]
    test_char_right = [w[1:] + [UNK_CHAR] for w in test_x_char]

    train_x_char = sequence.pad_sequences(train_x_char, maxlen=config.CHAR_MAXLEN, dtype='int32', padding='post', truncating='post', value=UNK_CHAR)
    train_x_word = sequence.pad_sequences(train_x_word, maxlen=config.WORD_MAXLEN, dtype='int32', padding='post', truncating='post', value=UNK_WORD)
    train_x_char_left = sequence.pad_sequences(train_char_left, maxlen=config.CHAR_MAXLEN, dtype='int32', padding='post', truncating='post', value=UNK_CHAR)
    train_x_word_left = sequence.pad_sequences(train_word_left, maxlen=config.WORD_MAXLEN, dtype='int32', padding='post', truncating='post', value=UNK_WORD)
    train_x_char_right = sequence.pad_sequences(train_char_right, maxlen=config.CHAR_MAXLEN, dtype='int32', padding='post', truncating='post', value=UNK_CHAR)
    train_x_word_right = sequence.pad_sequences(train_word_right, maxlen=config.WORD_MAXLEN, dtype='int32', padding='post', truncating='post', value=UNK_WORD)

    test_x_char = sequence.pad_sequences(test_x_char, maxlen=config.CHAR_MAXLEN, dtype='int32', padding='post', truncating='post', value=UNK_CHAR)
    test_x_word = sequence.pad_sequences(test_x_word, maxlen=config.WORD_MAXLEN, dtype='int32', padding='post', truncating='post', value=UNK_WORD)
    test_x_char_left = sequence.pad_sequences(test_char_left, maxlen=config.CHAR_MAXLEN, dtype='int32', padding='post', truncating='post', value=UNK_CHAR)
    test_x_word_left = sequence.pad_sequences(test_word_left, maxlen=config.WORD_MAXLEN, dtype='int32', padding='post', truncating='post', value=UNK_WORD)
    test_x_char_right = sequence.pad_sequences(test_char_right, maxlen=config.CHAR_MAXLEN, dtype='int32', padding='post', truncating='post', value=UNK_CHAR)
    test_x_word_right = sequence.pad_sequences(test_word_right, maxlen=config.WORD_MAXLEN, dtype='int32', padding='post', truncating='post', value=UNK_WORD)

    print('train_x char shape is: ', train_x_char.shape)
    print('train_x word shape is: ', train_x_word.shape)
    print('test_x char shape is: ', test_x_char.shape)
    print('test_x word shape is: ', test_x_word.shape)

    train = {}
    test = {}
    # tokenizer = tokenization.FullTokenizer(
                    # vocab_file=config.BERT_VOCAB_FILES, do_lower_case=False)

    # def get_bert_data(corpus):
        # input_ids = []
        # input_mask = []
        # input_segment_ids = []

        # for sent in train_df['word'].values:
            # sent = ''.join(sent.strip().split())
            # tmp_token_ids = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokenizer.tokenize(sent)[:188] + ['[SEP]'])
            # tmp_mask = [1] * len(tmp_token_ids)
            # tmp_segment_ids = [0] * len(tmp_token_ids)
            # if len(tmp_token_ids) < 190:
                # tmp_segment_ids.extend([0] * (190-len(tmp_token_ids)))
                # tmp_mask.extend([0] * (190-len(tmp_token_ids)))
                # tmp_token_ids.extend([0] * (190-len(tmp_token_ids)))
            # input_ids.append(tmp_token_ids)
            # input_mask.append(tmp_mask)
            # input_segment_ids.append(tmp_segment_ids)
        # return np.array(input_ids, dtype='int32'), np.array(input_mask, dtype='int32'), np.array(input_segment_ids, dtype='int32')

    # train['token_id'], train['mask_id'], train['type_id'] = get_bert_data(train_df['word'].values)
    # test['token_id'], test['mask_id'], test['type_id'] = get_bert_data(test_df['word'].values)

    train['word'] = train_x_word
    train['char'] = train_x_char
    # train['word_sent'] = train_x_sent_word
    # train['char_sent'] = train_x_sent_char
    # rcnn
    train['word_left'] = train_x_word_left
    train['word_right'] = train_x_word_right
    train['char_left'] = train_x_char_left
    train['char_right'] = train_x_char_right
    # han
    train['hann_word'] = hann_train_word
    train['hann_char'] = hann_train_char

    test['word'] = test_x_word
    test['char'] = test_x_char
    test['word_left'] = test_x_word_left
    test['word_right'] = test_x_word_right
    test['char_left'] = test_x_char_left
    test['char_right'] = test_x_char_right
    test['hann_word'] = hann_test_word
    test['hann_char'] = hann_test_char

    assert train['word_left'].shape == train['word_right'].shape == train['word'].shape
    assert train['char_left'].shape == train['char_right'].shape == train['char'].shape
    assert test['word_left'].shape == test['word_right'].shape == test['word'].shape
    assert test['char_left'].shape == test['char_right'].shape == test['char'].shape

    # batcher = TokenBatcher(config.elmo_word_vocab_file)
    # train['elmo_word'] = batcher.batch_sentences([str(w).split()[:config.WORD_MAXLEN] for w in train_df['word']])
    # test['elmo_word'] = batcher.batch_sentences([str(w).split()[:config.WORD_MAXLEN] for w in test_df['word']])

    # batcher = TokenBatcher(config.elmo_char_vocab_file)
    # train['elmo_char'] = batcher.batch_sentences([str(w).split()[:config.CHAR_MAXLEN] for w in train_df['char']])
    # test['elmo_char'] = batcher.batch_sentences([str(w).split()[:config.CHAR_MAXLEN] for w in test_df['char']])

    # batcher = TokenBatcher(config.elmo_qiuqiu_vocab_file)
    # train['elmo_qiuqiu'] = batcher.batch_sentences([str(w).split()[:config.WORD_MAXLEN] for w in train_df['word']])
    # test['elmo_qiuqiu'] = batcher.batch_sentences([str(w).split()[:config.WORD_MAXLEN] for w in test_df['word']])

    return train, train_y, test


def init_embedding(config, type='word'):
    model_file = config.word_w2v_file if type == 'word' else config.char_w2v_file
    item_to_id = word_stoi if type == 'word' else char_stoi
    vocab_len = len(item_to_id) + 2
    print('Vocabulaty size : ', vocab_len)
    print('create embedding matrix')

    def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(model_file).readlines()[1:])

    all_embs = np.stack(embeddings_index.values())
    embed_matrix = np.random.normal(all_embs.mean(), all_embs.std(), size=(vocab_len, config.EMBED_SIZE)).astype(dtype='float32')
    embed_matrix[-1] = 0  # padding

    for word, i in tqdm(item_to_id.items()):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embed_matrix[i] = embedding_vector
    return embed_matrix


def deep_data_cache():
    char_w2v_embed = init_embedding(config, type='char')
    word_w2v_embed = init_embedding(config, type='word')

    train, train_y, test = deep_data_prepare(config)
    os.makedirs('../data/cache/', exist_ok=True)
    pickle.dump((train, train_y, test, char_w2v_embed, word_w2v_embed), open('../data/cache/deep_data_oe{}_es{}_dt{}_f{}.pkl'.format(config.outer_embed, config.EMBED_SIZE, config.data_type, config.main_feature), 'wb'))


def deep_data_process():
    deep_data_cache()
    (train, train_y, test, char_w2v_embed, word_w2v_embed) = pickle.load(open('../data/cache/deep_data_oe{}_es{}_dt{}_f{}.pkl'.format(config.outer_embed, config.EMBED_SIZE, config.data_type, config.main_feature), 'rb'))
    config.char_embedding = char_w2v_embed
    config.word_embedding = word_w2v_embed

    model = config.model[args.model](config=config, n_folds=5)
    if config.data_type == 0:
        model.single_train_predict(train, train_y, test, option=config.option)
    elif config.data_type == 1:
        model.single_train_predict(train, train_y, test, option=config.option)

    elif config.data_type == 2:
        model.multi_train_predict(train, train_y, test, option=config.option)
    elif config.data_type == 3:
        model.four_classify_train_predict(train, train_y, test, option=config.option)
        # # model.multi_train_predict(train, train_y, test, option=config.option)
    # elif config.data_type == 4:
        # model.single_train_predict(train, train_y, test, option=config.option)
    # elif config.data_type == 5:
        # model.multi_train_predict(train, train_y, test, option=config.option)

    else:
        exit('错误数据类别')


def static_data_prepare():
    model_name = config.model_name
    if not model_name:
        model_name = "model_dict.pkl"
    logger.info('start load data')
    train_df = pd.read_csv(config.TRAIN_MULTI_X)
    test_df = pd.read_csv(config.TEST_X)
    if model_name in 'svc':
        content_word = pd.concat((train_df['word'], test_df['word']))
        content_char = pd.concat((train_df['char'], test_df['char']))
        word_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 5), min_df=5, norm='l2')
        char_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 5), min_df=1, norm='l2')

        ha = HashingVectorizer(ngram_range=(1, 1), lowercase=False)
        discuss_ha = ha.fit_transform(content_word)

        logger.info('start word feature extraction')
        word_feature = word_vectorizer.fit_transform(content_word)
        logger.info("complete word feature extraction models")
        logger.info("vocab len: %d" % len(word_vectorizer.vocabulary_.keys()))

        logger.info('start char feature extraction')
        char_feature = char_vectorizer.fit_transform(content_char)
        logger.info("complete char feature extraction models")
        logger.info("vocab len: %d" % len(char_vectorizer.vocabulary_.keys()))

        train_feature = hstack([word_feature[:len(train_df)], char_feature[:len(train_df)]]).tocsr()
        test_feature = hstack([word_feature[len(train_df):], char_feature[len(train_df):]]).tocsr()

        train_feature = hstack((word_feature[:len(train_df)], discuss_ha[:len(train_df)])).tocsr()
        test_feature = hstack((word_feature[len(train_df):], discuss_ha[len(train_df):])).tocsr()

        train_feature = word_feature[:len(train_df)]
        test_feature = word_feature[len(train_df):]

        logger.info("complete char feature extraction models")
        logger.info("train feature shape: {}".format(np.shape(train_feature)))
        logger.info("test feature shape: {}".format(np.shape(test_feature)))

        train_y = np.array(train_df.iloc[:, 6:].values)
    else:
        train_feature = np.asarray([train_df['word']]).T
        train_y = np.array(train_df.iloc[:, 6:].values)
        test_feature = np.asarray([test_df['word']]).T
    return train_feature, train_y, test_feature


def static_data_process():
    # model train
    train_x, train_y, test = static_data_prepare()
    model = config.model[args.model](config=config, n_folds=5)
    model.train_predict(train_x, train_y, test, option=config.option)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='6')
    parser.add_argument('--model', type=str, help='模型')
    parser.add_argument('--option', type=int, default=1, help='训练方式')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--data_type', type=int, default=3, help='问题模式, 0分单主题, 1分单情感, 2为十个四分类, 3为asp')
    parser.add_argument('--feature', default='word', type=str, help='选择word或者char作为特征')
    parser.add_argument('--es', default=300, type=int, help='embed size')
    parser.add_argument('--debug', default=False, action='store_true', help='debug只会跑一折')
    parser.add_argument('--oe', default=False, action='store_true', help='百度百科预训练词向量')
    parser.add_argument('--ml', default=False, action='store_true', help='是否使用传统模型')
    parser.add_argument('--car', default=False, action='store_true', help='是否用汽车之家数据训练的词向量')
    parser.add_argument('--balance', default=False, action='store_true', help='根据样例比修改loss权重')
    parser.add_argument('--bs', default=64, type=int, help='batch size')
    args = parser.parse_args()

    # 设置keras后台和gpu
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    config = Config()
    config.option = args.option
    config.outer_embed = args.oe
    config.n_epochs = args.epoch
    config.main_feature = args.feature
    config.model_name = args.model
    config.is_debug = args.debug
    config.BATCH_SIZE = args.bs
    config.gpu = args.gpu
    config.EMBED_SIZE = args.es
    config.data_type = args.data_type
    config.car = args.car
    config.balance = args.balance

    if config.model_name in ['svc', 'fasttext']:
        args.ml = True

    if args.ml:
        static_data_process()
    else:
        char_stoi = pickle.load(open(config.char_stoi_file, 'rb'))
        word_stoi = pickle.load(open(config.word_stoi_file, 'rb'))

        deep_data_process()

