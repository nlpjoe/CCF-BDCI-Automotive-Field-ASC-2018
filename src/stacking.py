import pickle
import glob
import pandas as pd
from config import Config
from keras.utils import np_utils
from keras.layers import *
from model.snapshot import SnapshotCallbackBuilder
from model.my_callbacks import JZTrainCategory
from keras.models import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score

from model.model_basic import BasicModel
import numpy as np
import os


def get_f1_score(x, y, verbose=False):
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


def data_prepare():
    train_df = pd.read_csv(config.TRAIN_X)

    if config.data_type == 0:
        train_y = {}
        sub_list = pickle.load(open('../data/sub_list.pkl', 'rb'))
        for sub in sub_list:
            train_y_val = train_df[sub].values
            train_y[sub] = np_utils.to_categorical(train_y_val, num_classes=config.n_class)
    elif config.data_type == 1:
        train_y = train_df['c_numerical'].values
        train_y = np_utils.to_categorical(train_y, num_classes=config.n_class)
    elif config.data_type == 2:
        train_y = {}
        train_y['subject'] = train_df['sub_numerical'].values
        train_y['subject'] = np_utils.to_categorical(train_y['subject'], num_classes=10)
        train_y['sentiment_value'] = train_df['sentiment_value'].values
        train_y['sentiment_value'] = np_utils.to_categorical(train_y['sentiment_value'], num_classes=3)

    elif config.data_type == 3:
        # 主要融合这个
        train_y = train_df.iloc[:, 6:].values
        targets = train_y.reshape(-1)
        one_hot_targets = np.eye(config.n_classes)[targets]
        train_y = one_hot_targets.reshape(-1, 10, config.n_classes)
    elif config.data_type == 4:
        train_y = (train_df['sentiment_value']+1).values
        train_y = np_utils.to_categorical(train_y, num_classes=config.n_class)
    elif config.data_type == 5:
        train_y = train_df.iloc[:, 4:].values

    else:
        exit('错误数据类别')

    # oof features
    filenames = glob.glob('../data/result-qiuqiu/*oof*')
    filenames.extend(glob.glob('../data/result-dt3-op1-embed300-debugFalse-distillation/*oof*'))
    filenames.extend(glob.glob('../data/11_11_result/*oof*'))
    # filenames.extend(glob.glob('../data/result-dt3-op1-embed300-debugFalse/*oof*'))
    # filenames.extend(glob.glob('../data/result-dt3-op1-embed300-debugFalse-enhance/*oof*'))

    # filenames = glob.glob('../data/result-stacking/*oof*'.format(args.data_type))
    # def filter(filename, f_value):
        # return float(filename.split('_')[-3][1:-4]) > f_value

    # filenames = [e for e in filenames if filter(e, args.f_value)]
    # filenames = glob.glob('../data/result-dt{}-op1-embed300-debugFalse-enhance/*oof*'.format(args.data_type))
    from pprint import pprint
    pprint(filenames)

    oof_filename = []
    test_filename = []
    for j, filename in enumerate(filenames):
        p_filename = filename.replace('_oof_', '_pre_')
        oof_filename.append(filename)
        test_filename.append(p_filename)

    oof_data = []
    test_data = []
    for i, (tra, tes) in enumerate(zip(oof_filename, test_filename)):

        oof_feature = pickle.load(open(tra, 'rb'))
        print(tra, oof_feature.shape)
        oof_data.append(oof_feature)

        oof_feature = pickle.load(open(tes, 'rb'))
        print(tes, oof_feature.shape)
        test_data.append(oof_feature)

    train_x = np.concatenate(oof_data, axis=-1)
    test_x = np.concatenate(test_data, axis=-1)
    #  train_x = np.reshape(train_x, [-1, train_x.shape[-1]])
    #  test_x = np.reshape(test_x, [-1, test_x.shape[-1]])
    print('train_x shape: ', train_x.shape)
    print('train_y shape: ', train_y.shape)
    print('test_x shape: ', test_x.shape)

    return train_x, train_y, test_x


def get_model(train_x):
    input_x = Input(shape=(train_x.shape[-2], train_x.shape[-1]), name='input')
    x = Dense(256, activation='relu')(input_x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4, activation="softmax")(x)
    res_model = Model(inputs=[input_x], outputs=x)
    return res_model


# 第一次stacking
def stacking_first(train, train_y, test):
    savepath = './stack_op{}_dt{}_f_value{}/'.format(args.option, args.data_type, args.f_value)
    os.makedirs(savepath, exist_ok=True)

    count_kflod = 0
    num_folds = 5
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=10)
    predict = np.zeros((test.shape[0], 10, 4))
    oof_predict = np.zeros((train.shape[0], 10, 4))
    scores = []

    for i, (train_index, test_index) in enumerate(kf.split(train)):
        print('第{}折'.format(i))

        kfold_X_train = {}
        kfold_X_valid = {}

        y_train, y_test = train_y[train_index], train_y[test_index]

        kfold_X_train, kfold_X_valid = train[train_index], train[test_index]

        model_prefix = savepath + 'DNN' + str(count_kflod)
        if not os.path.exists(model_prefix):
            os.mkdir(model_prefix)

        M = 3  # number of snapshots
        alpha_zero = 1e-3  # initial learning rate
        snap_epoch = 30

        snapshot = SnapshotCallbackBuilder(snap_epoch, M, alpha_zero)
        # M = 1  # number of snapshots
        # snap_epoch = 16
        # jz_schedule = JZTrainCategory(model_prefix, snap_epoch, M, save_weights_only=True,  monitor='val_loss', factor=0.7, patience=1)

        res_model = get_model(train)
        res_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        res_model.summary()

        # res_model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCH, verbose=1,  class_weight=class_weight)
        res_model.fit(kfold_X_train, y_train, batch_size=BATCH_SIZE, epochs=snap_epoch, verbose=1,
                      validation_data=(kfold_X_valid, y_test),
                      callbacks=snapshot.get_callbacks(model_save_place=model_prefix))

        evaluations = []
        for i in os.listdir(model_prefix):
            if '.h5' in i:
                evaluations.append(i)

        test_pred_ = np.zeros((test.shape[0], 10, 4))
        oof_pred_ = np.zeros((len(kfold_X_valid), 10, 4))
        for run, i in enumerate(evaluations):
            print('loading from {}'.format(os.path.join(model_prefix, i)))
            res_model.load_weights(os.path.join(model_prefix, i))
            test_pred_ += res_model.predict(test, verbose=1, batch_size=256) / len(evaluations)
            oof_pred_ += res_model.predict(kfold_X_valid, batch_size=256) / len(evaluations)

        predict += test_pred_ / num_folds
        oof_predict[test_index] = oof_pred_

        f1 = get_f1_score(np.argmax(oof_pred_, -1), np.argmax(y_test, -1), verbose=True)
        print(i, ' kflod cv f1 : ', str(f1))
        count_kflod += 1
        scores.append(f1)
    print('f1 {} -> {}'.format(scores, np.mean(scores)))
    return predict, oof_predict, np.mean(scores)

import lightgbm as lgb
def stacking_lightgbm(train, train_y, test):
    train_y = np.argmax(train_y, 1)
    count_kflod = 0
    num_folds = 5
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=10)
    predict = np.zeros((test.shape[0], config.n_class))
    oof_predict = np.zeros((train.shape[0], config.n_class))
    scores = []
    f1s = []

    params = {'objective': 'multiclass',
                            'bagging_seed': 10,
                            'boosting_type': 'gbdt',
                            'feature_fraction': 0.9,
                            'feature_fraction_seed': 10,
                            'lambda_l1': 0.5,
                            'lambda_l2': 0.5,
                            'learning_rate': 0.01,
                            'metric': 'multi_logloss',
                            'min_child_weight': 1,
                            # 'min_split_gain': 0,
                            'device': 'gpu',
                            'gpu_platform_id': 0,
                            'gpu_device_id': config.gpu,
                            'min_sum_hessian_in_leaf': 0.1,
                            'num_leaves': 64,
                            'num_thread': -1,
                            'num_class': config.n_class,
                            'verbose': 1}

    for train_index, test_index in kf.split(train):

        y_train, y_test = train_y[train_index], train_y[test_index]
        kfold_X_train, kfold_X_valid = train[train_index], train[test_index]

        d_train = lgb.Dataset(kfold_X_train, label=y_train)
        d_watch = lgb.Dataset(kfold_X_valid, label=y_test)

        best = lgb.train(params, d_train, num_boost_round=100, verbose_eval=5,
                         valid_sets=d_watch,
                         early_stopping_rounds=6)

        preds1 = best.predict(test)
        preds2 = best.predict(kfold_X_valid)

        predict += preds1 / num_folds
        # oof_predict[test_index] = preds2

        accuracy = mb.cal_acc(preds2, y_test)
        f1 = mb.cal_f_alpha(preds2, y_test, n_out=config.n_class)

        print('the kflod cv is : ', str(accuracy))
        print('the kflod f1 is : ', str(f1))
        count_kflod += 1
        scores.append(accuracy)
        f1s.append(f1)
    print('total scores is ', np.mean(scores))
    print('total f1 is ', np.mean(f1s))
    #  return predict, np.mean(scores)
    return predict


from sklearn.linear_model import LogisticRegression
def stacking_lr(train, train_y, test):
    train_y = np.argmax(train_y, 1)
    count_kflod = 0
    num_folds = 6
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=10)
    predict = np.zeros((test.shape[0], config.n_class))
    scores = []
    f1s = []
    for train_index, test_index in kf.split(train):

        y_train, y_test = train_y[train_index], train_y[test_index]
        kfold_X_train, kfold_X_valid = train[train_index], train[test_index]

        print('拟合数据')
        best = LogisticRegression(C=4, dual=True)
        best.fit(kfold_X_train, y_train)

        print('预测结果')
        preds1 = best.predict_proba(test)
        preds2 = best.predict_proba(kfold_X_valid)

        predict += preds1 / num_folds
        accuracy = mb.cal_acc(preds2, y_test)
        f1 = mb.cal_f_alpha(preds2, y_test, n_out=config.n_class)

        print('the kflod cv is : ', str(accuracy))
        print('the kflod f1 is : ', str(f1))
        count_kflod += 1
        scores.append(accuracy)
        f1s.append(f1)
    print('total scores is ', np.mean(scores))
    print('total f1 is ', np.mean(f1s))
    #  return predict, np.mean(scores)
    return predict

from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV

def stacking_svm(train, train_y, test):
    train_y = np.argmax(train_y, 1)
    count_kflod = 0
    num_folds = 6
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=10)
    predict = np.zeros((test.shape[0], config.n_class))
    scores = []
    f1s = []
    for train_index, test_index in kf.split(train):

        y_train, y_test = train_y[train_index], train_y[test_index]
        kfold_X_train, kfold_X_valid = train[train_index], train[test_index]

        print('拟合数据')
        best = svm.LinearSVC()
        best = CalibratedClassifierCV(best)
        best.fit(kfold_X_train, y_train)

        print('预测结果')
        preds1 = best.predict_proba(test)
        preds2 = best.predict_proba(kfold_X_valid)

        predict += preds1 / num_folds
        accuracy = mb.cal_acc(preds2, y_test)
        f1 = mb.cal_f_alpha(preds2, y_test, n_out=config.n_class)

        print('the kflod cv is : ', str(accuracy))
        print('the kflod f1 is : ', str(f1))
        count_kflod += 1
        scores.append(accuracy)
        f1s.append(f1)
    print('total scores is ', np.mean(scores))
    print('total f1 is ', np.mean(f1s))
    #  return predict, np.mean(scores)
    return predict


# 使用pseudo-labeling做第二次stacking
def stacking_pseudo(train, train_y, test, results):
    answer = np.reshape(np.argmax(results, axis=-1), [-1])
    answer = np.reshape(np.eye(4)[answer], [-1, 10, 4])

    train_y = np.concatenate([train_y, answer], axis=0)
    train = np.concatenate([train, test], axis=0)

    savepath = './pesudo_{}_dt{}/'.format(args.option, args.data_type)
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    count_kflod = 0
    num_folds = 5
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=10)
    predict = np.zeros((test.shape[0], 10, 4))
    oof_predict = np.zeros((train.shape[0], 10, 4))
    scores = []

    for i, (train_index, test_index) in enumerate(kf.split(train)):
        print('第{}折'.format(i))

        kfold_X_train = {}
        kfold_X_valid = {}

        y_train, y_test = train_y[train_index], train_y[test_index]

        kfold_X_train, kfold_X_valid = train[train_index], train[test_index]

        model_prefix = savepath + 'DNN' + str(count_kflod)
        if not os.path.exists(model_prefix):
            os.mkdir(model_prefix)

        M = 3  # number of snapshots
        alpha_zero = 1e-3  # initial learning rate
        snap_epoch = 30

        snapshot = SnapshotCallbackBuilder(snap_epoch, M, alpha_zero)
        # M = 1  # number of snapshots
        # snap_epoch = 16
        # jz_schedule = JZTrainCategory(model_prefix, snap_epoch, M, save_weights_only=True,  monitor='val_loss', factor=0.7, patience=1)

        res_model = get_model(train)
        res_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        res_model.summary()

        # res_model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCH, verbose=1,  class_weight=class_weight)
        res_model.fit(kfold_X_train, y_train, batch_size=BATCH_SIZE, epochs=snap_epoch, verbose=1,
                      validation_data=(kfold_X_valid, y_test),
                      callbacks=snapshot.get_callbacks(model_save_place=model_prefix))

        evaluations = []
        for i in os.listdir(model_prefix):
            if '.h5' in i:
                evaluations.append(i)

        test_pred_ = np.zeros((test.shape[0], 10, 4))
        oof_pred_ = np.zeros((len(kfold_X_valid), 10, 4))
        for run, i in enumerate(evaluations):
            print('loading from {}'.format(os.path.join(model_prefix, i)))
            res_model.load_weights(os.path.join(model_prefix, i))
            test_pred_ += res_model.predict(test, verbose=1, batch_size=256) / len(evaluations)
            oof_pred_ += res_model.predict(kfold_X_valid, batch_size=256) / len(evaluations)

        predict += test_pred_ / num_folds
        oof_predict[test_index] = oof_pred_

        f1 = get_f1_score(np.argmax(oof_pred_, -1), np.argmax(y_test, -1), verbose=True)
        print(i, ' kflod cv f1 : ', str(f1))
        count_kflod += 1
        scores.append(f1)
    print('f1 {} -> {}'.format(scores, np.mean(scores)))
    return predict, np.mean(scores)

def save_result(predict, prefix):
    os.makedirs('../data/result', exist_ok=True)
    with open('../data/result/{}.pkl'.format(prefix), 'wb') as f:
        pickle.dump(predict, f)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='6')
    parser.add_argument('--model', type=str, help='模型')
    parser.add_argument('--option', type=int, default=1, help='训练方式')
    parser.add_argument('--data_type', type=int, default=1, help='问题模式, 0为4分类, 1为单分类, 2为先分主题再分情感')
    parser.add_argument('--feature', default='word', type=str, help='选择word或者char作为特征')
    parser.add_argument('--es', default=200, type=int, help='embed size')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--bs', default=256, type=int, help='batch size')
    parser.add_argument('--f_value', default=0.0, type=float)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth=True
    set_session(tf.Session(config=tf_config))

    mb = BasicModel()
    config = Config()
    config.gpu = args.gpu
    config.data_type = args.data_type
    BATCH_SIZE = args.bs

    #  cv_stacking()

    # normal stacking
    train, train_y, test = data_prepare()

    predicts, oof_predicts, score = stacking_first(train, train_y, test)
    save_result(predicts, prefix=str(score))
    # save_result(oof_predicts, prefix='oof')

    # predicts = stacking_lightgbm(train, train_y, test)
    # save_result(predicts[:10000], prefix='stacking_lgb_first_op{}_{}_{}'.format(args.option, args.data_type, args.f_value))

    # predicts = stacking_lr(train, train_y, test)
    # save_result(predicts[:10000], prefix='stacking_lr_first_op{}_{}_{}'.format(args.option, args.data_type, args.f_value))

    # predicts = stacking_svm(train, train_y, test)
    # save_result(predicts[:10000], prefix='stacking_svm_first_op{}_{}_{}'.format(args.option, args.data_type, args.f_value))

    # 假标签
    predicts, score = stacking_pseudo(train, train_y, test, predicts)
    save_result(predicts, prefix=str(score))
