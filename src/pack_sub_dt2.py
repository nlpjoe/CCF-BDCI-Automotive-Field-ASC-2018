import os
import pickle
import pandas as pd
import glob
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score

test_df = pd.read_csv('../data/csvs/test_public.csv')
train_df = pd.read_csv('../data/csvs/train_multi.csv')
true_labels = train_df.iloc[:, 6:].values

submit_df = pd.DataFrame(columns=['content_id', 'subject', 'sentiment_value', 'sentiment_word'])
train_oof_df = pd.DataFrame(columns=['content_id', 'subject', 'sentiment_value', 'sentiment_word'])
submit_df['content_id'] = test_df['content_id']
train_oof_df['content_id'] = train_df['content_id']



pre_path = '../data/result/0.807*'
pre_filenames = glob.glob(pre_path)
train_oof_filenames = glob.glob(pre_path.replace('pre', 'oof'))

pre = np.argmax(pickle.load(open(pre_filenames[0], 'rb')), 2)
train_oof_pred = np.argmax(pickle.load(open(train_oof_filenames[0], 'rb')), 2)

print(pre_filenames)
label_itos = [s.split('_')[1] for s in pickle.load(open('../data/sub_list.pkl', 'rb'))]
n_none = 0
n_mul_label = {}

f1s = []

content_ids = []
subjects = []
sentiment_values = []
lost_ids = []

for idx, c_id in enumerate(test_df['content_id']):
    n_label = np.sum(pre[idx] > 0)
    if not n_label:
        n_none += 1
        lost_ids.append(c_id)
    else:
        n_mul_label[n_label] = n_mul_label.get(n_label, 0) + 1
    labels = list(np.where(pre[idx]>0)[0])
    for l in labels:
        content_ids.append(c_id)
        subjects.append(label_itos[l])
        sentiment_values.append(pre[idx][l]-2)

soft_df = pd.read_csv('../data/submit/676.csv')
lost_df = soft_df[soft_df['content_id'].isin(lost_ids)]
submit_df = pd.DataFrame({'content_id': content_ids + list(lost_df['content_id']),
                          'subject': subjects + list(lost_df['subject']),
                          'sentiment_value': sentiment_values + list(lost_df['sentiment_value']),
                          # 'subject': subjects + ['']*len(lost_ids),
                          # 'sentiment_value': sentiment_values + ['']*len(lost_ids),
                          'sentiment_word': ['']*(len(lost_df)+len(subjects))})

print('n_none:', n_none)
print('n_pad:', len(lost_df))
os.makedirs('../data/submit', exist_ok=True)
submit_df.to_csv('../data/submit/dt3_stacking_submission.csv', index=None)

#  for i in range(train_oof_pred.shape[1]):
    #  pre_label = train_oof_pred[:, i]
    #  true_label = true_labels[:, i]
    #  f1 = f1_score(true_label, pre_label, average='macro')
    #  f1s.append(f1)

#  f1 = np.mean(f1s)
#  print('f1s->', f1s)
#  print('mean f1', f1)
#  print('n_none:', n_none)
#  os.makedirs('../data/submit', exist_ok=True)

#  submit_df.to_csv('../data/submit/dt2_{}_submission.csv'.format(f1), index=None)


