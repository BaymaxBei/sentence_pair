import csv
import pandas as pd
from sklearn.model_selection import train_test_split


def data_format(queryfile, replyfile, savefile, mode='train'):
    if mode=='train':
        encode = 'utf-8'
    else:
        encode = 'GBK'
    query_data = pd.read_csv(queryfile, encoding=encode, sep='\t', header=None, names=['query_id', 'query'])
    query_data.set_index('query_id', inplace=True)
    print(len(query_data))
    print(query_data[:5])
    if mode=='train':
        columns = ['query_id', 'reply_id', 'reply', 'label']
    else:
        columns = ['query_id', 'reply_id', 'reply']
    reply_data = pd.read_csv(replyfile, encoding=encode, sep='\t', header=None, names=columns)
    print(len(reply_data))
    print(reply_data[:5])
    reply_data['query'] = reply_data['query_id'].apply(lambda x:query_data.loc[x, 'query'])
    print(reply_data[:5])
    if mode=='train':
        save_columns = ['query_id', 'reply_id', 'query', 'reply', 'label']
    else:
        save_columns = ['query_id', 'reply_id', 'query', 'reply']
    reply_data.loc[:, save_columns].to_csv(savefile, sep='\t', index=False, header=False)


def split_data(datafile, trainfile, devfile):
    data_list = []
    with open(datafile, encoding='utf-8') as f:
        for line in f:
            data_list.append(line)
    train_data, dev_data = train_test_split(data_list, test_size=0.2, random_state=8)
    print(len(train_data), len(dev_data))
    with open(trainfile, 'w', encoding='utf-8') as trainf:
        for line in train_data:
            trainf.write(line)
    with open(devfile, 'w', encoding='utf-8') as devf:
        for line in dev_data:
            devf.write(line)
    

def data_analysis(datafile, savefile):
    data = pd.read_csv(datafile, sep='\t', header=None, names=['query_id', 'reply_id', 'query', 'reply', 'label'])
    print(data[:5])
    data_group = data.groupby(by=['label']).count()
    print(data_group)
    pd.DataFrame(data_group).to_csv(savefile)


def data_balance(datafile, savefile):
    pos_data = []
    neg_data = []
    with open(datafile, encoding='utf-8') as dataf:
        for line in dataf:
            data_i = line.strip('\n').split('\t')
            if data_i[-1]=='1':
                pos_data.append(line)
            else:
                neg_data.append(line)
    
    for i in range(2):
        pos_data.extend(pos_data)
    print(len(pos_data))
    print(len(neg_data))
    with open(savefile, 'w', encoding='utf-8') as f:
        for line in pos_data:
            f.write(line)
        for line in neg_data:
            f.write(line)



if __name__ == '__main__':
    # data_format(queryfile='data/train/train.query.tsv', replyfile='data/train/train.reply.tsv', savefile='data/process_data/train.tsv')
    # data_format(queryfile='data/test/test.query.tsv', replyfile='data/test/test.reply.tsv', savefile='data/process_data/test.tsv', mode='test')
    # split_data(datafile='data/process_data/train.tsv', trainfile='data/process_data/train.train.tsv', devfile='data/process_data/train.dev.tsv')
    # data_analysis(datafile='data/process_data/train.train.tsv', savefile='data/process_data/train.train_analysis.txt')
    data_balance(datafile='data/process_data/train.train.tsv', savefile='data/process_data/train.train_balance.tsv')