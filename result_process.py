import pandas as pd
import csv
import os
import numpy as np


def generate_result(predfile, datafile, savefile):
    pred_result = []
    data_list = []
    with open(predfile, encoding='utf-8') as f:
        f_csv = csv.reader(f, delimiter='\t')
        for line in f_csv:
            pred_result.append(line)
    with open(datafile, encoding='utf-8') as f:
        f_csv = csv.reader(f, delimiter='\t')
        for line in f_csv:
            data_list.append(line)
    assert len(pred_result) == len(data_list)
    with open(savefile, 'w', encoding='utf-8') as f:
        for i in range(len(pred_result)):
            f.write('{}\t{}\t{}\n'.format(data_list[i][0], data_list[i][1], pred_result[i][0]))
    

def get_vote_result(filepath, savefile):
    result_all = []
    for file in os.listdir(filepath):
        file = os.path.join(filepath, file)
        result = []
        with open(file, encoding='utf-8') as f:
            f_csv = csv.reader(f, delimiter='\t')
            for line in f_csv:
                result.append(int(line[-1]))
        result_all.append(result)
    result_np = np.array(result_all)
    print(result_np.shape)
    result_np = np.transpose(result_np)
    print(result_np.shape)

    result_pd = pd.DataFrame(data=result_np)
    print(result_pd)
    result_pd['label'] = result_pd.apply(lambda line: 1 if np.sum(line)>=3 else 0, axis=1)
    # result_pd.to_csv(savefile, index=False, header=False)
    label = result_pd['label'].tolist()
    with open(savefile, 'w', encoding='utf-8') as wf:
        with open('roberta_cls_checkpoint/test_result/test_epoch6_batch500_f1_score_0.7991.pth_res.tsv') as f:
            f_csv = csv.reader(f, delimiter='\t')
            for i, line in enumerate(f_csv):
                wf.write('{}\t{}\t{}\n'.format(line[0], line[1], label[i]))
    

if __name__ == '__main__':
    datafile = './data/process_data/test.tsv'
    # generate_result(predfile='roberta_cls_checkpoint/predict_result/test_epoch0_batch1000_f1_score_0.7356.pth_res.tsv', datafile=datafile, savefile='roberta_cls_checkpoint/test_result/result_test_epoch0_batch1000_f1_score_0.7356.pth_res.tsv')
    # generate_result(predfile='roberta_cls_checkpoint/predict_result/test_epoch1_batch500_f1_score_0.7559.pth_res.tsv', datafile=datafile, savefile='roberta_cls_checkpoint/test_result/result_test_epoch1_batch500_f1_score_0.7559.pth_res.tsv')

    
    # generate_result(predfile='roberta_cls_checkpoint/predict_result/test_epoch1_batch1000_f1_score_0.7803.pth_res.tsv', datafile=datafile, savefile='roberta_cls_checkpoint/test_result/result_test_epoch1_batch1000_f1_score_0.7803.pth_res.tsv')
    # generate_result(predfile='roberta_cls_checkpoint/predict_result/test_epoch2_batch500_f1_score_0.7831.pth_res.tsv', datafile=datafile, savefile='roberta_cls_checkpoint/test_result/result_test_epoch2_batch500_f1_score_0.7831.pth_res.tsv')
    # generate_result(predfile='roberta_cls_checkpoint/predict_result/test_epoch3_batch1000_f1_score_0.7969.pth_res.tsv', datafile=datafile, savefile='roberta_cls_checkpoint/test_result/result_test_epoch3_batch1000_f1_score_0.7969.pth_res.tsv')
    # generate_result(predfile='roberta_cls_checkpoint/predict_result/test_epoch6_batch500_f1_score_0.7991.pth_res.tsv', datafile=datafile, savefile='roberta_cls_checkpoint/test_result/test_epoch6_batch500_f1_score_0.7991.pth_res.tsv')

    get_vote_result(filepath='roberta_cls_checkpoint/test_result/', savefile='roberta_cls_checkpoint/test_result/result_test_vote.tsv')