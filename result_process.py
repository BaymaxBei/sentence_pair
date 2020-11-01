import pandas as pd
import csv



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
    

if __name__ == '__main__':
    datafile = './data/process_data/test.tsv'
    # generate_result(predfile='roberta_cls_checkpoint/predict_result/test_epoch0_batch1000_f1_score_0.7356.pth_res.tsv', datafile=datafile, savefile='roberta_cls_checkpoint/test_result/result_test_epoch0_batch1000_f1_score_0.7356.pth_res.tsv')
    # generate_result(predfile='roberta_cls_checkpoint/predict_result/test_epoch1_batch500_f1_score_0.7559.pth_res.tsv', datafile=datafile, savefile='roberta_cls_checkpoint/test_result/result_test_epoch1_batch500_f1_score_0.7559.pth_res.tsv')

    
    # generate_result(predfile='roberta_cls_checkpoint/predict_result/test_epoch1_batch1000_f1_score_0.7803.pth_res.tsv', datafile=datafile, savefile='roberta_cls_checkpoint/test_result/result_test_epoch1_batch1000_f1_score_0.7803.pth_res.tsv')
    # generate_result(predfile='roberta_cls_checkpoint/predict_result/test_epoch2_batch500_f1_score_0.7831.pth_res.tsv', datafile=datafile, savefile='roberta_cls_checkpoint/test_result/result_test_epoch2_batch500_f1_score_0.7831.pth_res.tsv')
    # generate_result(predfile='roberta_cls_checkpoint/predict_result/test_epoch3_batch1000_f1_score_0.7969.pth_res.tsv', datafile=datafile, savefile='roberta_cls_checkpoint/test_result/result_test_epoch3_batch1000_f1_score_0.7969.pth_res.tsv')
    generate_result(predfile='roberta_cls_checkpoint/predict_result/test_epoch6_batch500_f1_score_0.7991.pth_res.tsv', datafile=datafile, savefile='roberta_cls_checkpoint/test_result/test_epoch6_batch500_f1_score_0.7991.pth_res.tsv')
