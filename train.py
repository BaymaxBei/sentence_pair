__author__='zhangbei'
__date__='2020/10/29'

import os
import csv
import tqdm
import random
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
import torch.nn.functional as F
import tokenizers
import transformers
from transformers import BertConfig, BertForSequenceClassification
from model_bert import RobertaModel


random.seed(8)


def read_data(mode='train'):
    file_name=f'./data/process_data/train.{mode}.tsv'
    queries, replies,labels=[],[],[]
    texts=[]
    with open(file_name,encoding='utf-8') as f:
        f_csv=csv.reader(f, delimiter='\t')
        for line in f_csv:
            texts.append(line)
    for text in tqdm.tqdm(texts,desc='loading '+file_name):
        queries.append(text[2])
        replies.append(text[3])
        labels.append(int(text[4]))
    return queries,replies,labels


class Dataset(torch.utils.data.Dataset):
    def __init__(self,mode='train'):
        self.tokenizer=tokenizers.BertWordPieceTokenizer("./model/RoBERTa/vocab.txt")
        self.tokenizer.enable_padding(length=64)
        self.tokenizer.enable_truncation(max_length=64)

        if mode=='train':
            self.queries, self.replies, self.labels=read_data(mode='train')
        else:
            self.queries, self.replies, self.labels=read_data(mode='dev')

    def __getitem__(self,index):
        token=self.tokenizer.encode(self.queries[index], self.replies[index])
        token.label=self.labels[index]

        return [torch.tensor(i) for i in (token.ids,token.label,token.attention_mask)]

    def __len__(self):
        return len(self.queries)


class CLASSIFIER:
    def __init__(self, config, load_pretrained=True):
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = RobertaModel(config).to(self.device)
        # self.model = BertForSequenceClassification(config).to(self.device)
        if load_pretrained:
            roberta_state=torch.load('model/RoBERTa/chinese-roberta-wwm-ext.bin', map_location=self.device).state_dict()
            self.model.roberta.load_state_dict(roberta_state)

    def val(self,loader_test):
        self.model.eval()
        y = np.array([])
        pred = np.array([])
        probs = np.array([])
        loss = 0
        with torch.no_grad():
            for step, (batch_x,batch_y,batch_mask) in enumerate(loader_test):
                y = np.append(y, batch_y)
                batch_max_length=batch_mask.sum(1).max().item()
                batch_x=batch_x[:,:batch_max_length].to(self.device)
                batch_mask=batch_mask[:,:batch_max_length].to(self.device)

                batch_output = self.model(batch_x,batch_mask.byte()).cpu()
                loss += F.binary_cross_entropy_with_logits(batch_output, batch_y).item()
                batch_probs = batch_output.sigmoid().numpy()
                batch_pred = [1 if i>0.5 else 0 for i in batch_probs]
                pred = np.append(pred, batch_pred)
                probs = np.append(probs, batch_probs)
        loss /= step+1
        self.model.train()
        precision = precision_score(y_true=y, y_pred=pred)
        recall = recall_score(y_true=y, y_pred=pred)
        f1 = f1_score(y_true=y, y_pred=pred)
        return precision,recall,f1,loss


    def predict(self, check_point):
        loader_test=torch.utils.data.DataLoader(dataset=Dataset('train'),batch_size=64,shuffle=False,num_workers=0)
        pth=torch.load(check_point,map_location='cpu')
        self.model.load_state_dict(pth['weights'])
        self.model.eval()
        y = np.array([])
        pred = np.array([])
        probs = np.array([])
        with torch.no_grad():
            for step, (batch_x,batch_y,batch_mask) in enumerate(loader_test):
                y = np.append(y, batch_y.numpy())
                batch_max_length=batch_mask.sum(1).max().item()
                batch_x=batch_x[:,:batch_max_length].to(self.device)
                batch_mask=batch_mask[:,:batch_max_length].to(self.device)

                batch_output = self.model(batch_x,batch_mask.byte()).cpu()
                batch_probs = batch_output.sigmoid().numpy()
                batch_pred = [1 if i>0.5 else 0 for i in batch_probs]
                pred = np.append(pred, batch_pred)
                probs = np.append(probs, batch_probs)
        precision = precision_score(y_true=y, y_pred=pred)
        recall = recall_score(y_true=y, y_pred=pred)
        f1 = f1_score(y_true=y, y_pred=pred)

        print('precision: {}\nrecall: {}\nf1 score: {}'.format(precision, recall, f1))
        return pred, probs

    def train(self,check_point='',epochs=10,batch_size=16):
        loader_train=torch.utils.data.DataLoader(dataset=Dataset('train'),batch_size=batch_size,shuffle=True,num_workers=0)
        loader_test=torch.utils.data.DataLoader(dataset=Dataset('test'),batch_size=128,shuffle=False,num_workers=0)

        optimizer=torch.optim.AdamW(self.model.parameters(),lr=1e-4)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=10,eta_min=1e-8)
        
        if check_point !='':
            pth=torch.load(check_point,map_location='cpu')
            self.model.load_state_dict(pth['weights'])
            optimizer.load_state_dict(pth['optimizer'])

        model_save_names=['','','']
        f1_scores=[0,0,0]
        self.model.train()
        for epoch in range(epochs):
            loader_t=tqdm.tqdm(loader_train,desc='epoch:{}/{}'.format(epoch,epochs))
            for step,(batch_x,batch_y,batch_mask) in enumerate(loader_t):
                batch_max_length=batch_mask.sum(1).max().item()
                batch_x=batch_x[:,:batch_max_length].to(self.device)
                batch_y=batch_y.to(self.device)
                batch_mask=batch_mask[:,:batch_max_length].to(self.device)
                
                batch_pred=self.model(batch_x,batch_mask.byte())
                loss = F.binary_cross_entropy_with_logits(batch_pred, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loader_t.set_postfix(training="loss:{:.6f}".format(loss.item()))

                if step%100==0:
                    pth={'weights':self.model.state_dict(),'optimizer':optimizer.state_dict()}

                    # if torch.os.path.exists(model_save_names[0]): torch.os.remove(model_save_names[0])
                    # model_save_names.append('epoch{}_batch{}.pth'.format(epoch, step))
                    # torch.save(pth,model_save_names[-1])
                    # model_save_names=model_save_names[1:]

                    precision,recall,f1, loss=self.val(loader_test)
                    print('precision: {:.4f}\trecall: {:.4f}\tf1_score: {:.4f}\tloss: {:.4f}'.format(precision, recall, f1, loss))

                    if f1>0.5 and f1>f1_scores[-1]:
                        f1_scores.append(f1)
                        # if torch.os.path.exists('f1_score_{:.4f}.pth'.format(f1_scores[0])): torch.os.remove('f1_score_{:.4f}.pth'.format(f1_scores[0]))
                        torch.save(pth,'roberta_cls_checkpoint/epoch{}_batch{}_f1_score_{:.4f}.pth'.format(epoch, step, f1_scores[-1]))
                        f1_scores=f1_scores[1:]

            scheduler.step()


def predict_analysis(check_point, result_file):
    config_roberta = BertConfig.from_pretrained('model/RoBERTa/config.json')
    print(config_roberta.num_labels)
    classifier=CLASSIFIER(config=config_roberta, load_pretrained=True)
    pred, probs = classifier.predict(check_point)
    texts=[]
    with open('data/test.tsv',encoding='utf-8') as f:
        f_csv=csv.reader(f, delimiter='\t')
        for line in f_csv:
            texts.append(line)
    with open(result_file, 'w', encoding='utf-8') as writefile:
        for index, line in enumerate(texts):
            writefile.write('{}\t{}\t{}\t{}\n'.format(int(pred[index]), probs[index], line[0], line[1]))

if __name__ == "__main__":
    config_roberta = BertConfig.from_pretrained('model/RoBERTa/config.json')
    classifier=CLASSIFIER(config=config_roberta, load_pretrained=True)
    classifier.train()

    # for i in range(2, 6):
    #     print('------------- epoch{} ------------'.format(i))
    # filename = 'roberta_checkpoint'
    # for file in  os.listdir(filename):
    #     filepath = os.path.join(filename, file)
    #     print(filepath)
    #     predict_analysis(check_point=filepath, result_file='{}/train_{}_res.tsv'.format(filename, file.split('.')[0]))
