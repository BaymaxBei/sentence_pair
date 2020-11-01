import transformers
import torch
from transformers import BertTokenizer, BertModel, BertConfig, BartForSequenceClassification


# model = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext')
# torch.save(model, 'model/RoBERTa/chinese-roberta-wwm-ext.bin')

# RobertaModel = BertModel.from_pretrained('model/RoBERTa/chinese-roberta-wwm-ext.bin')

class RobertaModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.roberta = transformers.BertModel(config)
        self.fc = torch.nn.Linear(config.hidden_size, 1)

    def forward(self, hidden_states, attention_mask):
        output = self.roberta(hidden_states, attention_mask)
        output = self.fc(output[1])
        return output
        

