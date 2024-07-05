import torch
from transformers import BertModel
import torch.nn as nn
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AdamW
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class BERTRegression(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased'):
        super(BERTRegression, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        score = self.regressor(cls_output)
        return score

class QADataset(Dataset):
    def __init__(self, dataframe):

        self.dataframe = dataframe
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        question = row['Question']
        answer = row['Answer']
        score = row['Score']
        inputs = tokenizer(question, answer, return_tensors='pt', padding=True, truncation=True)
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        return input_ids, attention_mask, torch.tensor(score, dtype=torch.float)

def collate_fn(batch):
    input_ids = [item[0] for item in batch]
    attention_masks = [item[1] for item in batch]
    scores = torch.tensor([item[2] for item in batch], dtype=torch.float)

    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    padded_attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)

    return padded_input_ids, padded_attention_masks, scores

if __name__ == "__main__":
    # Example instantiation and forward pass (for testing purposes)
    model = BERTRegression()
    print(model)
