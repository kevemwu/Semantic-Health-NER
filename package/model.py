import json
import torch
import torch.nn as nn
from transformers import BertTokenizer
from transformers import BertModel

class FewShot_NER(nn.Module):
    def __init__(self,base_model_path,tag2id,batch_size,tag_file):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.token_encoder = BertModel.from_pretrained(base_model_path).to(self.device)
        self.label_encoder = BertModel.from_pretrained(base_model_path).to(self.device)

        self.label_context = self.__read_file(tag_file)
        self.index_context = {
            "B":"開始詞",
            "I":"中間詞",
        }
        self.tokenizer = BertTokenizer.from_pretrained(base_model_path)

        self.batch_size = batch_size
        self.tag2id = tag2id


    def __read_file(self,file):
        with open(file,'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def build_label_representation(self,tag2id):
        labels = []
        for k,v in tag2id.items():
            if k.split('-')[-1] != 'O':
                idx,label = k.split('-')[0],k.split('-')[-1]
                label = self.label_context[label]
                labels.append(label+self.index_context[idx])
            else:
                labels.append("其他類別詞")
        '''
        mutul(a,b) a和b維度是否一致的問題
        A.shape =（b,m,n)；B.shape = (b,n,k)
        torch.matmul(A,B) 結果shape為(b,m,k)
        '''

        tag_max_len = max([len(l) for l in labels])
        tag_embeddings = []
        for label in labels:
            input_ids = self.tokenizer.encode_plus(label,return_tensors='pt',padding='max_length',max_length=tag_max_len)
            outputs = self.label_encoder(input_ids=input_ids['input_ids'].to(self.device),
                                         token_type_ids=input_ids['token_type_ids'].to(self.device),attention_mask = input_ids['attention_mask'].to(self.device))
            pooler_output = outputs.pooler_output
            tag_embeddings.append(pooler_output)
        label_embeddings = torch.stack(tag_embeddings,dim=0)
        label_embeddings = label_embeddings.squeeze(1)
        return label_embeddings

    def forward(self,inputs,flag = True):
        if flag:
            label_representation = self.build_label_representation(self.tag2id).to(self.device)
            # self.label_representation = label_representation.detach()
            self.label_representation = label_representation
        else:
            label_representation = self.label_representation
        outputs = self.token_encoder(input_ids=inputs['input_ids'],
                                     token_type_ids=inputs['token_type_ids'],attention_mask = inputs['attention_mask'])
        token_embeddings = outputs.last_hidden_state
        tag_lens,hidden_size = self.label_representation.shape
        current_batch_size  = token_embeddings.shape[0]
        label_embedding = self.label_representation.expand(current_batch_size,tag_lens,hidden_size)
        label_embeddings = label_embedding.transpose(2,1)
        matrix_embeddings = torch.matmul(token_embeddings,label_embeddings)
        softmax_embedding= nn.Softmax(dim=-1)(matrix_embeddings)
        label_indexs = torch.argmax(softmax_embedding,dim=-1)
        return matrix_embeddings,label_indexs