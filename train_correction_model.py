import sys
from package.model import FewShot_NER
from package.utils import *

from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup
from collections import Counter
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
import matplotlib.pyplot as plt

base_path = 'berts/chinese-roberta-wwm-ext'
save_models_path = 'save_models/label_correction_model/'
result_path = 'result/'

label_file = 'label_file/label_rocling.json'
train_file = 'data/train/correction_data_file.json'
LSM_file = 'save_models/label_semantics_model/LSM.pth' # 記得在訓練好LSM後把這邊改成訓練好的模型檔案

mistake_file_name = 'label_correction'

# 如果use_X_data設定了True，對應的資料集就要分配一定比例的資料，否則會出錯
use_dev_data = True
use_test_data = False
# 資料集分割的比例，只須設定train和dev，剩下的會自動分配給test
train_ratio = 0.9
dev_ratio = 0.1

# 註：這邊要注意資料集的格式
# 因為train集是json檔
# 且句子和標籤分別存在key為'Sentence'和'Full True Labels'裡
# 所以才採用這種方式先讀入每筆json檔
# 再分別讀取key的值到變數裡
def load_data(paths, train_ratio, dev_ratio):
    arr = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            data = json.loads(f.read())
            for item in data:
                arr.append(item)

    train_tokens,train_labels, dev_tokens, dev_labels, test_tokens, test_labels = [], [], [], [], [], []
    total_samples = len(arr)

    # 計算劃分的數量
    train_samples = int(total_samples * train_ratio)
    dev_samples = int(total_samples * dev_ratio)
    test_samples = total_samples - train_samples - dev_samples

    # 根據劃分數量進行切割
    train_data = arr[:train_samples]
    dev_data = arr[train_samples:train_samples+dev_samples]
    test_data = arr[train_samples+dev_samples:]

    train_tokens = [data['Sentence'] for data in train_data]
    train_labels = [data['Full True Labels'] for data in train_data]

    dev_tokens = [data['Sentence'] for data in dev_data]
    dev_labels = [data['Full True Labels'] for data in dev_data]

    test_tokens = [data['Sentence'] for data in test_data]
    test_labels = [data['Full True Labels'] for data in test_data]

    print("--------------------------------")
    print("訓練集大小：", len(train_tokens))
    print("驗證集大小：", len(dev_tokens))
    print("測試集大小：", len(test_tokens))

    return train_tokens, train_labels, dev_tokens, dev_labels, test_tokens, test_labels

def print_all_labels_count(train_labels, dev_labels, test_labels):
    all_labels = train_labels + dev_labels + test_labels
    flat_train_labels = [label for sublist in all_labels for label in sublist]
    label_counts = Counter(flat_train_labels)
    print("-------------所有標籤數量-------------")
    for label in label_counts:
        print(label + ": " + str(label_counts[label]))
    print("--------------------------------")

# ------------------載入資料-------------------
train_tokens,train_labels, dev_tokens, dev_labels, test_tokens, test_labels = load_data([train_file], train_ratio, dev_ratio)
if use_dev_data and len(dev_tokens)==1:
    print("use_dev_data設為True，dev_ratio必須不為0")
    sys.exit()
if use_test_data and len(test_tokens)==1:
    print("use_test_data設為True，必須分配給test部分資料")
    sys.exit()

# ------------------計算各標籤的數量-------------------
print_all_labels_count(train_labels, dev_labels, test_labels)

# ------------------模型參數-------------------
max_len = 128
bs = 32
epochs = 5
end_lr = 10
num_iter = 100

# ------------------載入BERT模型與標籤資料-------------------
tokenizer = BertTokenizer.from_pretrained(base_path)
tag2id,id2tag,short_labels = trans2id(label_file)

# ------------------把文本轉換為BERT模型的輸入格式，並創建DataLoader-------------------
train_ids,train_token_type_ids,train_attention_masks,train_tags,train_lengths = gen_features(train_tokens,train_labels,tokenizer,tag2id,max_len)
train_ids = torch.tensor([item.cpu().detach().numpy() for item in train_ids]).squeeze()
train_tags = torch.tensor(train_tags)
train_masks = torch.tensor([item.cpu().detach().numpy() for item in train_attention_masks]).squeeze()
train_token_type_ids = torch.tensor([item.cpu().detach().numpy() for item in train_token_type_ids]).squeeze()
train_data = TensorDataset(train_ids, train_masks, train_token_type_ids,train_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

if use_dev_data:
    dev_ids,dev_token_type_ids,dev_attention_masks,dev_tags,dev_lengths = gen_features(dev_tokens,dev_labels,tokenizer,tag2id,max_len)
    dev_ids = torch.tensor([item.cpu().detach().numpy() for item in dev_ids]).squeeze()
    dev_tags = torch.tensor(dev_tags)
    dev_masks = torch.tensor([item.cpu().detach().numpy() for item in dev_attention_masks]).squeeze()
    dev_token_type_ids = torch.tensor([item.cpu().detach().numpy() for item in dev_token_type_ids]).squeeze()
    valid_data = TensorDataset(dev_ids, dev_masks,dev_token_type_ids,dev_tags)
    valid_sampler = RandomSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)

if use_test_data:
    test_ids,test_token_type_ids,test_attention_masks,test_tags,test_lengths = gen_features(test_tokens,test_labels,tokenizer,tag2id,max_len)
    test_ids = torch.tensor([item.cpu().detach().numpy() for item in test_ids]).squeeze()
    test_tags = torch.tensor(test_tags)
    test_masks = torch.tensor([item.cpu().detach().numpy() for item in test_attention_masks]).squeeze()
    test_token_type_ids = torch.tensor([item.cpu().detach().numpy() for item in test_token_type_ids]).squeeze()
    test_data = TensorDataset(test_ids, test_masks,test_token_type_ids, test_tags)
    test_dataloader = DataLoader(test_data, batch_size=bs)

# ------------------模型設定-------------------
precesion_scores,recall_scores,f1_scores = [],[],[]

fewshot = FewShot_NER(base_path,tag2id,bs,label_file)
loss_function=CrossEntropyLoss()
optimizer = torch.optim.Adam(fewshot.parameters(), lr = 1e-5)
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fewshot.to(device)

# ------------------訓練-------------------
fewshot.load_state_dict(torch.load(LSM_file))

tra_loss,steps = 0.0,0
max_grad_norm = 10
F1_score = 0

scaler = torch.cuda.amp.GradScaler()
for i in range(epochs):
    fewshot.train()

    for step ,batch in enumerate(train_dataloader):
        input_ids,masks,token_type_ids,labels= (i.to(device) for i in batch)

        matrix_embeddings,label_indexs = fewshot({"input_ids":input_ids,"attention_mask":masks,"token_type_ids":token_type_ids})
        loss = loss_function(matrix_embeddings.view(-1, len(tag2id)),labels.view(-1)) # CrossEntropyLoss
        optimizer.zero_grad()
        loss.backward()

        tra_loss += loss
        steps += 1

        torch.nn.utils.clip_grad_norm_(parameters=fewshot.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        scheduler.step()

        if step % 30 == 0:
            print("epoch :{},step :{} ,Train loss: {}".format(i,step,tra_loss/steps))

    print("Training Loss of epoch {}:{}".format(i,tra_loss / steps))

    if use_dev_data:
        fewshot.eval()
        dev_loss = 0.0
        predictions , true_labels = [], []

        for batch in valid_dataloader:
            input_ids,masks,token_type_ids,labels= (i.to(device) for i in batch)

            with torch.no_grad():
                matrix_embeddings,output_indexs = fewshot({"input_ids":input_ids,"attention_mask":masks,"token_type_ids":token_type_ids},flag = False)

            predictions.extend(output_indexs.detach().cpu().numpy().tolist())
            true_labels.extend(labels.to('cpu').numpy().tolist())

        f1, precision, recall = measure(predictions,true_labels,dev_lengths,id2tag)

        precesion_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        print('epoch {} dev : Acc : {},Recall : {},F1 :{}'.format(i,precision,recall,f1))

        if F1_score < f1:
            F1_score = f1
            torch.save(fewshot.state_dict(), save_models_path+'model_{}_{}.pth'.format(i,F1_score))
    else:
        torch.save(fewshot.state_dict(), save_models_path+'model_{}.pth'.format(i))

# ------------------跑test集-------------------
if use_test_data:
    fewshot.eval()
    test_pre,test_true = [],[]
    for batch in test_dataloader:

        input_ids,masks,token_type_ids,labels= (i.to(device) for i in batch)

        with torch.no_grad():
            matrix_embeddings,output_indexs = fewshot({"input_ids":input_ids,"attention_mask":masks,"token_type_ids":token_type_ids},flag = False)

        test_pre.extend(output_indexs.detach().cpu().numpy().tolist())
        test_true.extend(labels.to('cpu').numpy().tolist())

    # ------------------test集數據-------------------
    test_f1, test_precision, test_recall, test_mistake, test_mistake_num = test_measure(test_pre,test_true,test_tokens,test_lengths,id2tag)
    print('Test Acc : {},Recall : {},F1 :{}'.format(test_precision,test_recall,test_f1))
    precesion_scores.append(test_precision)
    recall_scores.append(test_recall)
    f1_scores.append(test_f1)

    # ------------------輸出錯誤結果與correction model需要的資料-------------------
    print(len(test_mistake_num))
    output_mistake_result(test_precision,test_recall,test_f1,test_mistake, test_mistake_num, result_path, mistake_file_name)

# ------------------繪製結果折線圖-------------------
if use_dev_data:
    custom_labels = ['epoch {}'.format(i) for i in range(epochs)]

    if use_test_data:
        custom_labels.append('test')

    num_labels = len(custom_labels)  # 獲取 custom_labels 的大小

    plt.plot(range(1, num_labels + 1), precesion_scores, label='PRECISION')
    plt.plot(range(1, num_labels + 1), recall_scores, label='RECALL')
    plt.plot(range(1, num_labels + 1), f1_scores, label='F1')
    plt.legend()
    plt.xticks(range(1, num_labels + 1), custom_labels)
    plt.show()