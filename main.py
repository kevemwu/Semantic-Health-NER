from package.model import FewShot_NER
from package.utils import *

from transformers import BertTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import copy

base_path = 'berts/chinese-roberta-wwm-ext'
save_models_path = 'save_models/label_correction_model/'

label_file = 'label_file/label_rocling.json'
test_file = 'data/test/test.txt'
# test_file = 'data/test/test_answer.txt' # 這是有答案版的txt
LSM_file = 'save_models/label_semantics_model/LSM.pth' # 記得在訓練好LSM後把這邊改成訓練好的模型檔案
LCM_file = 'save_models/label_correction_model/LCM.pth' # 記得在訓練好LCM後把這邊改成訓練好的模型檔案

# 輸出結果的資料夾和檔名，默認輸出.txt，要改的話要去utils.py改函式內容
result_path = 'result/'
result_file = 'FT_three_stage'

is_answer = False # 設True的話會依test集中的答案計算數據，反之代表test集沒有答案，此次處理為會輸出預測結果

# ------------------載入資料-------------------
with open(test_file, 'r', encoding='utf-8') as file:
    content = file.read()
content = content.split("\n\n")

if is_answer:
    test_tokens,test_labels=[],[]

    for item in content:
        line = item.split('\n')
        text, label = '', []
        for tl in line:
            text=text+tl[:1]
            label.append(tl[2:])
        test_tokens.append(text)
        test_labels.append(label)

    test_tokens=test_tokens[:-1]
    test_labels=test_labels[:-1]
else:
    test_tokens=[]

    for item in content:
        item = item.replace('\n','')
        test_tokens.append(item)

# ------------------載入BERT模型與標籤資料-------------------
tokenizer = BertTokenizer.from_pretrained(base_path)
tag2id,id2tag,short_labels = trans2id(label_file)
bio_tags = [value[0] for value in id2tag.values()]

# ------------------模型參數-------------------
max_len = 512
bs = 32

# ------------------把文本轉換為BERT模型的輸入格式，並創建DataLoader-------------------
if is_answer:
    test_ids,test_token_type_ids,test_attention_masks,test_tags,test_lengths = gen_features(test_tokens,test_labels,tokenizer,tag2id,max_len)
    test_tags = torch.tensor(test_tags)
    test_ids = torch.tensor([item.cpu().detach().numpy() for item in test_ids]).squeeze()
    test_masks = torch.tensor([item.cpu().detach().numpy() for item in test_attention_masks]).squeeze()
    test_token_type_ids = torch.tensor([item.cpu().detach().numpy() for item in test_token_type_ids]).squeeze()
    test_data = TensorDataset(test_ids, test_masks,test_token_type_ids, test_tags)
    test_dataloader = DataLoader(test_data, batch_size=bs)
else:
    test_ids, test_token_type_ids, test_attention_masks, test_lengths = no_labels_gen_features(test_tokens, tokenizer, max_len)
    test_ids = torch.tensor([item.cpu().detach().numpy() for item in test_ids]).squeeze()
    test_masks = torch.tensor([item.cpu().detach().numpy() for item in test_attention_masks]).squeeze()
    test_token_type_ids = torch.tensor([item.cpu().detach().numpy() for item in test_token_type_ids]).squeeze()
    test_data = TensorDataset(test_ids, test_masks, test_token_type_ids)
    test_dataloader = DataLoader(test_data, batch_size=bs)

# ------------------模型設定-------------------
if is_answer:
    precesion_scores,recall_scores,f1_scores = [],[],[]

fewshot = FewShot_NER(base_path,tag2id,bs,label_file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fewshot.to(device)

# ------------------LSM-------------------
fewshot.load_state_dict(torch.load(LSM_file))
fewshot.eval()

if is_answer:
    test_pre,test_true = [],[]
    for batch in test_dataloader:

        input_ids,masks,token_type_ids,labels= (i.to(device) for i in batch)

        with torch.no_grad():
            matrix_embeddings,output_indexs = fewshot({"input_ids":input_ids,"attention_mask":masks,"token_type_ids":token_type_ids},flag = True)

        test_pre.extend(output_indexs.detach().cpu().numpy().tolist())
        test_true.extend(labels.to('cpu').numpy().tolist())
else:
    test_pre = []
    for batch in test_dataloader:
        input_ids, masks, token_type_ids = (i.to(device) for i in batch)

        with torch.no_grad():
            matrix_embeddings, output_indexs = fewshot({"input_ids": input_ids, "attention_mask": masks, "token_type_ids": token_type_ids}, flag=True)

        test_pre.extend(output_indexs.detach().cpu().numpy().tolist())

# ------------------紀錄LSM數據-------------------
if is_answer:
    test_f1, test_precision, test_recall, mistake, mistake_num = test_measure(test_pre,test_true,test_tokens,test_lengths,id2tag)
    precesion_scores.append(test_precision)
    recall_scores.append(test_recall)
    f1_scores.append(test_f1)

# ------------------過濾出需要校正的資料-------------------
tags = trans2label(id2tag,test_pre,test_lengths)
mistake_num = find_bio_error_num(tags)
corrected_tokens = [test_tokens[i] for i in mistake_num]
if is_answer:
    corrected_labels = [test_labels[i] for i in mistake_num]

# ------------------模型參數-------------------
max_len = 512
bs = 32

# ------------------把文本轉換為BERT模型的輸入格式，並創建DataLoader-------------------
if is_answer:
    corrected_ids,corrected_token_type_ids,corrected_attention_masks,corrected_tags,corrected_lengths = gen_features(corrected_tokens,corrected_labels,tokenizer,tag2id,max_len)
    corrected_ids = torch.tensor([item.cpu().detach().numpy() for item in corrected_ids]).squeeze()
    corrected_tags = torch.tensor(corrected_tags)
    corrected_masks = torch.tensor([item.cpu().detach().numpy() for item in corrected_attention_masks]).squeeze()
    corrected_token_type_ids = torch.tensor([item.cpu().detach().numpy() for item in corrected_token_type_ids]).squeeze()
    corrected_data = TensorDataset(corrected_ids, corrected_masks, corrected_token_type_ids, corrected_tags)
    corrected_dataloader = DataLoader(corrected_data, batch_size=bs)
else:
    corrected_ids,corrected_token_type_ids,corrected_attention_masks,corrected_lengths = no_labels_gen_features(corrected_tokens,tokenizer,max_len)
    corrected_ids = torch.tensor([item.cpu().detach().numpy() for item in corrected_ids]).squeeze()
    corrected_masks = torch.tensor([item.cpu().detach().numpy() for item in corrected_attention_masks]).squeeze()
    corrected_token_type_ids = torch.tensor([item.cpu().detach().numpy() for item in corrected_token_type_ids]).squeeze()
    corrected_data = TensorDataset(corrected_ids, corrected_masks, corrected_token_type_ids)
    corrected_dataloader = DataLoader(corrected_data, batch_size=bs)

# ------------------LCM-------------------
corrected_pre,corrected_true = [],[]

fewshot.load_state_dict(torch.load(LCM_file))
fewshot.eval()

for batch in corrected_dataloader:

    if is_answer:
        input_ids,masks,token_type_ids,labels= (i.to(device) for i in batch)
    else:
        input_ids,masks,token_type_ids= (i.to(device) for i in batch)

    with torch.no_grad():
        matrix_embeddings,output_indexs = fewshot({"input_ids":input_ids,"attention_mask":masks,"token_type_ids":token_type_ids},flag = True)

    corrected_pre.extend(output_indexs.detach().cpu().numpy().tolist())
    if is_answer:
        corrected_true.extend(labels.detach().cpu().numpy().tolist())

# ------------------替換校正內容-------------------
replaced_pre = copy.deepcopy(test_pre)

num = 0
for index in mistake_num:
    error_index_list = find_bio_error_index(tags[index])
    for i, error_index in enumerate(error_index_list):
        replaced_pre[index][error_index] = corrected_pre[num][error_index]
    num+=1

# ------------------規則式-------------------
three_stage_pre = rule_based_filter_alone_I_tags(replaced_pre, bio_tags, tag2id)
three_stage_pre = rule_based_consistent_BI_tags(replaced_pre, tag2id, id2tag)

# ------------------輸出預測結果-------------------
if not is_answer:
    tag_sequences = trans2label(id2tag,three_stage_pre,test_lengths)# pre會是id，要先轉成tag後再輸出
    output_predict(test_tokens,tag_sequences ,result_path, result_file)

# 如果test集有答案的話
# 這邊會執行並記錄消融實驗的結果
if is_answer:
    # ------------------紀錄LSM+LCM數據------------------- 
    replaced_f1, replaced_precision, replaced_recall, replaced_mistake, replaced_mistake_num = test_measure(replaced_pre,test_true,test_tokens,test_lengths,id2tag)
    precesion_scores.append(replaced_precision)
    recall_scores.append(replaced_recall)
    f1_scores.append(replaced_f1)

    # ------------------執行並紀錄LSM+規則式數據-------------------
    rule_based_pre = copy.deepcopy(test_pre)
    rule_based_pre = rule_based_filter_alone_I_tags(rule_based_pre, bio_tags, tag2id)
    rule_based_pre = rule_based_consistent_BI_tags(rule_based_pre, tag2id, id2tag)

    rule_based_f1, rule_based_precision, rule_based_recall, rule_based_mistake, rule_based_mistake_num = test_measure(rule_based_pre,test_true,test_tokens,test_lengths,id2tag)
    precesion_scores.append(rule_based_precision)
    recall_scores.append(rule_based_recall)
    f1_scores.append(rule_based_f1)

    # ------------------紀錄LSM+LCM+規則式數據-------------------
    three_stage_f1, three_stage_precision, three_stage_recall, three_stage_mistake, three_stage_mistake_num = test_measure(three_stage_pre,test_true,test_tokens,test_lengths,id2tag)
    precesion_scores.append(three_stage_precision)
    recall_scores.append(three_stage_recall)
    f1_scores.append(three_stage_f1)

    # ------------------繪製不同模型組合方式折線圖-------------------
    custom_labels = ['LSM', 'LSM+LCM', 'LSM+rule_based', 'three_stage']
    plt.plot(range(1, 5), precesion_scores, label='PRECISION')
    plt.plot(range(1, 5), recall_scores, label='RECALL')
    plt.plot(range(1, 5), f1_scores, label='F1')
    plt.legend()
    plt.xticks(range(1, 5), custom_labels)
    plt.show()