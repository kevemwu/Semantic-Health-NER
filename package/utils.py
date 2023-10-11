import json
from datetime import datetime
import uuid

def trans2id(label_file):
    with open(label_file, 'r', encoding='utf-8') as f:
        labels = json.load(f)

    #short labels
    short_labels = labels.keys()

    tag_set = []
    for line in short_labels:
        prefix = ['B-','I-']
        tag_set += [pre+line for pre in prefix]
    tag_set.append('O')

    tag_set = list(set(tag_set))
    idx = [i for i in range(len(tag_set))]
    tag2id = dict(zip(tag_set,idx))
    id2tag = dict(zip(idx,tag_set))
    return tag2id,id2tag,short_labels

def trans2label(id2tag,data,lengths):
    new = []
    for i,line in enumerate(data):
        tmp = [id2tag[word] for word in line]
        tmp = tmp[1:1 + lengths[i]]
        new.append(tmp)
    return new

def gen_features(tokens,labels,tokenizer,tag2id,max_len):
    tags,input_ids,token_type_ids,attention_masks,lengths = [],[],[],[],[]
    for i,(token,label) in enumerate(zip(tokens,labels)):
        sentence = ''.join(token)
        lengths.append(len(sentence))
        if len(token) >= max_len - 2:
            label = labels[i][0:max_len - 2]
        label = [tag2id['O']] + [tag2id[i] for i in label] + [tag2id['O']]
        if len(label) < max_len:
            label = label + [tag2id['O']] * (max_len - len(label))

        # assert len(label) == max_len
        tags.append(label)

        inputs = tokenizer.encode_plus(sentence, max_length=max_len,pad_to_max_length=True,return_tensors='pt')
        input_id,token_type_id,attention_mask = inputs['input_ids'],inputs['token_type_ids'],inputs['attention_mask']
        input_ids.append(input_id)
        token_type_ids.append(token_type_id)
        attention_masks.append(attention_mask)
    return input_ids,token_type_ids,attention_masks,tags,lengths

def no_labels_gen_features(tokens, tokenizer, max_len):
    input_ids, token_type_ids, attention_masks, lengths = [], [], [], []
    for i, token in enumerate(tokens):
        sentence = ''.join(token)
        lengths.append(len(sentence))
        if len(token) >= max_len - 2:
            token = token[0:max_len - 2]

        inputs = tokenizer.encode_plus(sentence, max_length=max_len, pad_to_max_length=True, return_tensors='pt')
        input_id, token_type_id, attention_mask = inputs['input_ids'], inputs['token_type_ids'], inputs['attention_mask']
        input_ids.append(input_id)
        token_type_ids.append(token_type_id)
        attention_masks.append(attention_mask)
    return input_ids, token_type_ids, attention_masks, lengths

def get_entities(tags):
    start, end = -1, -1
    prev = 'O'
    entities = []
    n = len(tags)
    tags = [tag.split('-')[1] if '-' in tag else tag for tag in tags]
    for i, tag in enumerate(tags):
        if tag != 'O':
            if prev == 'O':
                start = i
                prev = tag
            elif tag == prev:
                end = i
                if i == n -1 :
                    entities.append((start, i))
            else:
                entities.append((start, i - 1))
                prev = tag
                start = i
                end = i
        else:
            if start >= 0 and end >= 0:
                entities.append((start, end))
                start = -1
                end = -1
                prev = 'O'
    return entities

def measure(preds,trues,lengths,id2tag):
    correct_num = 0
    predict_num = 0
    truth_num = 0
    pred = trans2label(id2tag,preds,lengths)
    true = trans2label(id2tag,trues,lengths)
    # print(len(pred),len(true))
    assert len(pred) == len(true)
    for p,t in zip(pred,true):
        pred_en = get_entities(p)
        true_en = get_entities(t)
        correct_num += len(set(pred_en) & set(true_en))
        predict_num += len(set(pred_en))
        truth_num += len(set(true_en))
    precision = correct_num / predict_num if predict_num else 0
    recall = correct_num / truth_num if truth_num else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return f1, precision, recall

def test_measure(preds,trues,tokens,lengths,id2tag):
    correct_num = 0
    predict_num = 0
    truth_num = 0
    pred = trans2label(id2tag, preds, lengths)
    true = trans2label(id2tag, trues, lengths)
    assert len(pred) == len(true)

    mistake = []
    mistake_num = []

    for i, (p, t, k) in enumerate(zip(pred,true,tokens)):
        if i==1 :
            print(p, t, k)
        pred_en = get_entities(p)
        true_en = get_entities(t)
        correct_num += len(set(pred_en) & set(true_en))
        predict_num += len(set(pred_en))
        truth_num += len(set(true_en))

        if p != t:  # 檢查是否不一致，把不一致的部分存入 mistake
            mistake.append((t, p, k))
            mistake_num.append((i))

    precision = correct_num / predict_num if predict_num else 0
    recall = correct_num / truth_num if truth_num else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return f1, precision, recall, mistake, mistake_num

def find_bio_error_num(tags_list):
    error_num = []
    for i, tags in enumerate(tags_list):
        if find_bio_error_index(tags)!=[]:
            error_num.append(i)
    return error_num

def find_bio_error_index(tags):
    error_index,ner_index = [],[]

    pre_bio,pre_tag = '',''
    is_ner,is_inconsistent = False,False

    for i, tag in enumerate(tags):
        tag_content = tag.split('-')
        # 前是O就直接I的錯誤
        if tag_content[0] == 'I' and (pre_bio =='O'):
              error_index.append(i)

        # 如果是B開頭就進ner的判斷
        if tag_content[0] == 'B':
              is_ner = True
              ner_index.append(i) # 把ner B的index記錄起來
        # ner中且當下是I就進判斷
        if is_ner and tag_content[0]=='I':
              ner_index.append(i) # 把ner I的index記錄起來
              if tag_content[1] != pre_tag:# 前一個tag跟當下的tag不一樣
                  is_inconsistent = True
        # inconsistent且脫離ner範圍，或是當下是tags的最後一個元素就把ner加入error
        if is_inconsistent:
              if tag_content[0]=='O' or i == (len(tags)-1):
                  error_index = error_index + ner_index
                  ner_index = []
                  is_ner = False
                  is_inconsistent = False

        if tag_content[0] != 'O':
            pre_tag = tag_content[1]
        pre_bio = tag_content[0]

    return error_index

# 輸出錯誤結果
def output_mistake_result(precision, recall, f1, mistake, mistake_num, result_path, result_file_name):
    current_time = datetime.now()
    timestamp = current_time.strftime("%Y%m%d%H%M%S")
    unique_id = uuid.uuid4().hex

    file_name = f"{result_file_name}_{timestamp}_{unique_id}.json"

    result_data=[]
    result_item = {
        "mistake_num":len(mistake_num),
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    result_data.append(result_item)

    for j, item in enumerate(mistake):
        true_labels, predicted_labels, tokens = [], [], []
        for i, (t, p, k) in enumerate(zip(item[0], item[1], item[2])):
            if t != p:
                true_labels.append(t)
                predicted_labels.append(p)
                tokens.append(k)

        result_item = {
            "Sentence":item[2],
            "Num": mistake_num[j],
            "Tokens": tokens,
            "True Labels": true_labels,
            "Predicted Labels": predicted_labels,
            "Full True Labels":item[0],
            "Full Pre Labels":item[1]
        }

        result_data.append(result_item)

    with open(result_path + file_name, 'w') as f:
        json.dump(result_data, f, indent=4, ensure_ascii=False)

# 輸出correction_model需要的data
def output_correction_data(mistake, file_name):
    result_data=[]

    for j, item in enumerate(mistake):
        pre_bio,pre_tag = '',''
        is_BIO_mistake,is_ner = False,False
        for i, predicted_labels in enumerate(item[1]):
            tag_content = predicted_labels.split('-')
            # 前是O就直接I的錯誤
            if tag_content[0] == 'I' and (pre_bio =='O'):
                  is_BIO_mistake = True
                  break
            # 如果是B開頭就開始對ner的判斷
            if tag_content[0] == 'B':
                  is_ner = True
            # 在ner中且當下是I就進判斷
            if is_ner and tag_content[0]=='I':
                if tag_content[1] != pre_tag:# 前一個tag跟當下的tag不一樣
                      is_BIO_mistake = True
                      break
            if tag_content[0] != 'O':
                pre_tag = tag_content[1]
            pre_bio = tag_content[0]

        if is_BIO_mistake:
              result_item = {
            "Sentence":item[2],
            "Full True Labels":item[0],
            "Full Pre Labels":item[1]
            }
              result_data.append(result_item)

    print(len(result_data))
    
    with open(file_name, 'w') as f:
        json.dump(result_data, f, indent=4, ensure_ascii=False)

# 輸出"{token}\t{label}\n"格式的檔案
def output_predict(tokens, tags, result_path, result_file_name):
    current_time = datetime.now()
    timestamp = current_time.strftime("%Y%m%d%H%M%S")
    unique_id = uuid.uuid4().hex

    file_name = f"{result_file_name}_{timestamp}_{unique_id}.json"

    merged_results = []
    for i in range(len(tags)):
        merged_sentence = []
        for j in range(len(tags[i])):
            merged_sentence.append((tokens[i][j], tags[i][j]))
        merged_results.append(merged_sentence)
    with open(result_path + file_name, 'w', encoding='utf-8') as output_file:
        for sentence in merged_results:
            for token, label in sentence:
                output_file.write(f"{token}\t{label}\n")
            output_file.write("\n")

# 輸出"{token}\t{label}\n"格式的檔案
def output_predict(tokens, tags, result_path, result_file_name):
    current_time = datetime.now()
    timestamp = current_time.strftime("%Y%m%d%H%M%S")
    unique_id = uuid.uuid4().hex

    file_name = f"{result_file_name}_{timestamp}_{unique_id}.txt"

    merged_results = []
    for i in range(len(tags)):
        merged_sentence = []
        for j in range(len(tokens[i])):
            merged_sentence.append((tokens[i][j], tags[i][j]))
        merged_results.append(merged_sentence)
    with open(result_path + file_name, 'w', encoding='utf-8') as output_file:
        for sentence in merged_results:
            for token, label in sentence:
                output_file.write(f"{token}\t{label}\n")
            output_file.write("\n")
            
def rule_based_filter_alone_I_tags(tags, bio_labels, tag2id):
    for tag_sequence in tags:
        pre_bio = ''
        for i, tag_id in enumerate(tag_sequence):
            if bio_labels[tag_id] == 'I' and (pre_bio =='O'):
                tag_sequence[i] = tag2id['O']
                pre_bio = 'O'
            else:
                pre_bio = bio_labels[tag_id]
    return tags

def rule_based_consistent_BI_tags(tags, tag2id, id2tag):
    for tag_sequence in tags:
        pre_tag = ''
        is_ner = False
        for i, tag_id in enumerate(tag_sequence):
            if id2tag[tag_id] == 'O':
                tag_content = 'O'
            else:
                tag_content = id2tag[tag_id].split('-')
            if tag_content[0] == 'B':
                is_ner = True
                pre_tag = tag_content[1]
            if is_ner and tag_content[0]=='I':
                if tag_content[1] != pre_tag:
                    tag_sequence[i] = tag2id['I-'+pre_tag]
            if tag_content[0] == 'O':
                is_ner = False
                pre_tag = ''
    return tags