# https://dacon.io/competitions/official/235900/overview/description
# 데이콘 코드 유사성 판단 대회

# AutoTokenizer로 graphcodebert 사용하도록 설정 : tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")


################### 0. packages ####################


from collections import deque
from transformers import *
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from itertools import combinations

from rank_bm25 import BM25L
import torch
import torch.nn as nn
import random
import time
import datetime
import numpy as np
import pandas as pd
import os, re
import gc

################### 1. gpu setting #####################
torch.cuda.is_available()
torch.cuda.get_device_name(0)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(torch.cuda.device_count())  # 1
print(torch.cuda.get_device_name(0))  # GeForce RTX 2080 Ti
print(torch.cuda.is_available())  # True
# GPU 할당
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(device)  # a = a.to(device) 다음과 같이 사용할 수 있음

gc.collect()
torch.cuda.empty_cache()

import GPUtil
GPUtil.showUtilization()



################### 2. Preprocessing ####################

def preprocess_script(script):
    new_script = deque()
    with open(script, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            if line.lstrip().startswith('#'):
                continue
            line = line.rstrip()
            if '#' in line:
                line = line[:line.index('#')]
            line = line.replace('\n', '')
            line = line.replace('    ', '\t')

            if line == '':
                continue

            new_script.append(line)

        new_script = '\n'.join(new_script)
        new_script = re.sub('("""[\w\W]*?""")', '', new_script)
        new_script = re.sub('/^(http?|https?):\/\/([a-z0-9-]+\.)+[a-z0-9]{2,4}.*$/', '', new_script)
        # 조건 추가
        new_script = re.sub('\"\"\"([^\"]*)\"\"\"', "", new_script)
        new_script = re.sub(r'\'\w+', '', new_script)
        new_script = re.sub(r'\w*\d+\w*', '', new_script)
        new_script = re.sub(r'\s{2,}', ' ', new_script)
        new_script = re.sub(r'\s[^\w\s]\s', '', new_script)

    return new_script

def clean_extra_newlines(s):  # 오른쪽 공백, duplicate newlines를 제거
    """
    1) strip right spaces
    2) leave left spaces as is because this is indentation
    3) remove duplicate newlines
    """
    lines = s.split('\n')
    clean = []
    for line in lines:
        if line.strip() == '':
            continue
        line = line.rstrip()
        clean.append(line)
    return '\n'.join(clean)

def clean_singleline_comments(s):  # 한줄에 걸친 주석 없애기
    """
    Remove sharp-leading comments from the beginning and any other place
    """
    lines = s.split('\n')
    clean = []
    for line in lines:
        if line.lstrip().startswith('#'):
            continue
        if '#' in line:
            line = line[:line.index('#')].rstrip()
        clean.append(line)
    return '\n'.join(clean)

def clean_multiline_comments(s):  # 여러줄에 걸친 주석 없애기
    """
    Remove all strings enclosed in triple quotes:
    1) file-level doc-string
    2) class, function doc-strings
    3) any multiline string and related variable e.g. "x = '''hello'''" is completely removed
    """
    for q in ['"""', "'''"]:
        if q not in s:
            continue
        lines = s.split('\n')
        cleaned = []
        flag = False
        for line in lines:
            if line.count(q) == 2:
                continue
            if line.count(q) == 1:
                flag = not flag
                continue
            if flag:
                continue
            cleaned.append(line)
        s = '\n'.join(cleaned)
    return s

def clean_2to3(s):  # python2 코드를 python3코드로 바꾸기
    """
    Convert Python2 code to Python3
    """
    s = s.replace('xrange', 'range')
    s = s.replace('raw_input', 'input')
    lines = s.split('\n')
    clean = []
    for line in lines:
        if 'print' in line and not ('print(' in line or 'print (' in line):
            line = line.replace('print', 'print(')
            line += ')'
        clean.append(line)
    return '\n'.join(clean)

def clean_lex(s):  # """Remove imports and `if __name__`"""
    """Remove imports and `if __name__`"""
    lines = s.split('\n')
    clean = []
    for line in lines:
        if 'import' in line:
            continue
        if 'if __name__' in line:
            continue
        clean.append(line)
    return '\n'.join(clean)

def clean_indents(s):  # 여러종류의 space종류들을 single space로 바꾸기
    """
    Replace all types of spaces:
    space, tab, (multi)-new-line with a single space
    """
    return ' '.join(s.split())

def preproc(s):
    """Apply all cleaning functions"""
    s = clean_extra_newlines(s)
    s = clean_singleline_comments(s)
    s = clean_multiline_comments(s)
    s = clean_2to3(s)
    s = clean_lex(s)
    s = clean_indents(s)
    return s

def seed_everything(seed=1004):
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Seed set as {seed}")

seed_everything(1004)

code_folder = 'C:/Users/s_sanghkim/Desktop/김재경/NNAP/code/'  # "./input/code/"
problem_folders = os.listdir(code_folder)

preprocess_scripts = []  # list : 45101
problem_nums = []  # list : 45101

for problem_folder in tqdm(problem_folders):
    scripts = os.listdir(os.path.join(code_folder, problem_folder))
    problem_num = problem_folder
    for script in scripts:
        script_file = os.path.join(code_folder, problem_folder, script)
        preprocessed_script = preprocess_script(script_file)
        preprocessed_script = preproc(preprocessed_script) # 추가 : 데이터 클렌징 요소 추가시킴
        preprocess_scripts.append(preprocessed_script)

    problem_nums.extend([problem_num] * len(scripts))

df = pd.DataFrame(data={'code': preprocess_scripts, 'problem_num': problem_nums})  # 두개컬럼으로 구성된 data frame
df.to_csv('C:/Users/s_sanghkim/Desktop/김재경/NNAP/preprocessing_data.csv')

######  preprocessing에 대한 check (샘플 1개만) #####






############################ 3. tokenizer ##########################################

tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
tokenizer.truncation_side = 'left'
MAX_LEN = 512  # 최대길이 512

tokens = []
for code in df['code']:
    tokens.append(tokenizer.tokenize(code, max_length=MAX_LEN, truncation=True))

df['tokens'] = tokens
df['len'] = df['tokens'].apply(len)
# df 에 tokens 컬럼과 len컬럼이 추가됨 / df.columns : Index(['code', 'problem_num', 'tokens', 'len'], dtype='object')


############################ 4. Prepare datasets ##########################################
train_df, valid_df, train_label, valid_label = train_test_split(
    df,
    df['problem_num'],
    random_state=42,
    test_size=0.1,  # len(train_df) : 40590   ,  len(valid_df) : 4511
    stratify=df['problem_num']
)

##########  4-1 train data from task ############
codes = train_df['code'].to_list()  # list형태로 변환
problems = train_df['problem_num'].unique().tolist()  # list형태로 변환후 중복제거 길이 300 / 300 problem이기 떄문
problems.sort()  # 문제 번호순으로 정렬

total_positive_pairs = []  # list 30000  (300 x 100)
total_negative_pairs = []  # list 30000  (300 x 100)

for problem in tqdm(problems):  # 300개의 문제들이 하나씩 뽑힘
    solution_codes = train_df[train_df['problem_num'] == problem]['code'].to_list()  # 해당 problem number와 같은 것들
    other_codes = train_df[train_df['problem_num'] != problem]['code'].to_list()  # 해당 problem number와 다른 것들

    positive_pairs = list(combinations(solution_codes, 2))  # solution codes에서 2개를 뽑은 조합리스트
    random.shuffle(positive_pairs)  # 임의로 섞음
    positive_pairs = positive_pairs[:100]  # 그 중에서 100개 추출
    random.shuffle(other_codes)
    other_codes = other_codes[:100]  # 그외 코드 100개 추출

    negative_pairs = []
    for pos_codes, others in zip(positive_pairs, other_codes):  # positive_pairs에서 하나뽑고, other codes에서 하나를 뽑아 구성
        negative_pairs.append((pos_codes[0], others))

    total_positive_pairs.extend(positive_pairs)  # 총 30000개 list / 길이 2의 tuple로 구성됨 (대상코드, positive)
    total_negative_pairs.extend(negative_pairs)  # 총 30000개 list / 길이 2의 tuple로 구성됨 (대상코드, negative)

code1 = [code[0] for code in total_positive_pairs] + [code[0] for code in
                                                      total_negative_pairs]  # list 60000 : 30000 + 30000
code2 = [code[1] for code in total_positive_pairs] + [code[1] for code in
                                                      total_negative_pairs]  # list 60000 : 30000 + 30000
label = [1] * len(total_positive_pairs) + [0] * len(total_negative_pairs)  # list 60000 [1.....1, 0.....0]

train_data = pd.DataFrame(data={'code1': code1, 'code2': code2, 'similar': label})  # 60000 x 3
train_data = train_data.sample(frac=1).reset_index(drop=True)  # frac: 추출할 표본 비율 / index를 재배열함
train_data.to_csv('C:/Users/s_sanghkim/Desktop/김재경/NNAP/train_data_lv1.csv', index=False)  # 해당 데이터를 저장함

##########  4-2 valid data  from task ############
codes = valid_df['code'].to_list()  # len : 4511
problems = valid_df['problem_num'].unique().tolist()
problems.sort()

total_positive_pairs = []  # 60000
total_negative_pairs = []  # 60000

for problem in tqdm(problems):
    solution_codes = valid_df[valid_df['problem_num'] == problem]['code'].to_list()
    other_codes = valid_df[valid_df['problem_num'] != problem]['code'].to_list()

    positive_pairs = list(combinations(solution_codes, 2))
    random.shuffle(positive_pairs)
    positive_pairs = positive_pairs[:100]
    random.shuffle(other_codes)
    other_codes = other_codes[:100]

    negative_pairs = []
    for pos_codes, others in zip(positive_pairs, other_codes):
        negative_pairs.append((pos_codes[0], others))

    total_positive_pairs.extend(positive_pairs)
    total_negative_pairs.extend(negative_pairs)

code1 = [code[0] for code in total_positive_pairs] + [code[0] for code in total_negative_pairs]
code2 = [code[1] for code in total_positive_pairs] + [code[1] for code in total_negative_pairs]
label = [1] * len(total_positive_pairs) + [0] * len(total_negative_pairs)

valid_data = pd.DataFrame(data={'code1': code1, 'code2': code2, 'similar': label})  # 60000
valid_data = valid_data.sample(frac=1).reset_index(drop=True)  # code1, code2, similar 3개 컬럼으로 구성됨
valid_data.to_csv('C:/Users/s_sanghkim/Desktop/김재경/NNAP/valid_data_lv1.csv', index=False)

########### 4-3  (추가) IBM의 CodeNet 추가 코드  ###########

code_folder = 'C:/Users/s_sanghkim/Desktop/김재경/NNAP/Project_CodeNet_Python800'  # CodeNet 데이터 경로
problem_folders = os.listdir(code_folder)

preproc_scripts = []
problem_nums = []

for problem_folder in tqdm(problem_folders):
    scripts = os.listdir(os.path.join(code_folder, problem_folder))
    problem_num = int(problem_folder.split('p')[1])
    problem_num = 'problem' + str(problem_num)
    for script in scripts:
        script_file = os.path.join(code_folder, problem_folder, script)
        preprocessed_script = preprocess_script(script_file)
        preprocessed_script = preproc(preprocessed_script)

        preproc_scripts.append(preprocessed_script)
    problem_nums.extend([problem_num] * len(scripts))
codenet_df = pd.DataFrame(data={'code': preproc_scripts, 'problem_num': problem_nums}) # len : 240000

# [필터링] : codenet_df에서 test_df의 데이터와 겹치는 녀석들을 set (hash table) 으로 필터링 (거의 대부분)
concat_codes = np.concatenate([train_data['code1'].values,
                              train_data['code2'].values,
                              valid_data['code1'].values,
                               valid_data['code2'].values])
_codes_set = set() # 40128

for i in tqdm(range(len(concat_codes))):
    _codes_set.add(concat_codes[i])

usable_codes = []
usable_problem_nums = []

codenet_codes = codenet_df['code'].values
problem_nums = codenet_df['problem_num'].values

for i in tqdm(range(len(codenet_codes))):
    if codenet_codes[i] not in _codes_set:
        usable_codes.append(codenet_codes[i])
        usable_problem_nums.append(problem_nums[i])

filtered_codenet_df = pd.DataFrame(data={'code': usable_codes,
                                         'problem_num': usable_problem_nums})  # len : 195841

# 리소스 문제로, 완성된 filtered_codenet_df 중 50%의 데이터만을 이용해서 학습에 사용합니다.
filtered_codenet_df = filtered_codenet_df.sample(frac=0.5, random_state=42)  # 97920


codenet_train_df, codenet_valid_df, codenet_train_label, codenet_valid_label = train_test_split(
        filtered_codenet_df,
        filtered_codenet_df['problem_num'],
        random_state=42,
        test_size=0.1,
        stratify=filtered_codenet_df['problem_num']
    )
codenet_train_df = codenet_train_df.reset_index(drop=True)
codenet_valid_df = codenet_valid_df.reset_index(drop=True)

# negative/positive 만드는 함수
def get_pairs(input_df, tokenizer):
    codes = input_df['code'].to_list()
    problems = input_df['problem_num'].unique().tolist()
    problems.sort()

    tokenized_corpus = [tokenizer.tokenize(code) for code in codes]
    bm25 = BM25L(tokenized_corpus)

    total_positive_pairs = []
    total_negative_pairs = []

    for problem in tqdm(problems):
        solution_codes = input_df[input_df['problem_num'] == problem]['code']
        positive_pairs = list(combinations(solution_codes.to_list(),2))

        solution_codes_indices = solution_codes.index.to_list()
        negative_pairs = []

        first_tokenized_code = tokenizer.tokenize(positive_pairs[0][0])
        negative_code_scores = bm25.get_scores(first_tokenized_code)
        negative_code_ranking = negative_code_scores.argsort()[::-1] # 내림차순
        ranking_idx = 0

        for solution_code in solution_codes:
            negative_solutions = []
            while len(negative_solutions) < len(positive_pairs) // len(solution_codes):
                high_score_idx = negative_code_ranking[ranking_idx]

                if high_score_idx not in solution_codes_indices:
                    negative_solutions.append(input_df['code'].iloc[high_score_idx])
                ranking_idx += 1

            for negative_solution in negative_solutions:
                negative_pairs.append((solution_code, negative_solution))

        total_positive_pairs.extend(positive_pairs)
        total_negative_pairs.extend(negative_pairs)

    pos_code1 = list(map(lambda x:x[0],total_positive_pairs))
    pos_code2 = list(map(lambda x:x[1],total_positive_pairs))

    neg_code1 = list(map(lambda x:x[0],total_negative_pairs))
    neg_code2 = list(map(lambda x:x[1],total_negative_pairs))

    pos_label = [1]*len(pos_code1)
    neg_label = [0]*len(neg_code1)

    pos_code1.extend(neg_code1)
    total_code1 = pos_code1
    pos_code2.extend(neg_code2)
    total_code2 = pos_code2
    pos_label.extend(neg_label)
    total_label = pos_label
    pair_data = pd.DataFrame(data={
        'code1':total_code1,
        'code2':total_code2,
        'similar':total_label
    })
    pair_data = pair_data.sample(frac=1).reset_index(drop=True)
    return pair_data

codenet_train_bm25L = get_pairs(codenet_train_df, tokenizer)  # 10335134 x 3
codenet_valid_bm25L = get_pairs(codenet_valid_df, tokenizer)  # 116757 x 3
# 생성된 데이터를 저장합니다.
codenet_train_bm25L.to_csv("C:/Users/s_sanghkim/Desktop/김재경/NNAP/" + "graph_codenet_train_bm25L.csv",index=False)
codenet_valid_bm25L.to_csv("C:/Users/s_sanghkim/Desktop/김재경/NNAP/" + "graph_codenet_valid_bm25L.csv",index=False)

# codenet데이터와 해당 과제의 데이터를 합치기 / 데이터 합치는 부분 부터 시작해도됨
train_data = pd.read_csv("C:/Users/s_sanghkim/Desktop/김재경/NNAP/train_data_lv1.csv")
valid_data = pd.read_csv("C:/Users/s_sanghkim/Desktop/김재경/NNAP/valid_data_lv1.csv")
codenet_train_data = pd.read_csv("C:/Users/s_sanghkim/Desktop/김재경/NNAP/" + "graph_codenet_train_bm25L.csv")
codenet_valid_data = pd.read_csv("C:/Users/s_sanghkim/Desktop/김재경/NNAP/" + "graph_codenet_valid_bm25L.csv")
train_data = pd.concat([train_data, codenet_train_data], axis=0) # 10395134 x 3
valid_data = pd.concat([valid_data, codenet_valid_data], axis=0) # 176757 x 3

# 데이터가 너무 크기때문에 일부분 잘라낼예정 / 기존 train 6만 + valid 6만 총 12만 -> train 20만 + valid 10만 총 30만 (약 2.5배 증가)
train_data = train_data[:200000]
valid_data = train_data[:100000]



############################ 5. Traininig ##########################################

epochs = 3
# batch size 원래 32
test_batch_size = 16
batch_size = 16

# training data
c1 = train_data['code1'].values  # 코드1
c2 = train_data['code2'].values  # 코드2
similar = train_data['similar'].values  # target

N = train_data.shape[0]  # 전체 데이터셋 수
MAX_LEN = 512

input_ids = np.zeros((N, MAX_LEN), dtype=int)
attention_masks = np.zeros((N, MAX_LEN), dtype=int)
labels = np.zeros((N), dtype=int)

tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")

for i in tqdm(range(N), position=0, leave=True):
    try:
        cur_c1 = str(c1[i])
        cur_c2 = str(c2[i])
        encoded_input = tokenizer(cur_c1, cur_c2, return_tensors='pt', max_length=512, padding='max_length',
                                  truncation=True)
        input_ids[i,] = encoded_input['input_ids']
        attention_masks[i,] = encoded_input['attention_mask']
        labels[i] = similar[i]
    except Exception as e:
        print(e)
        pass


# validating data
c1 = valid_data['code1'].values # 코드1
c2 = valid_data['code2'].values # 코드2
similar = valid_data['similar'].values # target

N = valid_data.shape[0]

MAX_LEN = 512

valid_input_ids = np.zeros((N, MAX_LEN), dtype=int)
valid_attention_masks = np.zeros((N, MAX_LEN), dtype=int)
valid_labels = np.zeros((N), dtype=int)

for i in tqdm(range(N), position=0, leave=True):
    try:
        cur_c1 = str(c1[i])
        cur_c2 = str(c2[i])
        encoded_input = tokenizer(cur_c1, cur_c2, return_tensors='pt', max_length=512, padding='max_length',
                                  truncation=True)
        valid_input_ids[i,] = encoded_input['input_ids']
        valid_attention_masks[i,] = encoded_input['attention_mask']
        valid_labels[i] = similar[i]
    except Exception as e:
        print(e)
        pass


# torch.tensor로 전환

input_ids = torch.tensor(input_ids, dtype=int)
attention_masks = torch.tensor(attention_masks, dtype=int)
labels = torch.tensor(labels, dtype=int)

valid_input_ids = torch.tensor(valid_input_ids, dtype=int)
valid_attention_masks = torch.tensor(valid_attention_masks, dtype=int)
valid_labels = torch.tensor(valid_labels, dtype=int)


# Setup training
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

train_data = TensorDataset(input_ids, attention_masks, labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(valid_input_ids, valid_attention_masks, valid_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)


model = AutoModelForSequenceClassification.from_pretrained("microsoft/graphcodebert-base")
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-5)  # 아직 이게 정확하지 않음

total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
loss_f = nn.CrossEntropyLoss()

# Train
train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []
model.zero_grad()



for i in range(epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(i + 1, epochs))
    print('Training...')
    t0 = time.time()
    train_loss, train_accuracy = 0, 0
    model.train()
    for step, batch in tqdm(enumerate(train_dataloader), desc="Iteration", smoothing=0.05):
        if step % 10000 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            print('  current average loss = {}'.format(
                train_loss / step))  # bot.sendMessage(chat_id=chat_id, text = '  current average loss = {}'.format(train_loss / step))

        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs[0]
        logits = outputs[1]
        train_loss += loss.item()
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.detach().cpu().numpy()
        train_accuracy += flat_accuracy(logits, label_ids)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        model.zero_grad()
    avg_train_loss = train_loss / len(train_dataloader)
    avg_train_accuracy = train_accuracy / len(train_dataloader)
    train_losses.append(avg_train_loss)
    train_accuracies.append(avg_train_accuracy)
    print("  Average training loss: {0:.8f}".format(avg_train_loss))
    print("  Average training accuracy: {0:.8f}".format(avg_train_accuracy))
    print("  Training epoch took: {:}".format(format_time(time.time() - t0)))

    print("")
    print("Validating...")
    t0 = time.time()
    model.eval()
    val_loss, val_accuracy = 0, 0
    for step, batch in tqdm(enumerate(validation_dataloader), desc="Iteration", smoothing=0.05):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_input_mask)

        logits = outputs[0]
        logits = logits.detach().cpu()
        label_ids = b_labels.detach().cpu()
        val_loss += loss_f(logits, label_ids)

        logits = logits.numpy()
        label_ids = label_ids.numpy()
        val_accuracy += flat_accuracy(logits, label_ids)

    avg_val_accuracy = val_accuracy / len(validation_dataloader)
    avg_val_loss = val_loss / len(validation_dataloader)
    val_accuracies.append(avg_val_accuracy)
    val_losses.append(avg_val_loss)
    print("  Average validation loss: {0:.8f}".format(avg_val_loss))
    print("  Average validation accuracy: {0:.8f}".format(avg_val_accuracy))
    print("  Training epoch took: {:}".format(format_time(time.time() - t0)))

    # if np.min(val_losses) == val_losses[-1]:
    print("saving current best checkpoint")
    torch.save(model.state_dict(), "C:/Users/s_sanghkim/Desktop/김재경/NNAP/BM25L_1101.pt")  # train 모델 저장하기

############################ 6. Inference ##########################################

test_data = pd.read_csv("C:/Users/s_sanghkim/Desktop/김재경/NNAP/test.csv")
c1 = test_data['code1'].values # 179700
c2 = test_data['code2'].values

N = test_data.shape[0] # 179700
MAX_LEN = 512

test_input_ids = np.zeros((N, MAX_LEN), dtype=int)
test_attention_masks = np.zeros((N, MAX_LEN), dtype=int)

tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
tokenizer.truncation_side = "left"

for i in tqdm(range(N), position=0, leave=True):
    try:
        cur_c1 = str(c1[i])
        cur_c2 = str(c2[i])
        encoded_input = tokenizer(cur_c1, cur_c2, return_tensors='pt', max_length=512, padding='max_length',
                                  truncation=True)
        test_input_ids[i,] = encoded_input['input_ids']
        test_attention_masks[i,] = encoded_input['attention_mask']

    except Exception as e:
        print(e)
        pass

test_input_ids = torch.tensor(test_input_ids, dtype=int)  # 179700 x 512
test_attention_masks = torch.tensor(test_attention_masks, dtype=int)  # 179700 x 512

model = AutoModelForSequenceClassification.from_pretrained("microsoft/graphcodebert-base")

model.load_state_dict(torch.load("C:/Users/s_sanghkim/Desktop/김재경/NNAP/BM25L_1101.pt"))
model.cuda()

test_tensor = TensorDataset(test_input_ids, test_attention_masks)
test_sampler = SequentialSampler(test_tensor)
test_dataloader = DataLoader(test_tensor, sampler=test_sampler, batch_size=test_batch_size)

preds = np.array([])

for step, batch in tqdm(enumerate(test_dataloader), desc="Iteration", smoothing=0.05):
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask = batch

    with torch.no_grad():
        outputs = model(b_input_ids, attention_mask=b_input_mask)

    logits = outputs[0]
    logits = logits.detach().cpu()
    _pred = logits.numpy()
    pred = np.argmax(_pred, axis=1).flatten()
    preds = np.append(preds, pred)
############################# submission ######################################
# test데이터는 코드리스트
# submission는 답지개념

submission = pd.read_csv('C:/Users/s_sanghkim/Desktop/김재경/NNAP/sample_submission.csv')
submission['similar'] = preds
submission.to_csv('C:/Users/s_sanghkim/Desktop/김재경/NNAP/submission_1101.csv', index=False)





