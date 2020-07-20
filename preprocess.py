#!/usr/bin/env python
# coding: utf-8
import glob, re, os
from tqdm import tqdm
import torch
from kobert_tokenizer import KoBertTokenizer
tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
label_dict = {'neutral': 0, 'contradiction': 1, 'entailment': 2}

if not os.path.exists('./data/KorNLUDatasets/KorNLI/processed'):
    os.mkdir('./data/KorNLUDatasets/KorNLI/processed')

# Load raw data 
kakao_nli_kor = list(open('./data/KorNLUDatasets/KorNLI/snli_1.0_train.ko.tsv'))
kakao_multi_nli_kor = list(open('./data/KorNLUDatasets/KorNLI/multinli.train.ko.tsv'))
kakao_nli_kor_train = kakao_nli_kor[1:] + kakao_multi_nli_kor[1:] # avoid repeating head
kakao_nli_kor_dev = list(open('./data/KorNLUDatasets/KorNLI/xnli.dev.ko.tsv'))
kakao_nli_kor_test = list(open('./data/KorNLUDatasets/KorNLI/xnli.test.ko.tsv'))

tr_cls_sep_sent1= []
tr_cls_sep_sent2 = []
tr_label_idx = []

dev_cls_sep_sent1= []
dev_cls_sep_sent2 = []
dev_label_idx = []

test_cls_sep_sent1= []
test_cls_sep_sent2 = []
test_label_idx = []


mode = ['tr', 'dev', 'test']

for m in mode:
    print('start processing {} file'.format(m))
    
    if m=='tr':
        save_s1, save_s2, save_label = tr_cls_sep_sent1, tr_cls_sep_sent2, tr_label_idx
    elif m== 'dev':
        save_s1, save_s2, save_label = dev_cls_sep_sent1, dev_cls_sep_sent2, dev_label_idx
    else:
        save_s1, save_s2, save_label = test_cls_sep_sent1, test_cls_sep_sent2, test_label_idx
   
    for row in tqdm(kakao_nli_kor_train):
        elements = row.split('\t')
        sent1, sent2, label = elements[0], elements[1], elements[2].replace('\n','')
        
        s1_idx = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('[CLS]'+sent1+'[SEP]'))
        s2_idx = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('[CLS]'+sent2+'[SEP]'))
        label_idx = label_dict[label]
 
        save_s1.append(s1_idx)
        save_s2.append(s2_idx)
        save_label.append(label_idx) 
        
    assert len(save_s1) == len(save_s2) ==len(save_label)
    
     
    torch.save(save_s1, './data/KorNLUDatasets/KorNLI/processed/{}_s1.pt'.format(m))
    torch.save(save_s2, './data/KorNLUDatasets/KorNLI/processed/{}_s2.pt'.format(m))
    torch.save(save_label, './data/KorNLUDatasets/KorNLI/processed/{}_label.pt'.format(m))
        
    print('save {}_s1.t, {}_s2.pt and {}_label.pt'.format(m, m, m))  
