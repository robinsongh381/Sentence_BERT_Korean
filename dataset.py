from __future__ import absolute_import, division, print_function, unicode_literals
import torch
from torch.utils.data import Dataset
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences
import constant as model_config


# device = model_config.device


class SBERTDataset(Dataset): 
    def __init__(self,dtype, dname):
        
        
        self.sent1 = torch.load('./data/KorNLUDatasets/KorNLI/processed/{}_s1.pt'.format(dtype))
        self.sent2 = torch.load('./data/KorNLUDatasets/KorNLI/processed/{}_s2.pt'.format(dtype))
        self.label = torch.load('./data/KorNLUDatasets/KorNLI/processed/{}_label.pt'.format(dtype))
        
        assert len(self.sent1)==len(self.sent2)==len(self.label)
        self.length = len(self.label)
        
    def __getitem__(self, idx):
        return self.sent1[idx], self.sent2[idx], self.label[idx]
    
    def __len__(self):
        return self.length
    
    
    
def pad_collate(batch):

    (xx, yy, zz) = zip(*batch)
    
    # first sentence
    len_s1 = [len(x) for x in xx]
    max_token_len_s1 = max(len_s1)
    
    input_mask_s1 = torch.tensor(len_s1) # valid length
    token_ids_s1 = pad_sequences(xx, 
                              maxlen=max_token_len_s1, # model_config.maxlen 
                              value=model_config.pad_idx, 
                              padding='post',
                              dtype='long',
                              truncating='post')
    
    segment_ids_s1 = [len(i)*[0] for i in token_ids_s1]
    
    # sencond sentence
    len_s2 = [len(y) for y in yy]
    max_token_len_s2 = max(len_s2)
    # print('max s2 : {}'.format(max_token_len_s2))
    
    input_mask_s2 = torch.tensor(len_s2)
    token_ids_s2 = pad_sequences(yy, 
                              maxlen=max_token_len_s2, # model_config.maxlen 
                              value=model_config.pad_idx, 
                              padding='post',
                              dtype='long',
                              truncating='post')
    
    segment_ids_s2 = [len(i)*[0] for i in token_ids_s2]


    feature = {'sent1':[token_ids_s1, input_mask_s1, segment_ids_s1],
               'sent2':[token_ids_s2, input_mask_s2, segment_ids_s2],
               'label': zz
              }
    
    return feature



def transform_to_bert_input(tokenized_idx_with_cls_sep, device):
        token_ids = pad_sequences(tokenized_idx_with_cls_sep, 
                                  maxlen=model_config.maxlen,
                                  value=model_config.pad_idx, 
                                  padding='post',
                                  dtype='long',
                                  truncating='post')

        valid_length = torch.tensor([len(tokenized_idx_with_cls_sep[0])]) # .long()
        segment_ids = [len(tokenized_idx_with_cls_sep[0])*[0]]

        # torch-compatible format
        token_ids = torch.tensor(tokenized_idx_with_cls_sep).float().to(device)
        valid_length = valid_length.clone().detach().to(device)
        segment_ids = torch.tensor(segment_ids).long().to(device)
        
        return token_ids, valid_length, segment_ids

