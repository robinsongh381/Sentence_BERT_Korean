import torch
import torch.nn as nn
        

class SBERT(nn.Module):
    def __init__(self, config, bert_model, distill=False):
        super(SBERT, self).__init__()
        
        self.distill = distill
        self.bert = bert_model
        self.dropout = nn.Dropout(config.dropout)
        self.hidden_size = config.hidden_size
        self.num_class = config.num_class
        self.device = config.device
        self.linear = nn.Linear(3*self.hidden_size, self.num_class)
    
    def get_attention_mask(self, input_ids, valid_length):
        attention_mask = torch.zeros_like(input_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()
    
    def forward(self, input_ids, valid_length, token_type_ids=None):
        """
        Input tensor
            input_ids      : (batch, maxlen)
            valid_length   : (batch)
            token_type_ids : (batch, maxlen)
        
        """
        
        attention_mask = self.get_attention_mask(input_ids, valid_length)

        if self.distill:
            outputs = self.bert(input_ids=input_ids.long(), attention_mask=attention_mask)
            all_encoder_layers = outputs[0]
            
        else:
            all_encoder_layers, pooled_output = self.bert(input_ids=input_ids.long(),
                                                      token_type_ids=token_type_ids,
                                                      attention_mask=attention_mask)
        # (b, maxlen, hid_dim) -> (b, hid_dm)    
        mean_pooled_output = torch.mean(all_encoder_layers, 1) 
        drop = self.dropout(mean_pooled_output)
        
        return drop
        
        
    def get_logit(self, embed1, embed2, mode='diff'):
        
        if mode == None:
            linear = nn.Linear(2*self.hidden_size, self.num_class)
            final_embed = torch.cat([embed1, embed2], 1)
            pred = linear(final_embed)
        
        if mode == 'diff':
            embed3 = torch.abs(embed1-embed2)
            final_embed = torch.cat([embed1, embed2, embed3], 1)
            pred = self.linear(final_embed)
            
        return pred
        