from __future__ import absolute_import
import torch

# GPU device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Model
hidden_size = 768
maxlen = 128
epoch = 50
batch_size = 32
dropout = 0.1
learning_rate = 2e-5

# Optimizer
gradient_accumulation_steps = 1
max_grad_norm = 5

# Steps
evaluate_step = 2000
save_step = 5000

# Indicies
pack_sequence = False
num_class = 3   # len(label_to_idx)
pad_idx = 1      # tok.convert_tokens_to_ids('[PAD]')
# pad_label = 21   # entitiy_to_index['[PAD]']
# o_label = 20     # entitiy_to_index['O']
