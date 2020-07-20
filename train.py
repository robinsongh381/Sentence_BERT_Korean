from __future__ import absolute_import, division, print_function, unicode_literals

import argparse, random, os, logging, glob
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange, tqdm_notebook, tnrange

import constant as config
from trainer import Trainer
from model import SBERT
from dataset import SBERTDataset, pad_collate
from logger import logger, init_logger

from kobert.pytorch_kobert import get_pytorch_kobert_model
from transformers import DistilBertModel
##
device = config.device

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # log file
    parser.add_argument('-log_dir', default='./result')
    parser.add_argument('-log_folder', default='/{}_distill_{}_batch_{}_epoch_{}') 
    parser.add_argument('-dname', default='mnli', choices=['nli','mnli','nli+mnli'])
    parser.add_argument('-is_distill', default='True')
    
    # model config
    parser.add_argument("-epoch", default=config.epoch, type=int)
    parser.add_argument("-batch_size", default=config.batch_size, type=int)
    args = parser.parse_args()
    
    log_path = args.log_dir + args.log_folder.format(args.dname, args.is_distill, args.batch_size, args.epoch)
    print('log_path : {}'.format(log_path))
    init_logger(log_path,'/log/log.txt')
    
    # Tensorboard Writer
    tb_writer = SummaryWriter('{}/runs'.format(log_path))
    
    # Load dataset
    print('Data loading...')
    train_dataset = SBERTDataset('tr', args.dname)
    train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              collate_fn=pad_collate,
                              drop_last=True,
                              num_workers=0)
    train_examples_len = len(train_dataloader)
    
    valid_dataset = SBERTDataset('test', args.dname)
    valid_dataloader = DataLoader(dataset=valid_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              collate_fn=pad_collate,
                              drop_last = True,
                              num_workers=0)  
    valid_examples_len = len(valid_dataloader)    
    
    
    # Build model & Criterion
    if args.is_distill:
        kobert = DistilBertModel.from_pretrained('monologg/distilkobert')
        logger.info('Loading Distilled Bert...')
    else:
        kobert, _ = get_pytorch_kobert_model()
        logger.info('Loading full-layer Bert...')
        
    model = SBERT(config=config, bert_model=kobert, distill=args.is_distill)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Define Trainer 
    trainer = Trainer(
                args=args,
                config=config,
                model=model,
                criterion=criterion,
                train_dataloader=train_dataloader,
                valid_dataloader=valid_dataloader,
                logger=logger,
                save_path=log_path,
                tb_writer=tb_writer)
    
    logger.info('Train Start...')
    trainer.train()
    
    logger.info('Train finished !')