from __future__ import absolute_import, division, print_function, unicode_literals
import glob, os
import torch
from pytorch_transformers import AdamW, WarmupLinearSchedule
from tqdm import tqdm, trange, tqdm_notebook, tnrange


class Trainer:
    def __init__(self,
                 args,
                 config,
                 model, 
                 criterion, 
                 train_dataloader, 
                 valid_dataloader,
                 logger,
                 save_path,
                 tb_writer):
        
        self.args = args
        self.config = config
        self.model = model
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.logger = logger
        self.save_path = save_path
        self.tb_writer = tb_writer
        
        self.t_total = len(self.train_dataloader)*self.args.epoch
        self.device = self.config.device
        self.optimizer = AdamW(self.get_model_parameters(), lr=self.config.learning_rate)
        self.scheduler = WarmupLinearSchedule(self.optimizer, 0.1*self.t_total, self.t_total)
           
        self.global_step = 0
        self.best_eval_acc = 0.2


    def get_model_parameters(self):
            # Optimizer & Loss
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 
             'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 
             'weight_decay': 0.0}]
        
        return optimizer_grouped_parameters
        
        
    def train(self, do_eval=True, do_save=True):
 
        for epoch in range(self.args.epoch):
            self.train_epoch(epoch)
            self.evaluation(epoch)
            self.write_to_tb()
            self.save_model(epoch)
            
        self.tb_writer.close()

        
    def transform_to_bert_input(self,batch):

        input_ids, valid_length, token_type_ids = batch[0], batch[1], batch[2]

        input_ids = torch.from_numpy(input_ids).to(self.device) 
        valid_length = valid_length.clone().detach().to(self.device)
        token_type_ids = torch.tensor(token_type_ids).long().to(self.device)

        return input_ids, valid_length, token_type_ids
    
    
    def compute_acc(self, y_hat, y, mean=True):
        if mean:
            yhat = y_hat.max(dim=-1)[1] # [0]: max value, [1]: index of max value
            acc = (yhat == y).float().mean() # padding은 acc에서 제거
            return acc
        else:
            correct_count = (yhat==y).long().sum()
            return correct_count
        
        
    def train_epoch(self, epoch):       
        self.model.to(self.device)
        self.model.train()  
    
        tr_correct_cnt, tr_total_cnt = 0,0
        tr_loss = 0.0      
        # train_loader = tqdm(self.train_dataloader)
        train_loader = self.train_dataloader

        for step, batch in enumerate(train_loader):
                      
            self.model.zero_grad()   
            
            sent1 = batch['sent1']   
            input_1, valid_length_1, token_type_1 = self.transform_to_bert_input(sent1)
            embed1 = self.model(input_1, valid_length_1, token_type_1)

            sent2 = batch['sent2']
            input_2, valid_length_2, token_type_2 = self.transform_to_bert_input(sent2)
            embed2 = self.model(input_2, valid_length_2, token_type_2)

            label = batch['label']
            label = torch.tensor(label).long().to(self.device)

            pred = self.model.get_logit(embed1, embed2)
            loss = self.criterion(pred, label.view(-1))

            tr_loss += loss.item()
            loss.backward()            
            
            if step>0 and (step) % self.config.gradient_accumulation_steps == 0:
                self.global_step += self.config.gradient_accumulation_steps

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
                
                with torch.no_grad():    
                    accuracy = self.compute_acc(pred, label)
                    
                self.tr_acc = accuracy.item()
                self.tr_avg_loss = tr_loss / self.global_step

                if self.global_step % 100==0: #int(len(self.train_dataloader)/5) ==0:
                    
                    self.logger.info('epoch : {} /{}, global_step : {} /{}, tr_avg_loss: {:.3f}, tr_acc: {:.2%}'.format(
                        epoch+1, self.args.epoch, self.global_step, self.t_total, self.tr_avg_loss, self.tr_acc))
                    

    def evaluation(self, epoch):  
        self.model.eval()
        eval_correct_cnt, eval_total_cnt = 0,0
        eval_loss = 0.0
        
        eval_acc=0.0
        eval_step=1
        
        self.logger.info('*****************Evaluation*****************')
        valid_loader = tqdm(self.valid_dataloader)
        for step, batch in enumerate(valid_loader):    
            with torch.no_grad():

                sent1 = batch['sent1']   
                input_1, valid_length_1, token_type_1 = self.transform_to_bert_input(sent1)
                embed1 = self.model(input_1, valid_length_1, token_type_1)

                sent2 = batch['sent2']
                input_2, valid_length_2, token_type_2 = self.transform_to_bert_input(sent2)
                embed2 = self.model(input_2, valid_length_2, token_type_2)

                label = batch['label']
                label = torch.tensor(label).long().to(self.device)
                pred = self.model.get_logit(embed1, embed2)

            loss = self.criterion(pred, label.view(-1))                        
            eval_loss += loss.item()
            
            acc = self.compute_acc(pred, label)
            eval_acc += acc.item()
            eval_step += 1.0

        self.eval_avg_loss = eval_loss/eval_step
        self.eval_avg_acc = eval_acc/eval_step
        
        self.logger.info('epoch : {} /{}, global_step : {} /{}, eval_loss: {:.3f}, eval_acc: {:.2%}'.format(
            epoch+1, self.args.epoch, self.global_step, self.t_total, self.eval_avg_loss, self.eval_avg_acc))                
                
    def save_model(self, epoch):
        if self.eval_avg_acc > self.best_eval_acc:
            self.best_eval_acc = self.eval_avg_acc
        
            self.model.to(torch.device('cpu'))
            state = {'epoch': epoch+1,
                     'model_state_dict': self.model.state_dict(),
                     'opt_state_dict': self.optimizer.state_dict()}

            save_model_path = '{}/epoch_{}_step_{}_tr_acc_{:.3f}_tr_loss_{:.3f}_eval_acc_{:.3f}_eval_loss_{:.3f}.pt'.format(
                        self.save_path, epoch+1, self.global_step, self.tr_acc,self.tr_avg_loss, self.eval_avg_acc,  self.eval_avg_loss)
                
                
            # Delte previous checkpoint
            if len(glob.glob(self.save_path+'/epoch*.pt'))>0:
                os.remove(glob.glob(self.save_path+'/epoch*.pt')[0])
            torch.save(state, save_model_path)
            self.logger.info(' Model saved to {}'.format(save_model_path))
            
            os.mkdir(self.save_path+'/epoch_{}_eval_loss_{:.3f}_eval_acc_{:.3f}'.format(epoch+1, self.eval_avg_loss, self.eval_avg_acc))  


    def write_to_tb(self):
        self.tb_writer.add_scalars('loss', {'train': self.tr_avg_loss, 'val': self.eval_avg_loss}, self.global_step)
        self.tb_writer.add_scalars('acc', {'train': self.tr_acc, 'val': self.eval_avg_acc}, self.global_step)