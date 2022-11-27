import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# PyTorch modules
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset
# transformers
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
# Util modules
from tokenized_dataset import TokenizedDataset, collate
from custom_model import *
from utils.helper_fn import *

dataset_path = "./dataset/"
models_path = "./models/"




tensor_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("USING {}.".format(tensor_device))

class CFG:
    # wandb=True
    # competition='FB3'
    # _wandb_kernel='nakama'
    debug=True 
    apex=True
    print_freq=20
    num_workers=4
    model="microsoft/deberta-v3-base"
    # gradient_checkpointing=True
    scheduler='cosine' # 'linear'か 'cosine'かを選べる
    batch_scheduler=True
    num_cycles=0.5
    num_warmup_steps=0
    n_epochs=20  # 4  # for debug
    encoder_lr=2e-5
    decoder_lr=2e-5
    min_lr=1e-6
    eps=1e-6
    betas=(0.9, 0.999)
    batch_size=2  # 8
    max_len=512
    weight_decay=0.01
    gradient_accumulation_steps= 8
    max_grad_norm=1000
    target_cols=['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
    seed=42
    # n_fold=4
    # trn_fold=[0, 1, 2, 3]
    # train=True
    device = tensor_device

if CFG.debug:
    CFG.n_epochs = 1



def train_fn(train_loader, model, criterion, optimizer, epoch, scheduler):
    model.train()
    """
    torch.cuda.ampについて 
    これは、GPU上での計算の高速化の設定。
    これを使うためには、
    - loss         e.g) Scaler.scale(loss).backward()
    - optimizer    e.g) Scaler.step(optimizer)
    をこのScalerでかぶせて
    あげる必要がある。
    """    
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.apex)  #
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0  # Gradient accumulationを行うときの、何回実際にパラメータ更新を行ったか。
    for step, (inputs, labels) in enumerate(train_loader):
        """
        inputs : token_idのシーケンスのバッチ
        labels : 評価スコアのバッチ
        """
        inputs = collate(inputs)  # ここが最大の謎。
        
        # --- send to GPU ---
        for k, v in inputs.items():
            inputs[k] = v.to(cfg.device)
        labels = labels.to(cfg.device)
        
        batch_size = labels.size(0)


        # --- forwarding ---
        with torch.cuda.amp.autocast(enabled=CFG.apex):
            y_preds = model(inputs)
            loss = criterion(y_preds, labels)

        # --- gradiation accumulation ---
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()  # ampを使うときにはScalerをlossにかぶせる
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)  # clipping
        
        # --- update parameters when gradients are accumulated ---
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scaler.step(optimizer)  # invokes optimizer.step() using the unscaled gradients
            scaler.update()  # updates the scale factor. 
            optimizer.zero_grad()  # make model parameters zeros.
            global_step += 1
            if CFG.batch_scheduler:
                scheduler.step()
        end = time.time()
        
        if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  'LR: {lr:.8f}  '
                  .format(epoch+1, step, len(train_loader), 
                          remain=timeSince(start, float(step+1)/len(train_loader)),
                          loss=losses,
                          grad_norm=grad_norm,
                          lr=scheduler.get_lr()[0]))

    return losses.avg


def valid_fn(valid_loader, model, criterion, epoch):
    ...


def train_loop():
    # --- 先にTokenizerをロードしておく
    tokenizer = AutoTokenizer.from_pretrained(CFG.model)
    tokenizer.save_pretrained('./tokenizer/')  # 保存している    

    # --- load dataset ---
    data = pd.read_csv(os.path.join(dataset_path, 'train.csv'))
    # if cfg.debug:
    #     data = data.head(16)

    # TODO: ここはKFoldに将来的に変更する
    train, valid = train_test_split(data, test_size=0.2)
    
    train_dataset = TokenizedDataset(cfg, train, tokenizer)
    valid_dataset = TokenizedDataset(cfg, valid, tokenizer)

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.batch_size,
                              shuffle=True,
                              num_workers=cfg.num_workers,
                              pin_memory=True,
                              drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=cfg.batch_size,
                              shuffle=True,
                              num_workers=cfg.num_workers,
                              pin_memory=True,
                              drop_last=True)                              
    # --- load Model ---
    model = CustomModel(cfg, use_pretrained=True)
    model.to(cfg.device)
    
    # --- Criterion ---
    criterion = nn.SmoothL1Loss(reduction='mean') # RMSELoss(reduction="mean")

    # ===============================
    #   optimizerを作って上げる
    # ===============================
    """
    手こんでるなぁ。。。
    """
    def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': weight_decay},
            {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if "model" not in n],
             'lr': decoder_lr, 'weight_decay': 0.0}
        ]
        return optimizer_parameters

    optimizer_parameters = get_optimizer_params(model,
                                                encoder_lr=CFG.encoder_lr, 
                                                decoder_lr=CFG.decoder_lr,
                                                weight_decay=CFG.weight_decay)
    optimizer = AdamW(optimizer_parameters, lr=CFG.encoder_lr, eps=CFG.eps, betas=CFG.betas)

    # ====================================================
    #   scheduler (基本的にはLrの調整)
    # ただ、Transformer専用のを使っているので、重要そう？
    # ====================================================
    def get_scheduler(cfg, optimizer, num_train_steps):
        if cfg.scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps
            )
        elif cfg.scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps, num_cycles=cfg.num_cycles
            )
        return scheduler    
    
    # !! 注意！　KFoldを使うときには、変更の必要あり。
    num_train_steps = int(CFG.n_epochs / CFG.batch_size)  # 何回更新があるか、ということ。train_foldsまで考えているのは謎。
    scheduler = get_scheduler(CFG, optimizer, num_train_steps)


    # --- start model-training ---
    for epoch in range(cfg.n_epochs):
        train_fn(train_loader,model, criterion, optimizer, epoch, scheduler)
        # valid_fn()

    # --- save model ---
    return


if __name__=="__main__":
    print("training model.")
    cfg = CFG()
    train_loop()