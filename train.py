import os
import sys
import time
import random
import string
import argparse
from collections import namedtuple
import copy

import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
from torch import autograd
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.nn.parallel import DistributedDataParallel as pDDP

from torchsummary import summary
from torchvision.utils import save_image
import horovod.torch as hvd
import gin

import numpy as np
from tqdm import tqdm, trange
from PIL import Image

import apex
from apex.parallel import DistributedDataParallel as aDDP
from apex.fp16_utils import *
from apex import amp
from apex.multi_tensor_apply import multi_tensor_applier

import wandb
import ds_load

from utils import CTCLabelConverter, Averager, ModelEma, Metric
from cnv_model import OrigamiNet, ginM
from test import validation

parOptions = namedtuple('parOptions', ['DP', 'DDP', 'HVD'])
parOptions.__new__.__defaults__ = (False,) * len(parOptions._fields)

pO = None
OnceExecWorker = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_bn(model):
    if type(model) in [torch.nn.InstanceNorm2d, torch.nn.BatchNorm2d]:
        init.ones_(model.weight)
        init.zeros_(model.bias)

    elif type(model) in [torch.nn.Conv2d]:
        init.kaiming_uniform_(model.weight)
    
def WrkSeeder(_):
    return np.random.seed((torch.initial_seed()) % (2 ** 32))

@gin.configurable
def train(opt, AMP, WdB, train_data_path, train_data_list, test_data_path, test_data_list, experiment_name, 
            train_batch_size, val_batch_size, workers, lr, valInterval, num_iter, wdbprj, continue_model=''):

    HVD3P = pO.HVD or pO.DDP

    os.makedirs(f'./saved_models/{experiment_name}', exist_ok=True)

    if OnceExecWorker and WdB:
        wandb.init(project=wdbprj, name=experiment_name)
        wandb.config.update(opt)
    
    train_dataset = ds_load.myLoadDS(train_data_list, train_data_path)
    valid_dataset = ds_load.myLoadDS(test_data_list, test_data_path , ralph=train_dataset.ralph)

    if OnceExecWorker:
        print(pO)
        print('Alphabet :',len(train_dataset.alph),train_dataset.alph)
        for d in [train_dataset, valid_dataset]:
            print('Dataset Size :',len(d.fns))
            print('Max LbW : ',max(list(map(len,d.tlbls))) )
            print('#Chars : ',sum([len(x) for x in d.tlbls]))
            print('Sample label :',d.tlbls[-1])
            print("Dataset :", sorted(list(map(len,d.tlbls))) )
            print('-'*80)
    
    if opt.num_gpu > 1:
        workers = workers * ( 1 if HVD3P else opt.num_gpu )

    if HVD3P:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=opt.world_size, rank=opt.rank)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset, num_replicas=opt.world_size, rank=opt.rank)

    train_loader  = torch.utils.data.DataLoader( train_dataset, batch_size=train_batch_size, shuffle=True if not HVD3P else False, 
                    pin_memory = True, num_workers = int(workers),
                    sampler = train_sampler if HVD3P else None,
                    worker_init_fn = WrkSeeder,
                    collate_fn = ds_load.SameTrCollate
                )

    valid_loader  = torch.utils.data.DataLoader( valid_dataset, batch_size=val_batch_size , pin_memory=True, 
                    num_workers = int(workers), sampler=valid_sampler if HVD3P else None)
    
    model = OrigamiNet()
    model.apply(init_bn)
    model.train()

    if OnceExecWorker: import pprint;[print(k,model.lreszs[k]) for k in sorted(model.lreszs.keys())]

    biparams    = list(dict(filter(lambda kv: 'bias'     in kv[0], model.named_parameters())).values())
    nonbiparams = list(dict(filter(lambda kv: 'bias' not in kv[0], model.named_parameters())).values())

    if not pO.DDP:
        model = model.to(device)
    else:
        model.cuda(opt.rank)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=10**(-1/90000))

    if OnceExecWorker and WdB:
        wandb.watch(model, log="all")

    if pO.HVD:
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
        # optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters(), compression=hvd.Compression.fp16)
    
    if pO.DDP and opt.rank!=0:
        random.seed()
        np.random.seed()

    if AMP:
        model, optimizer = amp.initialize(model, optimizer, opt_level = "O1")
    if pO.DP:
        model = torch.nn.DataParallel(model)
    elif pO.DDP:
        model = pDDP(model, device_ids=[opt.rank], output_device=opt.rank,find_unused_parameters=False)

    
    
    model_ema = ModelEma(model)

    if continue_model != '':
        if OnceExecWorker: print(f'loading pretrained model from {continue_model}')
        checkpoint = torch.load(continue_model, map_location=f'cuda:{opt.rank}' if HVD3P else None)
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        model_ema._load_checkpoint(continue_model, f'cuda:{opt.rank}' if HVD3P else None)

    criterion = torch.nn.CTCLoss(reduction='none', zero_infinity=True).to(device)
    converter = CTCLabelConverter(train_dataset.ralph.values())
    
    if OnceExecWorker:
        with open(f'./saved_models/{experiment_name}/opt.txt', 'a') as opt_file:
            opt_log = '------------ Options -------------\n'
            args = vars(opt)
            for k, v in args.items():
                opt_log += f'{str(k)}: {str(v)}\n'
            opt_log += '---------------------------------------\n'
            opt_log += gin.operative_config_str()
            opt_file.write(opt_log)
            if WdB:
                wandb.config.gin_str = gin.operative_config_str().splitlines()


        print(optimizer)
        print(opt_log)

    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = 1e+6
    best_CER = 1e+6
    i = 0
    gAcc = 1
    epoch = 1
    btReplay = False and AMP
    max_batch_replays = 3

    if HVD3P: train_sampler.set_epoch(epoch)
    titer = iter(train_loader)

    while(True):
        start_time = time.time()

        model.zero_grad()
        train_loss = Metric(pO,'train_loss')

        for j in trange(valInterval, leave=False, desc='Training'):

            try:
                image_tensors, labels = next(titer)
            except StopIteration:
                epoch += 1
                if HVD3P: train_sampler.set_epoch(epoch)
                titer = iter(train_loader)
                image_tensors, labels = next(titer)
                
            image = image_tensors.to(device)
            text, length = converter.encode(labels)
            batch_size = image.size(0)

            replay_batch = True
            maxR = 3
            while replay_batch and maxR>0:
                maxR -= 1
                
                preds = model(image,text).float()
                preds_size = torch.IntTensor([preds.size(1)] * batch_size).to(device)
                preds = preds.permute(1, 0, 2).log_softmax(2)
                
                if i==0 and OnceExecWorker:
                    print('Model inp : ',image.dtype,image.size())
                    print('CTC inp : ',preds.dtype,preds.size(),preds_size[0])

                # To avoid ctc_loss issue, disabled cudnn for the computation of the ctc_loss
                torch.backends.cudnn.enabled = False
                cost = criterion(preds, text.to(device), preds_size, length.to(device)).mean() / gAcc
                torch.backends.cudnn.enabled = True

                train_loss.update(cost)

                optimizer.zero_grad()
                default_optimizer_step = optimizer.step  # added for batch replay

                if not AMP:
                    cost.backward()
                    replay_batch = False
                else:
                    with amp.scale_loss(cost, optimizer) as scaled_loss:
                        scaled_loss.backward()
                        if pO.HVD: optimizer.synchronize()

                    if optimizer.step is default_optimizer_step or not btReplay:
                        replay_batch = False
                    elif maxR>0:
                        optimizer.step()
                    
                    
            if btReplay: amp._amp_state.loss_scalers[0]._loss_scale = mx_sc
            
            if (i+1) % gAcc == 0:

                if pO.HVD and AMP:
                    with optimizer.skip_synchronize(): 
                        optimizer.step()
                else:
                    optimizer.step()
                
                model.zero_grad()
                model_ema.update(model, num_updates=i/2)

                if (i+1) % (gAcc*2) == 0:
                    lr_scheduler.step()
            
            i += 1
            
        # validation part
        if True:

            elapsed_time = time.time() - start_time
            start_time = time.time()

            model.eval()
            with torch.no_grad():

                
                valid_loss, current_accuracy, current_norm_ED, ted, bleu, preds, labels, infer_time = validation(
                    model_ema.ema, criterion, valid_loader, converter, opt, pO)
        
            model.train()
            v_time = time.time() - start_time

            if OnceExecWorker:                
                if current_norm_ED < best_norm_ED:
                    best_norm_ED = current_norm_ED
                    checkpoint = {
                        'model': model.state_dict(),
                        'state_dict_ema': model_ema.ema.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
                    torch.save(checkpoint, f'./saved_models/{experiment_name}/best_norm_ED.pth')

                if ted < best_CER:
                    best_CER = ted
                
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy

                out  = f'[{i}] Loss: {train_loss.avg:0.5f} time: ({elapsed_time:0.1f},{v_time:0.1f})'
                out += f' vloss: {valid_loss:0.3f}'
                out += f' CER: {ted:0.4f} NER: {current_norm_ED:0.4f} lr: {lr_scheduler.get_lr()[0]:0.5f}'
                out += f' bAcc: {best_accuracy:0.1f}, bNER: {best_norm_ED:0.4f}, bCER: {best_CER:0.4f}, B: {bleu*100:0.2f}'
                print(out)

                with open(f'./saved_models/{experiment_name}/log_train.txt', 'a') as log: log.write(out + '\n')

                if WdB:
                    wandb.log({'lr': lr_scheduler.get_lr()[0], 'It':i, 'nED': current_norm_ED,  'B':bleu*100,
                    'tloss':train_loss.avg, 'AnED': best_norm_ED, 'CER':ted, 'bestCER':best_CER, 'vloss':valid_loss})

        if i == num_iter:
            print('end the training')
            sys.exit()

def gInit(opt):
    global pO, OnceExecWorker
    gin.parse_config_file(opt.gin)
    pO = parOptions(**{ginM('dist'):True})

    if pO.HVD:
        hvd.init()
        torch.cuda.set_device(hvd.local_rank())

    OnceExecWorker = (pO.HVD and hvd.rank() == 0) or (pO.DP)
    cudnn.benchmark = True


def rSeed(sd):
    random.seed(sd)
    np.random.seed(sd)
    torch.manual_seed(sd)
    torch.cuda.manual_seed(sd)

def launch_fn(rank, opt):
    global OnceExecWorker
    gInit(opt)
    OnceExecWorker = OnceExecWorker or (pO.DDP and rank==0)
    mp.set_start_method('fork', force=True)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(opt.port)

    dist.init_process_group("nccl", rank=rank, world_size=opt.num_gpu)

    #to ensure identical init parameters
    rSeed(opt.manualSeed)

    torch.cuda.set_device(rank)
    opt.world_size = opt.num_gpu
    opt.rank       = rank

    train(opt)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gin', help='Gin config file')

    opt = parser.parse_args()
    gInit(opt)
    opt.manualSeed = ginM('manualSeed')
    opt.port = ginM('port')

    if OnceExecWorker:
        rSeed(opt.manualSeed)

    opt.num_gpu = torch.cuda.device_count()
    

    if pO.HVD:
        opt.world_size = hvd.size()
        opt.rank       = hvd.rank()
    
    if not pO.DDP:
        train(opt)
    else:
        mp.spawn(launch_fn, args=(opt,), nprocs=opt.num_gpu)