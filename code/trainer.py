## This module takes a network, a loss function, optimizer and data and trains a DNN denoiser 

import numpy as np
import torch.nn as nn
import os
import time
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.insert(0, '../code')
from network import *
from dataloader_func import weights_init_kaiming
from noise import *
from quality_metrics_func import batch_ave_psnr_torch, calc_psnr
from plotting_func import plot_loss,plot_psnr, plot_denoised_range

########################################################### util training functions ###########################################################

def show_denoised_range(model, im,noise_range, args, file_name, writer,h, im_cond):
    '''
    Takes an image, adds different levels of noise, denoises them and plots the denoised and noisy 
    '''
    model.eval()
    C, _, _= im.shape
    with torch.no_grad():
        noisy , noise = add_noise_torch_range(im, noise_range, device=args.device, coarse=args.coarse)
        if im_cond is None: 
            output = model(noisy)
        else: 
            im_cond = torch.stack([im_cond]*noise_range.shape[0])

            output = model(noisy, im_cond)
            
        if args.skip is True:            
            if args.coarse is True:
                denoised = noisy - output
                   
            else:
                denoised = noisy[:,1::] - output
        else:
            denoised = output

    if args.coarse is True:
        file_name = file_name + '.png'
        plot_denoised_range(im, noisy, denoised, noise_range, args.dir_name+ file_name, 1,writer,h)
  
    else:
        for o in range(3):
            file_name = file_name +str(o)+ '.png'
            plot_denoised_range(im, noisy[:,o+1:o+2], denoised[:,o:o+1], noise_range, args.dir_name+file_name, 1,writer,h)

def make_loader(dataset, batch_size, dataset_cond, self_cond):
    '''
    dataset is a list of tensors. Each element in the list contains image tensors of the same image class.
    dataset_cond is a list of tensors. Each element in the list contains image tensors of the same image class. This can be None. 
    self_cond: if conditioning on the same image 
    returns: 
    '''
    ## If there is no conditioner dataset, then make a normal loader object
    if dataset_cond is None: 
        if type(dataset)==list: 
            dataset = torch.cat(dataset)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        dataloader_cond = None
    ## If there is a conditioner dataset, then shuffle dataset_cond within each set, and then shuffle both dataset and 
    ## dataset_cond together. 
    else: 
        if self_cond: 
            if type(dataset)==list: 
                dataset = torch.cat(dataset)
            dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
            dataloader_cond = dataloader
            
        else: 
            if len(dataset) != len(dataset_cond): 
                raise ValueError('dataset length does not match dataset_cond')
            for i in range(len(dataset)): 
                if dataset[i].shape[0] != dataset_cond[i].shape[0]: 
                    raise ValueError('Subset '+str(i)+' in datasets does not match dataset_cond')            
                    
            # permute data globally. Preserve the permutation indices 
            dataset = torch.cat(dataset)
            idx = torch.randperm(dataset.shape[0])
            dataset = dataset[idx]
            
            # within each set in the dataset_cond shuffle images in the N dimensoin 
            dataset_cond_local_shuffled = [dataset_cond[i][torch.randperm(dataset_cond[i].shape[0])] for i in range(len(dataset_cond))]
            # concatinate all shuffled subsets into one big set 
            dataset_cond_local_shuffled = torch.cat(dataset_cond_local_shuffled)
            # shuffle the conditioing images globally according to train set global shuffle indices 
            dataset_cond_global_shuffled = dataset_cond_local_shuffled[idx]
    
            dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
            dataloader_cond = DataLoader(dataset=dataset_cond_global_shuffled, batch_size=batch_size, shuffle=False)
        
    return dataloader, dataloader_cond


########################################################### training ###########################################################
def one_iter(model, batch ,criterion, args, cond_batch):
    '''
    @batch: clean batch of input of size N,C,H,W
    @cond_batch: clean batch of conditioning images of size N,C,H,W
    '''
    
    ## clean data
    clean = batch.to(args.device)
    
    _,C,_,_ = clean.shape
    
    if args.rescale:
        clean = clean * torch.rand(size=(batch.size()[0], 1,1,1),device = args.device)
    
    ## add noise 
    noisy , noise , stds = add_noise_torch(all_patches=clean, noise_level=args.noise_level_range, sigma_dist = args.sigma_dist, coarse=args.coarse ) 
    
    # if args.loss_weight:
        # stds = noise.std(dim=(1,2,3), keepdim=True)
    
    ## denoise 
    if cond_batch is None: 
        output = model(noisy)
    else: 
        cond_batch = cond_batch.to(args.device)
        output = model(noisy,cond_batch)

    # if args.sparsify_phi and self.training: 
        # output, phis = output 
        
    ## handle skip and coarse vs fine
    if args.skip is True:        
        target = noise.detach()
        if args.coarse:
            denoised = noisy - output

        else:
            denoised = noisy[:,1::] - output
    else:
        if args.coarse is True:
            target = clean

        else:
            target = clean[1::] #C=3

        denoised = output
        
    ## compute loss

    if args.loss_weight:
        loss =  criterion((1/stds.detach()) *output, (1/stds.detach()) *target)/ (clean.size()[0])
    else: 
        # regulizer = (torch.abs(clean.size()[1] * clean.size()[2] * clean.size()[3] * stds.squeeze()**2 - (noisy - denoised).norm(dim = (2,3)).norm(dim =1)**2 )).mean()
        loss =  criterion( output,  target)/ (clean.size()[0]) 

    # if args.sparsify_phi:
        # l1_penalty = model.module.stored_x_means[3].mean()
        # loss = loss + args.lambda_l1 * l1_penalty
    
    ## compute psnr
    with torch.no_grad():
        if args.coarse is True:
            psnr = batch_ave_psnr_torch(clean, denoised ,max_I=1.)
                
        else:
            psnr = batch_ave_psnr_torch(clean[:,1::], denoised ,max_I=1.)

    return model, loss, psnr


def train_epoch(model, trainloader,criterion,optimizer, args, trainloader_cond):
    loss_sum = 0
    psnr_sum = 0
    model.train() 
    if trainloader_cond is not None: 
        trainloader_cond_list = list(enumerate(trainloader_cond))                                                
                                                                             
    for i, batch in enumerate(trainloader, 0):          
        optimizer.zero_grad()
        if trainloader_cond is None: 
            batch_cond = None
        else: 
            batch_cond = trainloader_cond_list[i][1]
        model, loss, psnr = one_iter(model, batch,criterion, args, batch_cond)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        psnr_sum += psnr.item()
                                                
    return model, loss_sum/(i+1), psnr_sum/(i+1)
                                                
                                                
def test_epoch(model, testloader,criterion, args, testloader_cond):
    loss_sum = 0
    psnr_sum = 0
    model.eval()    
    if testloader_cond is not None: 
        testloader_cond_list = list(enumerate(testloader_cond))
                                                
    with torch.no_grad():
        for i, batch in enumerate(testloader, 0):
            if testloader_cond is None: 
                batch_cond = None
            else: 
                batch_cond = testloader_cond_list[i][1]
                                                
            model, loss, psnr = one_iter(model, batch,criterion, args, batch_cond)
            loss_sum+= loss.item()
            psnr_sum += psnr.item()

                                                
    return loss_sum/(i+1), psnr_sum/(i+1)


    
def run_training(model, train_set, test_set ,criterion,optimizer, args, train_set_cond=None, test_set_cond=None):
    '''
    @train_set: either tensor of shape N,C,H,W or list of tensors of shape  N,C,H,W
    trains a denoiser neural network
    '''
    ###
    start_time_total = time.time()
    epoch_loss_list_train = []
    epoch_psnr_list_train = []
    epoch_loss_list_test = []
    epoch_psnr_list_test = []
    psnr_range_list = []
    writer = SummaryWriter(log_dir=args.dir_name)

    # Callista edit: resume from checkpoint if one exists
    start_epoch = 0
    checkpoint_path = args.dir_name + '/checkpoint.pt'
    if os.path.exists(checkpoint_path):
        print(f'Resuming from checkpoint: {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1
        epoch_loss_list_train = checkpoint['loss_train']
        epoch_psnr_list_train = checkpoint['psnr_train']
        epoch_loss_list_test = checkpoint['loss_test']
        epoch_psnr_list_test = checkpoint['psnr_test']
        print(f'Resuming from epoch {start_epoch}')

    # if train_set_cond is None:
    trainloader , trainloader_cond = make_loader(dataset=train_set, batch_size=args.batch_size, dataset_cond=train_set_cond, self_cond = args.self_cond)
    testloader, testloader_cond = make_loader(dataset=test_set, batch_size=args.batch_size, dataset_cond=test_set_cond, self_cond = args.self_cond)

    im_train = next(iter(trainloader))[0].to(args.device)
    im_test = next(iter(testloader))[0].to(args.device)
    im_train_cond= None
    im_test_cond = None

    ### loop over number of epochs
    for h in range(start_epoch, args.num_epochs):
        print('epoch ', h )
        if h >= args.lr_freq and h%args.lr_freq==0:
            for param_group in optimizer.param_groups:
                args.lr = args.lr/2
                param_group["lr"] = args.lr
        
        #train
        if train_set_cond is not None: #shuffle train conditioing images before each epoch
            trainloader, trainloader_cond = make_loader(dataset=train_set, batch_size=args.batch_size, dataset_cond = train_set_cond, self_cond = args.self_cond)
            
            if h==0:   
                testloader, testloader_cond = make_loader(dataset=test_set, batch_size=args.batch_size, dataset_cond = test_set_cond, self_cond = args.self_cond)            
                im_train = next(iter(trainloader))[0].to(args.device)
                im_test = next(iter(testloader))[0].to(args.device)                
                im_train_cond = next(iter(trainloader_cond))[0].to(args.device)
                im_test_cond = next(iter(testloader_cond))[0].to(args.device)
                
        model, epoch_loss_train, epoch_psnr_train = train_epoch(model=model, trainloader=trainloader, criterion=criterion, optimizer=optimizer, args=args, trainloader_cond =trainloader_cond)
        epoch_loss_list_train.append(epoch_loss_train)
        epoch_psnr_list_train.append(epoch_psnr_train)
        writer.add_scalar('PSNR/Train', epoch_psnr_train, global_step=h)
        # writer.add_scalar('Loss/Train', epoch_loss_train, global_step=h)
        print('train loss = ', epoch_loss_train, 'train psnr = ',epoch_psnr_train )
        
        #eval        
        epoch_loss_test, epoch_psnr_test = test_epoch(model=model, testloader=testloader, criterion=criterion, args=args, testloader_cond=testloader_cond)
        epoch_loss_list_test.append(epoch_loss_test)
        epoch_psnr_list_test.append(epoch_psnr_test)
        writer.add_scalar('PSNR/Test', epoch_psnr_test, global_step=h)
        # writer.add_scalar('Loss/Test', epoch_loss_test, global_step=h)
        print('test loss = ', epoch_loss_test, 'test psnr = ',epoch_psnr_test )
        
        #plot and save
        plot_loss(epoch_loss_list_train, epoch_loss_list_test, args.dir_name+'/loss_epoch.png')
        plot_psnr(epoch_psnr_list_train , epoch_psnr_list_test ,args.dir_name+'/psnr_epoch.png' )
        noise_range = torch.logspace(0,np.log10(args.noise_level_range[1]),10, device=args.device).reshape(10,1,1,1)
        if h%100 ==0:
            fig_writer = writer
        else: 
            fig_writer = None
        show_denoised_range(model=model, im=im_train, noise_range=noise_range, args=args, file_name='/denoised_train_image', writer=fig_writer, h=h, im_cond=im_train_cond)
        show_denoised_range(model=model, im=im_test, noise_range=noise_range, args=args, file_name='/denoised_test_image', writer=fig_writer, h=h, im_cond=im_test_cond)
        noise_range_psnr = torch.logspace(0,np.log10(args.noise_level_range[1]),10, device=args.device).reshape(10,1,1,1)
        psnr_range = calc_psnr(denoiser=model,loader=testloader, sigma_range=noise_range_psnr, device=args.device, skip=args.skip, loader_cond=testloader_cond)
        writer.add_scalars('psnr_range', psnr_range, global_step=h )
      
        
        # Callista edit: save full checkpoint after every epoch so training can resume if job is killed
        torch.save({
            'epoch': h,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'loss_train': epoch_loss_list_train,
            'psnr_train': epoch_psnr_list_train,
            'loss_test': epoch_loss_list_test,
            'psnr_test': epoch_psnr_list_test,
        }, checkpoint_path)
        # also save model weights alone for easy loading later
        if args.coarse is True:
            torch.save(model.state_dict(), args.dir_name  + '/model.pt')
        else:
            torch.save(model.state_dict(), args.dir_name  + '/model_scale'+str(args.SLURM_ARRAY_TASK_ID)+'.pt')


    print("--- %s seconds ---" % (time.time() - start_time_total))
    writer.close()
    
    return model

