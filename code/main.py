import numpy as np
import torch.nn as nn
import os
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, ConcatDataset
import torchvision
import torchvision.transforms as transforms
import argparse
import sys
sys.path.insert(0, '../code')
from network import *
from model_loader_func import initialize_network
from trainer import run_training
from dataloader_func import weights_init_kaiming 
from  synthetic_data_generators import oval_dataset, generate_circles_texture, generate_squares_texture, generate_ovals_texture, generate_object_texture_mix
import pickle

####################################################################################################################
########################################################## Experiment specific functions #######################
####################################################################################################################
def build_path(args):
    '''
    build the path to save results of trainig. 
    General pattern is: architecture name, data name, noise level, etc 
    This should change depending on specific of path names
    '''
    dir_name = args.dir_name + args.arch_name 

    
    dir_name = dir_name + '/'+ args.data_name + '/'+str(args.noise_level_range[0])+'to'+ str(args.noise_level_range[1]) 
    if args.RF is not None: 
        dir_name = dir_name + '_RF_'+str(args.RF)+'x'+str(args.RF) 

    if args.set_size is not None: 
        dir_name = dir_name + '_set_size_' + str(args.set_size)
    if args.swap is True:
        dir_name = dir_name + '_swapped'
    
    if args.optional_dir_label is not None: 
        dir_name = dir_name + '_'+ args.optional_dir_label
        
    return dir_name


####### data prep functions for different datasets #########

    
def load_prep_specific_class(args): 
    '''
    '''
    args.data_name = ['img_align_celeba', 'bedroom', 'nabirds', 'zk-wood-textures'][args.SLURM_ARRAY_TASK_ID]
    args.data_path = args.data_root_path + args.data_name
    
    train_set = []
    test_set = []

    data = torch.load( args.data_path +'/train_80x80.pt', weights_only=True)    
    
    if args.debug: 
        N = args.batch_size
    else: 
        # if data.shape[0] > 120000: 
        #     N = 130000
        # else: 
        N = data.shape[0] - 1000  

    train_set = [data[0:N]]    
    test_set = [data[N::]]
        
    return train_set, test_set



    
def load_prep_texture(args, color, dim=80, n_test=4): 
    '''
    returns lists of tensors. Each tensor in the list contains all patches from the same image. 
    '''
    # ### for 80x80 patches
    args.data_path = args.data_root_path + 'texture_EPS'
    if dim==80:
        if color: 
            data = torch.load( args.data_path + '/patched_1024x1024_to_80x80_all_sets_color.pt', weights_only=True)
            n_patches = int(1024/dim)**2 # 144
            
        else:     
            data1 = torch.load( args.data_path + '/patched_1024x1024_to_80x80.pt', weights_only=True)
            data2 = torch.load( args.data_path + '/patched_1024x1024_to_80x80_down.pt', weights_only=True)    
            data = torch.cat([data1, data2])
            n_patches = int(1024/dim)**2 # 144
            
    ## for 128x128 patches             
    if dim==128: 
        if color: 
            data = torch.load( args.data_path + '/patched_1024x1024_to_128x128_all_sets_color.pt', weights_only=True)
            n_patches = int(1024/dim)**2 # 64
            
        else: 
            data = torch.load( args.data_path + '/patched_1536x1536_to_128x128.pt', weights_only=True)
            n_patches = int(1536/dim)**2 # 144
            
    num_classes = int(data.shape[0]/n_patches) # for 80x80: 882 + 786 # for 128x128:786
    
    all_data_train = []
    all_data_test = []
    
    if args.debug: 
        num_classes = 5
    for d in range(num_classes): 
        all_data_train.append(data[n_patches * d: (n_patches * (d+1)) -n_test ])
        all_data_test.append(data[(n_patches * (d+1)) -n_test : (n_patches * (d+1)) ] ) # in each image, leave the last 4 patches for test set       

    if color is False:
        # add my wood texture images     
        data_zk = torch.load( args.data_root_path + '/zk-wood-textures/train_80x80.pt', weights_only=True)
        all_data_train.append(data_zk[0:n_patches-n_test]) #the first 140 patches of  the entire dataset 
        all_data_test.append(data_zk[-n_test::]) #the last 4 of the entire dataset 
    
    return all_data_train, all_data_test





####################################################################################################################
################################################# main #################################################
####################################################################################################################

def main():
    main_parser = argparse.ArgumentParser(add_help=False,description='set up arguments for training a DNN denoiser')
    
    main_parser.add_argument('--arch_name', type=str, default= 'UNet_flex') ### !!choose for run!! ###  
    
    ######### optimization variables #########
    main_parser.add_argument('--lr', type=float, default=0.001)
    main_parser.add_argument('--batch_size', type=int, default=512)
    main_parser.add_argument('--num_epochs', type=int, default=1000)
    main_parser.add_argument('--lr_freq', type=int, default=100)
    main_parser.add_argument('--loss_weight', default=False) ### !!choose for run!! ###  
    
    
    ######### dataset variables #########
    main_parser.add_argument('--noise_level_range', default= [1, 3 * 255])
    main_parser.add_argument('--sigma_dist', default = 'inv_sqrt' )     
    main_parser.add_argument('--rescale', default=False ,help='rescale intensities. Do not rescale for conditional denoisers.')
    main_parser.add_argument('--swap', default=False)
    main_parser.add_argument('--set_size', default=100_000)
    main_parser.add_argument('--imagenet_subset_ids', default=None)
    
    ######### directory-related variables #########
    # main_parser.add_argument('--data_name', type=str , default = 'multi_class_dataset')  ### !!choose for run!! ###
    # main_parser.add_argument('--data_name', type=str , default = 'nano_imagenet')  ### !!choose for run!! ###
    main_parser.add_argument('--data_name', type=str , default = 'imagenet')  ### !!choose for run!! ###
    # main_parser.add_argument('--data_root_path', default= '/mnt/home/zkadkhodaie/ceph/datasets/')
    main_parser.add_argument('--data_root_path', default= '/mnt/home/gkrawezik/ceph/AI_DATASETS/ImageNet/2012/')    
    main_parser.add_argument('--dir_name', default= '/mnt/home/zkadkhodaie/ceph/22_representation_in_UNet_denoiser/denoisers/', help='folder where outputs will be saved (modify accordingly)')
    main_parser.add_argument('--optional_dir_label', default='color_no_skip_deep_dec_inv_sqrt', help='will be added to denoiser type when building dir name') ### choose for run ###  
    
    ######### other variables #########
    main_parser.add_argument('--device', type=str, default='cuda')
    main_parser.add_argument('--debug', default= False)
    #main_parser.add_argument('--SLURM_ARRAY_TASK_ID',type=int) ### !!choose for run!! ###  

    ######### architecture variables shared between networks #########
    main_parser.add_argument('--kernel_size', default= 3)
    main_parser.add_argument('--padding', default= 1)
    main_parser.add_argument('--skip', default= False)
    main_parser.add_argument('--num_channels', help='set 1 for grayscale and 3 for color')
    main_parser.add_argument('--bias', default=True)
    main_parser.add_argument('--RF' , help = 'Receptive field of the network. For BF_CNN_RF only values in this set {5,9,13, 23, 43}')
    main_parser.add_argument('--coarse', default = True, help = 'putting this here because it is called in trainer.py. remove later!')
    main_parser.add_argument('--self_cond', default=False, help ='condition on the same image')

    main_args, _ = main_parser.parse_known_args()
    
    ######### architecture variables specific to each arch #########
    parser = argparse.ArgumentParser(parents=[main_parser])
    if main_args.arch_name== 'UNet_flex_v2': 
        parser.add_argument('--num_kernels', default= [64,128, 256, 512],help='list of len num_blocks+1')
        parser.add_argument('--num_blocks',type=int, help='this will be inferred from num_kernels len')    
        parser.add_argument('--num_enc_conv', default= [2,2,2], help='min is 2')  
        parser.add_argument('--num_mid_conv', default= 3, help='min is 2')  
        parser.add_argument('--num_dec_conv', default= [6,6,6], help='min is 2') 
        parser.add_argument('--NormType', default= 'LayerNorm') ## choose for run
        parser.add_argument('--dilations', default= [2,4,6,8]) ## choose for run
        
    if main_args.arch_name== 'UNet_flex': 
        parser.add_argument('--num_kernels', default= [64,128, 256, 512],help='list of len num_blocks+1')
        parser.add_argument('--num_blocks',type=int, help='this will be inferred from num_kernels len')    
        parser.add_argument('--num_enc_conv', default= [2,2,2], help='min is 2')  
        parser.add_argument('--num_mid_conv', default= 3, help='min is 2')  
        parser.add_argument('--num_dec_conv', default= [6, 6,6], help='min is 2') 
        parser.add_argument('--NormType', default= 'LayerNorm') ## choose for run
        parser.add_argument('--inter_skip', default= True) ## choose for run
        parser.add_argument('--dilations', default= None) ## choose for run
        parser.add_argument('--upsample_with_bias', default= False)
        # parser.add_argument('--normalize_phi', default= False) ##
        # parser.add_argument('--sparsify_phi', default= False) ##  choose for run
        # parser.add_argument('--lambda_l1' ) ##
        
    elif main_args.arch_name=='UNet_conditional_mean_matching': 
        parser.add_argument('--num_kernels', default= [64,128,256, 512],help='list of len num_blocks+1')
        parser.add_argument('--num_blocks',type=int, help='this will be inferred from num_kernels len')    
        parser.add_argument('--num_enc_conv', default= [4,4,4,4], help='min is 2')  
        parser.add_argument('--num_mid_conv', default= 4, help='min is 2')  
        parser.add_argument('--num_dec_conv', default= [6,6,6,6], help='min is 2') 
        parser.add_argument('--NormType', default= 'LayerNorm') ## choose for run
        parser.add_argument('--match_only_mid', default= True, help='If True, only match means in the mid layer') ## choose for run
        parser.add_argument('--nonlinear_enc', default= True ) 
        parser.add_argument('--upsample_with_bias', default= False) 
        parser.add_argument('--match_std', default= False) 
        
    elif main_args.arch_name== 'UNet':
        parser.add_argument('--num_kernels', default= 64,help='list of len num_blocks+1')
        parser.add_argument('--num_blocks',default = 3, type=int, help='this will be inferred from num_kernels len')
        parser.add_argument('--num_enc_conv', default= 2, help='min is 2')
        parser.add_argument('--num_mid_conv', default= 2, help='min is 2')
        parser.add_argument('--num_dec_conv', default= 2, help='min is 2')

    elif main_args.arch_name== 'BF_CNN': 
        parser.add_argument('--first_layer_linear', default= False, help='For BF_CNN model')    
        parser.add_argument('--num_layers', default= 20)  
        parser.add_argument('--num_kernels', default=64)
        
    elif main_args.arch_name== 'BF_CNN_RF': 
        parser.add_argument('--num_layers', default= 21)  
        parser.add_argument('--num_kernels', default=64)        
        parser.add_argument('--coarse', default = True, help = 'For BF_CNN_RF model. Denoiser for coarse or fine coefficients')
        parser.add_argument('--j', default = 0, type=int, help='scale for the multi-scale data. Strats from 0')

    args = parser.parse_args()  
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
           
    #################################################################################

    # if args.SLURM_ARRAY_TASK_ID == 1: 
    #     args.swap = True
    # elif args.SLURM_ARRAY_TASK_ID == 2: 
    #     args.swap = False

    
    ######### load raw data #########

    if args.debug: 
        args.num_epochs=3

    
    if 'texture' in args.data_name.split('_'):
        train_set, test_set = load_prep_texture(args, color=True,dim=80, n_test=4)
    
    elif args.data_name == 'imagenet':
        args.data_path = args.data_root_path + args.data_name

        if args.debug:
            print('debug mode')
            # Callista edit: standardized filenames to 80x80
            # load first 3 classes from val set as a tiny train set for quick sanity checking (avoids loading large train file)
            train_set = torch.load( args.data_path + '/test_80x80_color_list.pt', weights_only=True) [0:3]
            # load next 3 classes from val set as test set
            test_set = torch.load( args.data_path + '/test_80x80_color_list.pt', weights_only=True) [3:6]
        else:
            # Callista edit: standardized filenames to 80x80
            # load full training set
            train_set = torch.load( args.data_path + '/train_80x80_color_list.pt', weights_only=True)
            # load full val set
            test_set = torch.load( args.data_path + '/test_80x80_color_list.pt', weights_only=True)    
            # train_set, test_set = load_imagenet_subset(args, 200)
    
    elif args.data_name == 'face_bedroom': 
        train_set, test_set = load_prep_half_bed_half_face(args)

    elif args.data_name == 'multi_class_dataset': 
        train_set, test_set = load_prep_multi_class_data(args)

    elif args.data_name == 'bedroom': 
        data = torch.load( args.data_root_path  + '/bedroom/train_color_80x80.pt', weights_only=True)    
        train_set = [data[0:150_000]]
        test_set = [data[0:100]]

    elif args.data_name == 'img_align_celeba': 
        data = torch.load( args.data_root_path  + '/img_align_celeba/train_color_80x80.pt', weights_only=True)    
        train_set = [data[0:150_000]]
        test_set = [data[0:100]]
        
    args.set_size = torch.cat(train_set).shape[0]
    image_size = torch.cat(train_set).shape 
    args.num_channels = image_size[1] #set number of input channels
    print('train data size: ', image_size )
    



    ######### initialize a model #########
    model = initialize_network(args.arch_name, args)
    args.RF = model.RF    
    model.apply(weights_init_kaiming)
    if torch.cuda.is_available():
        print('[ Using CUDA ]')
        model = nn.DataParallel(model).cuda()
    print('number of parameters is ' , sum(p.numel() for p in model.parameters() if p.requires_grad))

        
    ######### build path #########
    args.dir_name = build_path(args) 
    args.dir_name = args.dir_name + '_'+str(image_size[2]) +'x'+ str(image_size[3])
    if not os.path.exists(args.dir_name):
        os.makedirs(args.dir_name)
        
    print(args.dir_name)
    ######### select criterion and optimizer #########
    criterion = nn.MSELoss(reduction='sum')
    optimizer = Adam(filter(lambda p: p.requires_grad,model.parameters()), lr = args.lr)

    ## save model args in case 
    with open( args.dir_name +'/exp_arguments.pkl', 'wb') as f:
        pickle.dump(args.__dict__, f)

   
    ######## train #########  
    model = run_training(model=model, 
                         train_set=train_set, 
                         test_set=test_set, 
                         criterion=criterion, 
                         optimizer=optimizer, 
                         args=args, 
                         train_set_cond=None, 
                         test_set_cond=None) 


if __name__ == "__main__" :
    main()


