import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import torch.nn as nn
import torch
import os
from PIL import Image
import gzip
from matplotlib.patches import Circle, Polygon
import random
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader  
import math
import time 
######################################################################################
################################ data loaders ################################
######################################################################################

def load_dataset( folder_path, s=None,n_images=None, crop=True, shuffle=True):
    '''
    @s: if float (0,1), new size will be computed. If tuple, that would be the new size     
    return images in a tensor. Output array is in dims: B, H, W, C
    @crop: crops the image if it is rectangular to the largest square
    '''
    images = []    
    names = os.listdir(folder_path)
    names.sort()    
    if n_images is None: 
        n_images = len(names)

    n = 0
    m = 0
    # Callista edit: original condition "n-m < n_images" crashes when a folder contains grayscale images.
    # Each grayscale image increments m (skipped count), so the loop tries to go beyond the end of the file list.
    # Adding "n < len(names)" stops the loop when files run out, returning however many RGB images were found.
    while n-m < n_images and n < len(names):
        im = plt.imread(folder_path + names[n])

        if crop is True: 
            im = center_crop(im)
        
        if s is not None:
            im =  resize_image(im, s)
        if len(im.shape) == 3:
            images.append(im);
        else: 
            m+=1
            
        n=n+1     

    images = torch.tensor(np.array(images)).permute(0,3,1,2) #list to tensor
    if images.dtype == torch.uint8: 
        images = images/255
    if shuffle is True: 
        idx = torch.randperm(images.shape[0])
        images = images[idx]
        
    return images



def load_nested_dataset( folder_path, s=None,n_folders=None,n_images=None,crop=True):
    '''
    @one_tensor: it mixes the images of different folders.
    @folder_path takes path to either train or val folders. 
    @s:float between 0 and 1 (or larger) to resize the image 
    @n: number of images taken from each original folders 
    return images in one numpy array. Output array is in dims: B, H, W, C
    '''
    
    ### prepare transfomrms 
    # Dynamically crop to square using the smaller dimension
    square_crop = transforms.Lambda(
        lambda img: transforms.functional.center_crop(
            img, min(img.size)  # img.size = (width, height)
        )
    )
    go_rgb = transforms.Lambda(lambda img: img.convert("RGB"))  # ensures 3 channels

    ## put transforms together 
    my_transforms = [transforms.ToTensor()]
    if s is not None: 
        my_transforms.insert(0, transforms.Resize(s, transforms.InterpolationMode.BICUBIC))
    if crop: 
        my_transforms.insert(0, square_crop)

    my_transforms.insert(0, go_rgb)    
    transform = transforms.Compose(my_transforms)
    
    start_time = time.time()
    
    #### get forlder names 
    images = []
    folder_names = os.listdir(folder_path )
    folder_names = [name for name in folder_names if os.path.isdir(os.path.join(folder_path, name))]    
    folder_names.sort()
    if n_folders is not None: 
        folder_names = folder_names[0:n_folders]

    # go through folders 
    for folder in folder_names:
        
        names = os.listdir(folder_path + '/' + folder + '/' )
        names.sort()
        if n_images is not None: 
            names = names[0:n_images]

        temp = [transform(default_loader(folder_path + '/' + folder + '/'+ path)) for  path in names ]
        temp = torch.stack(temp)
        if temp.dtype == torch.uint8: 
            temp = temp/255        
        images.append(temp)        
    print('loading time: ', time.time() - start_time)
    return images




######################################################################################
################################ image pre-processing ################################
######################################################################################

def prep_dataset(images,k=None, mean_zero=False, grayscale=True):
    '''
    Take 
    @k: number of replica of each image with a different intensity
    '''
    images = int_to_float(images)
    if grayscale: 
        images = rgb_to_gray(images) # convert to grayscale
    if mean_zero:
        #images = remove_mean(images)
        images = images - images.mean()
    if k is not None:
        images = change_intensity_dataset(images,k)
    images = torch.FloatTensor(images).permute(0,3,1,2).contiguous() # (B, C, H, W)
    return images
    




def center_crop(im):
    h = im.shape[0]
    w = im.shape[1]
    # first drop one column if h or w are odd
    if h%2 !=0: 
        im = im[0:-1]
    if w%2 !=0: 
        im = im[:,0:-1]

    #crop to square 
    h = im.shape[0]
    w = im.shape[1]
    r = int(min(h,w)/2) 
    o_h = int(h/2)
    o_w = int(w/2)
    im = im[o_h-r:o_h+r , o_w-r:o_w+r]
    return im 


def resize_image(im, s):
    '''
    @s: if float (0,1), new size will be computed. If tuple, that would be the new size 
    '''
    image_pil = Image.fromarray(im) # data type needed is uint8
    if type(s) is float:
        newsize = (int(image_pil.size[0] * s), int(image_pil.size[1] * s))
    else: 
        newsize = s
    image_pil_resize = image_pil.resize(newsize, resample=Image.BICUBIC)
    image_re = np.array(image_pil_resize)
    return image_re


def float_to_int(X):
    if X.dtype != 'uint8':
        return X
    else:
        return (X*255).astype('uint8')


def int_to_float(X):    
    if X.dtype == 'uint8':
        return (X/255).astype('float32')
    else:
        return X
    
def convert_8bit_to4bit(x): 
    
    for i in range(0,255,16): 
        print(i)
        mask = (x>=i) & (x<i+16)
        x[mask] = i/16
    return x

def rgb_to_gray(data):
    # n, h, w, c = data.shape
    return data.mean(-1).reshape(-1,1)

def change_intensity(  im , k):
    temp = np.zeros((k,im.shape[0], im.shape[1], im.shape[2])).astype('float32')
    for i in range(k):
        temp[i] = np.random.rand(1).astype('float32') * im
    return temp

def change_intensity_dataset(dataset, k):
    temp = []
    for im in dataset:
        temp.append(change_intensity(im,k))
    return np.concatenate(temp)

def rescale_image_range(im,  max_I, min_I=0):

    temp = (im - im.min())  /((im.max() - im.min()))
    return temp *(max_I-min_I) + min_I



def rescale_image(im):
    if im.device.type == 'cuda': 
        im = im.cpu()

    if type(im) == torch.Tensor:
        im = im.numpy()
    return ((im - im.min()) * (1/(im.max() - im.min()) * 255)).astype('uint8')


def remove_mean(images):
    '''
    remove mean of images in a numpy batch of size N,W,H,1
    '''
    return images - images.mean(axis=(1,2)).reshape(-1,1,1,1)

def patch_generator(all_images, patch_size, stride, one_tensor=True):
    '''images: a 4D tensor of image: B, C, H, W
    patch_size: a tuple indicating the size of patches
    stride: a tuple indicating the size of the strides
    def patch_generator(all_images, patch_size, stride, one_tensor=True):
    '''
    n_ims = all_images.shape[0]    
    im_height = all_images.shape[2]
    im_width = all_images.shape[3]

    h = int(im_height/stride[0]) * stride[0]
    w = int(im_width/stride[1]) * stride[1]

    all_patches = []
    patch_per_im = 0    
    for x in range(0,h- patch_size[0] + 1, stride[0]):
        for y in range(0,w - patch_size[1] + 1, stride[1]):
            patch = all_images[:,:, x:x+patch_size[0] , y:y+patch_size[1]]
            all_patches.append(patch)
            patch_per_im +=1
            
    print('patches per image: ',patch_per_im)
    out = torch.cat(all_patches)
    
    ### reorder the array to put the patches from the same image next to each other 
    all_data = []
    for d in range(n_ims): 
        ids = [i* n_ims + d for i in range(patch_per_im)]
        all_data.append(out[ids])    
    

    if one_tensor: 
        return torch.cat(all_data)
    else: #patches extracted from each image in one tensor. List of tensors=list of patches from different images 
        return all_data    


def patch_generator_np(all_images, patch_size, stride):
    '''images: a 4D numpy of image: B, C, H, W
    patch_size: a tuple indicating the size of patches
    stride: a tuple indicating the size of the strides
    '''
    im_height = all_images.shape[2]
    im_width = all_images.shape[3]

    h = int(im_height/stride[0]) * stride[0]
    w = int(im_width/stride[1]) * stride[1]

    all_patches = []
    for x in range(0,h- patch_size[0] + 1, stride[0]):
        for y in range(0,w - patch_size[1] + 1, stride[1]):
            patch = all_images[:,:, x:x+patch_size[0] , y:y+patch_size[1]]
            all_patches.append(patch)

    return np.concatenate(all_patches)


def patch_generator_with_scale(all_images, patch_size, stride, scales, resample = Image.BICUBIC):
    '''images: a 4D numpy array of image: B, H, W, C
    patch_size: a tuple indicating the size of patches
    stride: a tuple indicating the size of the strides
    scales: a list of float values by which the image is scaled
    '''

    all_images_patches = []
    # loop through all images in the set
    for image in all_images:
        image_patches = [] # holder for all patches of one image from different scales
        # loop through all the scales in the list
        for i in range(len(scales)):
            # resize the image (and blur if needed)
            image_pil = Image.fromarray(image) # data type needed is uint8
            # if blur is True:
                 # image_pil = image_pil.convert('L').filter(ImageFilter.GaussianBlur(1))
                 # image_pil = image_pil.convert('F')
            newsize = (int(image_pil.size[0] * scales[i]), int(image_pil.size[1] * scales[i]))
            image_pil_resize = image_pil.resize(newsize, resample=resample)
            image_re = np.array(image_pil_resize)


            im_height = image_re.shape[0]
            im_width = image_re.shape[1]


            patches = []
            h = int(im_height/stride[0]) * stride[0]
            w = int(im_width/stride[1]) * stride[1]
            # create patches for an image of a certain scale
            for x in range(0,h- patch_size[0] + 1, stride[0]):
                for y in range(0,w - patch_size[1] + 1, stride[1]):
                    # patches[counter] = image_re[ x:x+patch_size[0] , y:y+patch_size[1]]
                    patch = image_re[ x:x+patch_size[0] , y:y+patch_size[1]]
                    # patches.append(patch.reshape(1, patch.shape[0], patch.shape[1])) # add a dimension
                    patches.append(patch) # add a dimension

            patches = np.stack(patches, 0) # all the patches from one image at one scale
            image_patches.append(patches)
        image_patches = np.concatenate(image_patches, axis=0)
        all_images_patches.append(image_patches)
    return np.concatenate(all_images_patches, axis=0)


def data_augmentation(image,mode):
    if mode == 1:
        return image

    if mode == 2: # flipped
        image = np.flipud(image);
        return image

    elif mode == 3: # rotation 90
        image = np.rot90(image,1);
        return image;

    elif mode == 4 :# rotation 90 & flipped
        image = np.rot90(image,1);
        image = np.flipud(image);
        return image;

    elif mode == 5: # rotation 180
        image = np.rot90(image,2);
        return image;

    elif mode == 6: # rotation 180 & flipped
        image = np.rot90(image,2);
        image = np.flipud(image);
        return image;

    elif mode == 7: # rotation 270
        image = np.rot90(image,3);
        return image;

    elif mode == 8: # rotation 270 & flipped
        image = np.rot90(image,3);
        image = np.flipud(image);
        return image;
    else:
        raise ValueError('the requested mode is not defined')


def augment_training_data(train_set):

    augmented_train_set = np.zeros_like(train_set)
    for i in range(train_set.shape[0]):
        mode = np.random.randint(1,9)
        augmented_train_set[i,:,:] =  data_augmentation(train_set[i,:,:], mode)

    train_set = np.concatenate((train_set, augmented_train_set))
    return train_set



def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        try:
            nn.init.constant_(m.bias, 0)  # Initialize biases to zero
        except AttributeError: 
            pass
            
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
        
    elif classname.find('MultiDilationconv') != -1: 
        for c in m.convs: 
            nn.init.kaiming_normal_(c.weight.data, a = 0, mode= 'fan_in')
            try:
                nn.init.constant_(c.bias, 0)  # Initialize biases to zero
            except AttributeError: 
                pass
    elif classname.find('MultiDilationParentConv') != -1: 
        for c in m.convs: 
            nn.init.kaiming_normal_(c.weight.data, a = 0, mode= 'fan_in')
            try:
                nn.init.constant_(c.bias, 0)  # Initialize biases to zero
            except AttributeError: 
                pass
