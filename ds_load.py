import numpy as np
import torch
from scipy import ndimage
import scipy.misc
import skimage
import os,sys
import itertools
from skimage import transform as stf
from PIL import Image
from torch.utils.data import Dataset
import imageio
from math import floor, ceil
import pickle
import gin

def RndTform(img,val):
    Ih,Iw = img[0].shape[:2]
    
    sgn = torch.randint(0,2,(1,)).item() * 2 - 1

    if sgn>0:
        dw = val
        dh = 0
    else:
        dw = 0
        dh = val

    def rd(d): return torch.empty(1).uniform_(-d,d).item()
    def fd(d): return torch.empty(1).uniform_(-dw,d).item()

    # generate a random projective transform
    # adapted from https://navoshta.com/traffic-signs-classification/
    tl_top = rd(dh)
    tl_left = fd(dw)
    bl_bottom = rd(dh)
    bl_left = fd(dw)
    tr_top = rd(dh)
    tr_right = fd( min(Iw * 3/4 - tl_left,dw) )
    br_bottom = rd(dh)
    br_right = fd( min(Iw * 3/4 - bl_left,dw) )

    tform = stf.ProjectiveTransform()
    tform.estimate(np.array((
        (tl_left, tl_top),
        (bl_left, Ih - bl_bottom),
        (Iw - br_right, Ih - br_bottom),
        (Iw - tr_right, tr_top)
    )), np.array((
        [0, 0 ],
        [0, Ih - 1 ],
        [Iw-1, Ih-1 ],
        [Iw-1, 0]
    )))

    # determine shape of output image, to preserve size
    # trick take from the implementation of skimage.transform.rotate
    corners = np.array([
        [0, 0 ],
        [0, Ih - 1 ],
        [Iw-1, Ih-1 ],
        [Iw-1, 0]
    ])

    corners = tform.inverse(corners)
    minc = corners[:, 0].min()
    minr = corners[:, 1].min()
    maxc = corners[:, 0].max()
    maxr = corners[:, 1].max()
    out_rows = maxr - minr + 1
    out_cols = maxc - minc + 1
    output_shape = np.around((out_rows, out_cols))

    # fit output image in new shape
    translation = (minc, minr)
    tform4 = stf.SimilarityTransform(translation=translation)
    tform = tform4 + tform
    # normalize
    tform.params /= tform.params[2, 2]
    

    ret = []
    for i in range(len(img)):
        img2 = stf.warp(img[i], tform, output_shape=output_shape, cval=1.0)
        img2 = stf.resize(img2, (Ih,Iw), preserve_range=True).astype(np.float32)
        ret.append(img2)


    return ret

@gin.configurable
def SameTrCollate(batch, prjAug, prjVal):
    images, labels = zip(*batch)

    images = [image.transpose((1,2,0)) for image in images]
    
    if prjAug:
        images = [RndTform([image], val=prjVal)[0] for image in images] #different transform to each batch
        # images = RndTform(images, val=prjVal) #apply same transform to all images in a batch
   
    image_tensors = [torch.from_numpy(np.array(image, copy=False)) for image in images]
    image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)
    image_tensors = image_tensors.permute(0,3,1,2)

    return image_tensors, labels


class myLoadDS(Dataset):
    def __init__(self, flist, dpath, ralph=None, fmin=True, mln=None):
        self.fns = get_files(flist, dpath)
        self.tlbls = get_labels(self.fns)
        
        if ralph == None:
            alph  = get_alphabet(self.tlbls)
            self.ralph = dict (zip(alph.values(),alph.keys()))
            self.alph = alph
        else:
            self.ralph = ralph
        
        if mln != None:
            filt = [len(x) <= mln if fmin else len(x) >= mln for x in self.tlbls]
            self.tlbls = np.asarray(self.tlbls)[filt].tolist()
            self.fns   = np.asarray(self.fns  )[filt].tolist()
    
    def __len__(self):
        return len(self.fns)

    def __getitem__(self, index):
        timgs = get_images(self.fns[index])
        timgs = timgs.transpose((2,0,1))

        return ( timgs , self.tlbls[index] )

def get_files(nfile, dpath):
    fnames = open(nfile, 'r').readlines()
    fnames = [ dpath + x.strip() for x in fnames ]
    return fnames

def npThum(img, max_w, max_h):
    x, y = np.shape(img)[:2]

    y = min(int( y * max_h / x ),max_w)
    x = max_h

    img = np.array(Image.fromarray(img).resize((y,x)))
    return img

@gin.configurable
def get_images(fname, max_w, max_h, nch):

    try:

        image_data = np.array(Image.open(fname))
        image_data = npThum(image_data, max_w, max_h)
        image_data = skimage.img_as_float32(image_data)

        h, w = np.shape(image_data)[:2]
        if image_data.ndim < 3:
            image_data = np.expand_dims(image_data, axis=-1)
        
        if nch==3 and image_data.shape[2]!=3:
            image_data = np.tile(image_data,3)

        image_data = np.pad(image_data,((0,0),(0,max_w-np.shape(image_data)[1]),(0,0)), mode='constant', constant_values=(1.0))

    except IOError as e:
        print('Could not read:', fname, ':', e)
    
    return image_data

def get_labels(fnames):

    labels = []
    for id,image_file in enumerate(fnames):
        fn  = os.path.splitext(image_file)[0] + '.txt'
        lbl = open(fn, 'r').read()
        lbl = ' '.join(lbl.split()) #remove linebreaks if present
        
        labels.append(lbl)

    return labels

def get_alphabet(labels):
    
    coll = ''.join(labels)     
    unq  = sorted(list(set(coll)))
    unq  = [''.join(i) for i in itertools.product(unq, repeat = 1)]
    alph = dict( zip( unq,range(len(unq)) ) )

    return alph