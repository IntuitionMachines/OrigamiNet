import numpy as np
import torch
from scipy import ndimage
import scipy.misc
import skimage
from skimage import exposure
import matplotlib.pyplot as plt
import os,sys
from collections import namedtuple
from random import shuffle
import itertools
from sklearn.externals import joblib
from skimage import transform as stf
from PIL import Image
import unicodedata
import copy
from torch.utils.data import Dataset
import imageio
from math import floor, ceil
import random

try:
   import cPickle as pickle
except:
   import pickle

# Based on Elastic distortions in
# https://github.com/mdbloice/Augmentor/blob/master/Augmentor/Operations.py
class Distort3():

    def __init__(self, probability, grid_width, grid_height, magnitudeX, magnitudeY, Isize, min_h_sep, min_v_sep):

        self.probability = probability
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.xmagnitude = abs(magnitudeX)
        self.ymagnitude = abs(magnitudeY)
        self.randomise_magnitude = True

        w, h = Isize

        horizontal_tiles = self.grid_width
        vertical_tiles = self.grid_height

        width_of_square = int(floor(w / float(horizontal_tiles)))
        height_of_square = int(floor(h / float(vertical_tiles)))

        width_of_last_square = w - (width_of_square * (horizontal_tiles - 1))
        height_of_last_square = h - (height_of_square * (vertical_tiles - 1))

        dimensions = []
        shift = [[(0,0) for x in range(horizontal_tiles)] for y in range(vertical_tiles)]

        for vertical_tile in range(vertical_tiles):
            for horizontal_tile in range(horizontal_tiles):
                if vertical_tile == (vertical_tiles - 1) and horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_last_square + (horizontal_tile * width_of_square),
                                       height_of_last_square + (height_of_square * vertical_tile)])
                elif vertical_tile == (vertical_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_square + (horizontal_tile * width_of_square),
                                       height_of_last_square + (height_of_square * vertical_tile)])
                elif horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_last_square + (horizontal_tile * width_of_square),
                                       height_of_square + (height_of_square * vertical_tile)])
                else:
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_square + (horizontal_tile * width_of_square),
                                       height_of_square + (height_of_square * vertical_tile)])

                sm_h = min( self.xmagnitude, width_of_square  - (min_h_sep+shift[vertical_tile][horizontal_tile-1][0])  ) if horizontal_tile>0 else self.xmagnitude
                sm_v = min( self.ymagnitude, height_of_square - (min_v_sep+shift[vertical_tile-1][horizontal_tile][1])  ) if vertical_tile>0   else self.ymagnitude

                dx = random.randint(-sm_h, self.xmagnitude)
                dy = random.randint(-sm_v, self.ymagnitude)
                shift[vertical_tile][horizontal_tile] = (dx,dy)


        shift = list(itertools.chain.from_iterable(shift))

        
        last_column = []
        for i in range(vertical_tiles):
            last_column.append((horizontal_tiles - 1) + horizontal_tiles * i)

        last_row = range((horizontal_tiles * vertical_tiles) - horizontal_tiles, horizontal_tiles * vertical_tiles)

        polygons = []
        for x1, y1, x2, y2 in dimensions:
            polygons.append([x1, y1, x1, y2, x2, y2, x2, y1])

        polygon_indices = []
        for i in range((vertical_tiles * horizontal_tiles) - 1):
            if i not in last_row and i not in last_column:
                polygon_indices.append([i, i + 1, i + horizontal_tiles, i + 1 + horizontal_tiles])

        for id,(a, b, c, d) in enumerate(polygon_indices):
        
            dx = shift[id][0]
            dy = shift[id][1]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[a]
            polygons[a] = [x1, y1,
                           x2, y2,
                           x3 + dx, y3 + dy,
                           x4, y4]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[b]
            polygons[b] = [x1, y1,
                           x2 + dx, y2 + dy,
                           x3, y3,
                           x4, y4]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[c]
            polygons[c] = [x1, y1,
                           x2, y2,
                           x3, y3,
                           x4 + dx, y4 + dy]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[d]
            polygons[d] = [x1 + dx, y1 + dy,
                           x2, y2,
                           x3, y3,
                           x4, y4]

        generated_mesh = []
        for i in range(len(dimensions)):
            generated_mesh.append([dimensions[i], polygons[i]])

        self.generated_mesh = generated_mesh

    def perform_operation(self, image):
        return image.transform(image.size, Image.MESH, self.generated_mesh, resample=Image.BICUBIC)


def aug_ED2(imgs, w, h, n_ch, tst=False):

    d = Distort3(1.0, 10, 10, 0, 25, [w, h], 1, 1)

    for i in range(len(imgs)):        
        res = d.perform_operation(Image.fromarray(np.squeeze((imgs[i] * 255).astype(np.uint8))))
        imgs[i] = np.reshape(res, [h, w, n_ch])

    return np.squeeze(imgs)


def RndTform(img,val=125 * 1.5):

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

    tl_top = rd(dh)  # Top left corner, top margin
    tl_left = fd(dw)  # Top left corner, left margin
    bl_bottom = rd(dh)  # Bottom left corner, bottom margin
    bl_left = fd(dw)  # Bottom left corner, left margin
    tr_top = rd(dh)  # Top right corner, top margin
    tr_right = fd( min(Iw * 3/4 - tl_left,dw) )  # Top right corner, right margin
    br_bottom = rd(dh)  # Bottom right corner, bottom margin
    br_right = fd( min(Iw * 3/4 - bl_left,dw) )  # Bottom right corner, right margin
    

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
    
    translation = (minc, minr)
    tform4 = stf.SimilarityTransform(translation=translation)
    tform = tform4 + tform
    tform.params /= tform.params[2, 2]
    
    ret = []
    for i in range(len(img)):
        img2 = stf.warp(img[i], tform, output_shape=output_shape, cval=1.0)
        img2 = stf.resize(img2, (Ih,Iw), preserve_range=True).astype(np.float32)
        ret.append(img2)

    return ret

def npThum(img, max_w, max_h):
    x, y = np.shape(img)[:2]

    y = min(int( y * max_h / x ),max_w)
    x = max_h

    img = np.array(Image.fromarray(img).resize((y,x)))
    return img

image_data = np.array(Image.open(sys.argv[1]))

image_data = npThum(image_data, 750, 750)
image_data = skimage.img_as_float32(image_data)
if image_data.ndim < 3:
    image_data = np.expand_dims(image_data, axis=-1)

images = image_data[None,...]

images = np.array([RndTform([image], val=140)[0] for image in images])
sh = images.shape
images = np.array([aug_ED2(image[None,...],sh[2], sh[1], sh[3], tst=False) for image in images])
imageio.imwrite(sys.argv[1],images[0].astype(np.uint8))
