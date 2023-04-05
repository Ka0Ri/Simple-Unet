import numpy as np
import cv2
import os
import h5py
from random import randint
path = os.getcwd() + "/dataset-master/"

default = 128
def crop_random(img, mask, n):
    h, w = img.shape[:2]
    result_img = []
    result_mask = []
    i = 0
    while(True):
        top_left = [randint(0, h - default), randint(0, w - default)]

        size = randint(default, h - default)
        if(top_left[0] + size > h or top_left[1] + size > w):
            continue
        crop_mask = mask[top_left[0]:top_left[0] + size, top_left[1]:top_left[1] + size]
        crop_mask = cv2.resize(crop_mask, (default, default))
        if(np.sum(crop_mask) > default*default/1.2):
            continue
        result_mask.append(crop_mask)
        crop_img = img[top_left[0]:top_left[0] + size, top_left[1]:top_left[1] + size]
        crop_img = cv2.resize(crop_img, (default, default))
        result_img.append(crop_img)
        i = i + 1
        if(i == n):
            break
    return result_img, result_mask

imgs = np.zeros((1, default, default, 3))
masks = np.zeros((1, default, default))
for name in sorted(os.listdir(path + "images/")):
    img = cv2.imread(path + "images/" + name)/255.0
    mask = cv2.imread(path + "masks/" + name[:3] + "_mask.png", 0)/255.0
    result_img, result_mask = crop_random(img, mask, 500)
    imgs = np.concatenate([imgs, result_img], axis=0)
    masks = np.concatenate([masks, result_mask], axis=0)

hf = h5py.File('data.h5', 'w')
hf.create_dataset('data', data=imgs[1:])
hf.create_dataset('groundtruth', data=masks[1:])
hf.close()


hf = h5py.File('data.h5', 'r')
n1 = np.array(hf.get('groundtruth'))
print(n1.shape)
hf.close()


    
    