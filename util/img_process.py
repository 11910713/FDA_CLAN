from PIL import Image
import os.path as osp
import numpy as np
from numpy import inf

ca_list = {7: [8, 10, 27, 20, 32], 9: [22], 12: [28], 13: [21], 14: [24], 15: [18], 16: [23]}

def center_crop(img, new_size):
    W, H = img.size
    (n_W, n_H) = new_size
    left = (W - n_W)/2
    top = (H - n_H)/2
    right = (W + n_W)/2
    bottom = (H + n_H)/2
    return img.crop((left, top, right, bottom))

def get_frequency(datapath, replace_list=ca_list, n_cls=36):
    name_list = [name.strip() for name in open(osp.join(datapath, 'training.txt'))]
    frequency = [0 for i in range(n_cls)]
    for name in name_list:
        img = Image.open(osp.join(datapath, 'training', 'label', name))
        img = np.asarray(img)
        img = label_cls_replace(img, replace_list)
        count = np.bincount(img.reshape(-1), minlength=n_cls)
        frequency = [frequency[i] + count[i] for i in range(n_cls)]
    with open(osp.join(datapath, 'frequency.txt'), 'w') as fp:
        for f in frequency:
            fp.write("%d\n" % f )
    print(frequency)

def get_frequency_weight(datapath):
    frequency = [int(f.strip()) for f in open(osp.join(datapath, 'frequency.txt'))]
    frequency =  np.array(frequency)
    median = np.median(frequency)
    print(frequency)
    print(median)
    frequency = median / frequency
    frequency[frequency == inf] = 1
    # print(frequency)
    return frequency


def label_cls_replace(label, replace_list=ca_list):
    for select, replaces in replace_list.items():
        for replace in replaces:
            label[label==replace] = select
    return label

if __name__ == '__main__':
    get_frequency_weight("/data/liuhaofeng/oumingyang/eye_cataract/dataset/CADIS-2")