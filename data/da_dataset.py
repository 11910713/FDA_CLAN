import os
import csv
import random
import torch
import os.path as osp
import numpy as np
import math
from util import scipy 
from PIL import Image
from util.data_info_gerneration import InfoGenerator
from util.img_process import center_crop
from data.base_dataset import BaseDataset, get_params, get_transform

# category integration list for source domain
ca_list_source = [2, 16, 17, 12, 3, 16, 15, 7, 7, 13, 7, 8, 5, 9, 11, 10, 6, 7, 10, 14, 7, 9, 13, 6, 11, 25, 26, 7, 5, 29, 30, 31, 7, 14, 34, 35]
ca_list_source = [2, 16, 16, 12, 3, 16, 15, 7, 7, 13, 7, 8, 5, 9, 11, 10, 6, 7, 10, 14, 7, 9, 13, 6, 11, 1, 1, 7, 5, 1, 1, 1, 7, 14, 1, 1]
ca_list_source = [1, 3, 3, 5, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# category intergration list for target domain
ca_list_target = [0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 3, 0, 0]

class DADataset(BaseDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--targetroot', type=str, required=True, help='path to target images')
        parser.add_argument('--L', type=float, default=0.005)
        parser.add_argument('--high_freq_r', type=int, default=5)
        return parser
    
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.mode = opt.phase
        self.opt = opt
        self.L = opt.L
        self.high_freq_r = opt.high_freq_r
        # configurtion for source domain
        self.source_root = opt.dataroot
        generator_s = InfoGenerator(self.source_root, None, None, [self.mode])
        generator_s.generate_info()
        source_img_ids = [i_id.strip() for i_id in open(osp.join(self.source_root, '{}.txt'.format(self.mode)))]
        self.source_infos = []
        for name in source_img_ids:
            img_path = osp.join(self.source_root, self.mode, "image/%s" % name)
            label_path = osp.join(self.source_root, self.mode, "label/%s" % name)
            info = {
                "img_name": name,
                "img_path": img_path,
                "label_path": label_path
            }
            self.source_infos.append(info)
        
        # configuration for target domain
        self.target_root = opt.targetroot
        generator_t = InfoGenerator(self.target_root, None, None, [self.mode])
        generator_t.generate_info()
        target_img_ids = [i_id.strip() for i_id in open(osp.join(self.target_root, '{}.txt'.format(self.mode)))]
        if self.opt.phase == 'training':
            target_img_ids = target_img_ids * int(math.ceil(len(source_img_ids) / len(target_img_ids)))
        self.target_infos = []
        for name in target_img_ids:
            label_name = name[:-4] + "_png_Label.csv"
            img_path = osp.join(self.target_root, self.mode, "image/%s" % name)
            label_path = osp.join(self.target_root, self.mode, "label/%s" % label_name)
            info = {
                "img_name": name,
                "img_path": img_path,
                "label_path": label_path
            }
            self.target_infos.append(info)

    def __getitem__(self, index):
        # get target image and label
        if self.opt.phase != 'training':
            target_info = self.target_infos[index]
        else:
            target_info = self.target_infos[random.randint(0, len(self.target_infos)-1)]
        target_img_path, target_label_path, target_name = target_info['img_path'], target_info["label_path"], target_info["img_name"]
        target_image = Image.open(target_img_path).convert("RGB")

        
        try:
            target_label = self.get_label(target_label_path, 'target')
        except Exception as e:
            print(e)
            index = index - 1 if index > 0 else index + 1
            return self.__getitem__(index)
        # the following two line is to crop the black border line of target images and labels
        target_image = center_crop(target_image, (680, 500))
        target_label = center_crop(target_label, (680, 500))
        target_high_freq = self.get_high_freq(target_image)
        
        # get source image and label
        source_info = self.source_infos[index]
        source_img_path, source_label_path, source_name = source_info["img_path"], source_info["label_path"], source_info["img_name"]
        source_image = Image.open(source_img_path)
        source_image = self.size_align_scale(source_image, target_image)
        if self.mode == 'training' and "FDA" in self.opt.name:
            source_image = self.get_fda_img(source_image, target_image, L=self.L)
        source_high_freq = self.get_high_freq(source_image)
        try:
            source_label = self.get_label(source_label_path, 'source')
        except Exception as e:
            print(e)
            index = index - 1 if index > 0 else index + 1
            return self.__getitem__(index)
        source_label = self.size_align_scale(source_label, target_image)
        if self.mode == 'training':
            if random.uniform(0, 1) > 0.55:
                rand_d = random.uniform(-30, 30)
            else:
                rand_d = 0
            hsv = random.uniform(0, 1) > 0.55
        else:
            hsv = False
            rand_d = 0
        rand_d = 0
        trans_param = get_params(self.opt, source_image.size)
        # process source domain
        source_image_trans = get_transform(self.opt, trans_param, grayscale=False, rotate=rand_d)
        source_label_trans = get_transform(self.opt, trans_param, grayscale=True, method=Image.NEAREST, convert=False, rotate=rand_d)
        source_image = source_image_trans(source_image)
        source_label = source_label_trans(source_label)
        source_label = torch.tensor(np.asarray(source_label), dtype=torch.float32)
        source_high_freq_trans = get_transform(self.opt, trans_param, grayscale=True)
        source_high_freq = source_high_freq_trans(source_high_freq)
        # proprocess target domain
        target_trans_params = get_params(self.opt, target_image.size)
        target_image_trans = get_transform(self.opt, target_trans_params, grayscale=False, rotate=rand_d)
        target_label_trans = get_transform(self.opt, target_trans_params, grayscale=False, method=Image.NEAREST, convert=False, rotate=rand_d)
        target_image = target_image_trans(target_image)
        target_label = target_label_trans(target_label)
        target_label = torch.tensor(np.asarray(target_label), dtype=torch.float32)
        target_high_freq_trans = get_transform(self.opt, target_trans_params, grayscale=True)
        target_high_freq = target_high_freq_trans(target_high_freq)

        return {"source_input": source_image, "source_label": source_label, "target_input": target_image, "source_name": source_name, "target_name": target_name, "target_label":target_label, "source_high_freq": source_high_freq, "target_high_freq": target_high_freq}
    
    def get_fda_img(self, src_img, tar_img, L=0.0025):
        src_img = np.asarray(src_img, np.float32).transpose((2, 0, 1))
        tar_img = np.asarray(tar_img, np.float32).transpose((2, 0, 1))
        result = scipy.FDA_adapt(src_img, tar_img, L=L)
        result = result.transpose((1, 2, 0))
        result = scipy.toimage(result, cmin=0.0, cmax=255.0)
        return result
    
    def size_align_crop(self, src_img, tar_img):
        W_s, H_s = src_img.size
        W_t, H_t = tar_img.size
        
        scale = W_s/W_t
        tar_img = tar_img.resize((W_s, int(H_t * scale)), Image.BICUBIC)
        cur_w, cur_h = tar_img.size
        left = (cur_w - W_s)/2
        top = (cur_h - H_s)/2
        right = (cur_w + W_s)/2
        bottom = (cur_h + H_s)/2
        tar_img = tar_img.crop((left, top, right, bottom))
        return tar_img

    def size_align_scale(self, ori_img, tar_img):
        W, H = tar_img.size
        ret = ori_img.resize((W, H), Image.NEAREST)
        return ret

    def get_label(self, path, domain, channel=1):
        # read label in RGB mode
        if "csv" in path:
            csvData = open(path)
            label = np.loadtxt(csvData, delimiter=",")
            label = label[1: , :] 
        else:
            label = Image.open(path)
        # NEAREST, ANTIALIAS
        label = np.asarray(label, np.int64)
        def func(e):
            if domain is 'source' :
                return ca_list_source[e]
            else :
                return ca_list_target[e]  
        vfunc = np.vectorize(func)
        label = vfunc(label)
        label = Image.fromarray(np.uint8(label))
        return label

    def __len__(self):
        if self.opt.phase == 'training':
            return len(self.source_infos)
        else:
            return len(self.target_infos)

    def get_valid_input(self, index):
        pass

    def get_high_freq(self, img):
        from PIL import ImageOps
        img = ImageOps.grayscale(img)
        img = np.asarray(img)
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)

        r = self.high_freq_r
        h, w = img.shape
        ch, cw = h//2, w//2
        fshift[ch-r:ch+r, cw-r:cw+r] = 0
        
        ishift = np.fft.ifftshift(fshift)
        img = np.fft.ifft2(ishift)
        img = np.abs(img)
        img = Image.fromarray(img)
        return img
        