import random
import torch

from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
from util.data_info_gerneration import InfoGenerator
from PIL import Image
import numpy as np
import os.path as osp
import math 

# category integration list
ca_list = {7: [8, 10, 27, 20, 32], 9: [22], 12: [28], 13: [21], 14: [24], 15: [18], 16: [23]}

class SegmentationDataset(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.data_root = opt.dataroot
        self.opt = opt
        self.mode = opt.phase
        self.input_nc = 3
        self.output_nc = 1
        generator = InfoGenerator(self.data_root, None, None, [self.mode], dtype='.png')
        generator.generate_info()
        self.info_path = osp.join(self.data_root, '{}.txt'.format(self.mode))
        self.img_ids = [i_id.strip() for i_id in open(self.info_path)]
        self.img_infos = []

        # aggregate image information
        for name in self.img_ids:
            img_file = osp.join(self.data_root, self.mode, "image/%s" % name)
            label_path = osp.join(self.data_root, self.mode, "label/%s" % name)
            info = {
                "img_path": img_file,
                "image_name": name,
                "label_path": label_path
            }
            self.img_infos.append(info)

        if opt.phase == 'training':
            generator = InfoGenerator(self.data_root, None, None, ['test'], dtype='.png')
            generator.generate_info()
            valid_path = osp.join(self.data_root, 'test.txt')
            # valid_path = osp.join()
            valid_infos = []
            for name in [name.strip() for name in open(valid_path)]:
                img_path = osp.join(self.data_root, 'test', 'image', name)
                label_path = osp.join(self.data_root, 'test', 'label', name)
                info = {
                    "img_path": img_path,
                    "label_path": label_path,
                    "img_name": name,
                }
                valid_infos.append(info)
            self.valid_infos = valid_infos * int(math.ceil(len(self.img_infos) / len(valid_infos)))

    def __getitem__(self, index):
        info = self.img_infos[index]
        image_path = info["img_path"]
        image_name = info["image_name"]
        try:
            label = self.get_label(info["label_path"])
        except Exception as e:
            print(e)
            index = index - 1 if index > 0 else index + 1
            return self.__getitem__(index)
        image = Image.open(image_path).convert('RGB')
        if self.mode == 'training':
            if random.uniform(0, 1) > 0.55:
                rand_d = random.uniform(-30, 30)
            else:
                rand_d = 0
            hsv = random.uniform(0, 1) > 0.55
        else:
            hsv = False
            rand_d = 0
        transform_params = get_params(self.opt, image.size)
        image_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1), rotate=rand_d, HSV=hsv)
        label_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1),
                                        method=Image.NEAREST, convert=False, rotate=rand_d)

        image = image_transform(image)
        label = label_transform(label)
        label = torch.tensor(np.asarray(label), dtype=torch.float32)
        return {'image': image, 'label': label, 'image_name': image_name}

    def get_label(self, path):
        # read label in RGB mode
        label = Image.open(path)
        # NEAREST, ANTIALIAS
        label = np.asarray(label, np.int64)
        for select, replaces in ca_list.items():
            for replace in replaces:
                label[label==replace] = select
        label = Image.fromarray(np.uint8(label))
        return label

    def __len__(self):
        return len(self.img_ids)

    def get_valid_input(self, index):
        info = self.valid_infos[index]
        img_path = info["img_path"]
        img_name = info['img_name']
        label_path = info['label_path']
        try:
            label = self.get_label(label_path)
        except Exception as e:
            index = index - 1 if index > 0 else index + 1
            return self.get_valid_input(index)
        img = Image.open(img_path).convert('RGB')
        transform_params = get_params(self.opt, img.size)
        img_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        label_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1), method=Image.NEAREST, convert=False)
        img = img_transform(img)
        label = label_transform(label)
        label = torch.tensor(np.asarray(label), dtype=torch.float32)
        return {'img': img, 'label': label, 'img_name': img_name}

