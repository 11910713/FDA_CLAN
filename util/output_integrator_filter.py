from glob import glob 
from PIL import Image
from seg_metrics import SegMetrics
import os.path as osp
import argparse
import numpy as np


class OutputFilter(object):

    def __init__(self, opt):
        self.root = opt.root
        self.is_split = opt.is_split
        if opt.is_split: 
            self.crop_size = opt.crop_size
            self.intervene_width = opt.it_width
        else:
            self.load_size = opt.load_size
            self.epoch = opt.epoch
        self.img_names = [name.strip() for name in open(opt.info_path)]
        self.img_type = opt.type
        self.output_dir = opt.out_dir
        self.num_classes = opt.num_classes
        self.target_dirs = ["CADIS-2_FCN8s_c36_b2_ep80", "CADIS-2_FCN16s_c36_b7_ep80", "CADIS-2_FCN32s_c36_b7_ep80", "CADIS-2_MNet_c36_b2_ep80", "CADIS-2_UNet_c36_b2_ep80", "CADIS-2_U2D_c36_b3_ep80_crop" ] # 
        self.target_dirs = ["CADIS-2_U2D_c36_b3_ep80" ] # 

    
    def get_single_img_confusion_matrix(self, gt, mask):
        mask = np.asarray(mask, np.uint8)
        gt = np.asarray(gt, np.uint8)
        matrix = np.zeros((self.num_classes, self.num_classes), np.float64)
        matrix += self.cal_pixel_prediction(gt, mask)
        return matrix

    def cal_pixel_prediction(self, gt, mask):
        # flatten image # TODO: It is good to think why we need uint32 here
        gt = np.asarray(gt, np.uint32)
        mask = np.asarray(mask, np.uint32)
        gt = gt.flatten()
        mask = mask.flatten()
        num_class = self.num_classes
        # calculate Pij

        result = np.bincount(gt*num_class + mask, minlength=num_class ** 2).reshape(num_class, num_class)
        return result

    def integrate_output(self, img_name, suffix, target_dir):
        img_name = img_name.split(self.img_type)[0] + suffix + self.img_type
        if self.is_split:
            path = osp.join(self.root, target_dir, "left", "images", img_name)
            left = Image.open(path)
            path = osp.join(self.root, target_dir, "right", "images", img_name)
            right = Image.open(path)
            right = right.crop((self.intervene_width, 0, self.crop_size, self.crop_size))
            cat = Image.new(right.mode, (left.width+right.width, left.height))
            cat.paste(left, (0, 0))
            cat.paste(right, (left.width, 0))
            return cat
        else:
            path = osp.join(self.root, self.epoch, "images", img_name)
            img = Image.open(path)
            img = img.resize(self.load_size)
            return img
    
    def cal_accuracy(self, matrix):
        return np.diag(matrix).sum() / matrix.sum()
    
    def cal_fiou(self, matrix):
        freq = np.sum(matrix, 1) / np.sum(matrix)
        iou = np.diag(matrix) / (matrix.sum(0) + matrix.sum(1) - np.diag(matrix))
        fiou = (freq[freq > 0] * iou[freq > 0]).sum()
        return fiou     

    def eval_per_img(self):
        with open(osp.join(self.output_dir, "metric_per_img.csv"), 'w') as fo:
            header = "name"
            for dir in self.target_dirs:
                header += ",{}:pa".format(dir)
                # header += ",{}:fiou".format(dir)
            header += '\n'
            fo.write(header)
        for name in self.img_names:
            metrics = name
            for dir in self.target_dirs:
                gt = self.integrate_output(name, "_ground_truth", dir)
                mask = self.integrate_output(name, "_mask", dir)
                matrix = self.get_single_img_confusion_matrix(gt, mask)
                pa = self.cal_accuracy(matrix)
                fiou = self.cal_fiou(matrix)
                # metrics += ",%.04f,%.04f" % (pa, fiou) 
                metrics += ",%.04f" % ( pa) 

            metrics +='\n'
            with open(osp.join(self.output_dir, "metric_per_img.csv"), 'a') as fo:
                    fo.write(metrics)
            print(metrics)
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--is_split", type=bool, default=False)
    parser.add_argument("--crop_size", type=int, default=480)
    parser.add_argument("--it_width", type=float, default=106.7)
    parser.add_argument("--info_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=True)
    parser.add_argument("--type", type=str, default=".png")
    parser.add_argument("--num_classes", type=int, default=36)
    parser.add_argument("--load_size", type=int, default=480)
    
    opt = parser.parse_args()
    filter = OutputFilter(opt)
    filter.eval_per_img()
    print("--------------------------------")

# /data/liuhaofeng/miniconda3/envs/python3.6/bin/python /data/liuhaofeng/oumingyang/eye_cataract/SegmentationCodeBase/util/output_integrator_filter.py --root /data/liuhaofeng/oumingyang/exp_results --info_path /data/liuhaofeng/oumingyang/exp_results/CADIS-2_U2D_c36_b3_ep80_crop/left/test.txt --out_dir /data/liuhaofeng/oumingyang/exp_results/CADIS-2_U2D_c36_b3_ep80_crop 
