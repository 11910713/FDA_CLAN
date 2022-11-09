import os.path as osp
import csv
from pickletools import read_decimalnl_long
import numpy as np
import argparse
from PIL import Image


class SegMetrics(object):
    
    def __init__(self, opt):
        '''
            classes.csv: file should include name and corresponding id in the form of "{id},{name}"
        '''
        # specify root 
        self.opt = opt
        self.root = osp.join(opt.results_dir, opt.name, 'test_{}'.format(opt.epoch))
        self.img_type = opt.type
        # get class info
        class_path = osp.join(opt.dataroot, 'classes.csv')
        self.classes = {}
        with open(class_path) as fi:
            infos = [info for info in csv.reader(fi)]
            self.classes = {int(info[0]): {'name': info[1], 'metric_results': {}} for info in infos[1:] if len(infos) > 1} # Metric result is never accessed
        self.num_classes = len(self.classes)
        # get img info
        self.img_infos = []
        with open(osp.join(self.root, 'test.txt'), 'r') as fi:
            names = [name.strip().split(self.img_type)[0] for name in fi]
            self.img_infos = [{'mask': osp.join(self.root, 'images', '{}_mask{}'.format(name, self.img_type)) \
                , 'gt': osp.join(self.root, 'images', '{}_ground_truth{}'.format(name, self.img_type))} for name in names]
        self.results = {}

    def cal_pixel_prediction(self, gt, mask):
        # flatten image # TODO: It is good to think why we need uint32 here

        gt = np.asarray(gt, np.uint32)
        gt = gt.flatten()
        mask = np.asarray(mask, np.uint32)
        mask = mask.flatten()
        num_class = len(self.classes)
        # calculate Pij

        result = np.bincount(gt*num_class + mask, minlength=num_class ** 2)
        result = result.reshape(num_class, num_class)
        return result

    def  get_confusion_matrix(self):
        matrix = np.zeros((self.num_classes, self.num_classes), np.float64)
        for index, info in enumerate(self.img_infos):
            # get gt 
            gt =  Image.open(info['gt'])
            gt = np.asarray(gt, np.uint8)
            mask = Image.open(info['mask'])
            mask = np.asarray(mask, np.uint8)
            img_statics = self.cal_pixel_prediction(gt, mask)
            matrix += img_statics
        self.matrix = matrix
        
    def cal_miou(self):
        iou = np.diag(self.matrix) / (self.matrix.sum(0) + self.matrix.sum(1) - np.diag(self.matrix))
        selected = [i for i in range(self.num_classes) if self.classes[i]["name"] != "null"]
        miou = 0
        for i in selected:
            print(iou[i], i)
            miou += iou[i]
        miou /= len(selected)
        self.results['miou'] = miou

    def cal_fiou(self):
        freq = np.sum(self.matrix, 1) / np.sum(self.matrix)
        iou = np.diag(self.matrix) / (self.matrix.sum(0) + self.matrix.sum(1) - np.diag(self.matrix))
        fiou = (freq[freq > 0] * iou[freq > 0]).sum()
        for cls in self.classes.keys():
            self.classes[cls]['metric_results']['iou'] = iou[cls]
        self.results['fiou'] = fiou
    
    def cal_pa(self):
        pa = np.diag(self.matrix).sum() / self.matrix.sum()
        self.results['pa'] = pa
    
    def cal_recall(self):
        recall = np.diag(self.matrix) / self.matrix.sum(1)
        recall = np.nanmean(recall)
        self.results['recall'] = recall

    def cal_precision(self):
        precision = np.diag(self.matrix) / self.matrix.sum(0)
        precision = np.nanmean(precision)
        self.results['precision'] = precision

    def cal_dice(self):
        p, r = self.results['precision'], self.results['recall']
        dice = 2 * p * r / (p + r)  
        self.results['dice'] = dice
    
    def print_results(self):
        with open(osp.join(self.root, 'metrics.txt'), 'w') as fi:
            for name, value in self.results.items():
                print("%s : %.04f" % (name, value))
                fi.write("%s : %.04f\n" % (name, value))
            fi.write("===============================iou================================\n")
            print("===============================iou================================")
            for cls in self.classes.keys():
                # if cls in [0, 1, 2, 3, 4, 5, 6, 8, 15, 17] :
                # if cls == 10:
                print("%s : %.04f" % (cls, self.classes[cls]['metric_results']['iou']))
                # fi.write("%s,%.04f\n" % (self.classes[cls]['name'], self.classes[cls]['metric_results']['iou']))
                fi.write("%.04f\n" % (self.classes[cls]['metric_results']['iou']))

    
    def print_category_results(self, name):
        for dic in self.classes.values():
            print("%s : %.03f" % (dic['name'], dic['metric_results'][name]))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--epoch", type=str, required=True)
    parser.add_argument("--type", type=str, default='.png')
    opt = parser.parse_args()
    metrics = SegMetrics(opt)
    metrics.get_confusion_matrix()
    metrics.cal_pa()
    metrics.cal_recall()
    metrics.cal_miou()
    metrics.cal_precision()
    metrics.cal_dice()
    metrics.cal_fiou()
    metrics.print_results()
    print("--------------------------------")

        