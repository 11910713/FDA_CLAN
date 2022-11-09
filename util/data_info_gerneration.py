import glob
import csv
from os import path as osp

class InfoGenerator(object):

    def __init__(self, data_root, label_class_path, output_path=None, ttv_directory=None, dtype='.png'):
        self.data_root = data_root
        self.label_class_path = label_class_path
        self.ttv_directory = [''] if ttv_directory is None else ttv_directory
        self.output_path = data_root if output_path is None else output_path
        self.dtype = dtype

    def generate_info(self):
        for subdirectory in self.ttv_directory:
            filename = 'root' if subdirectory == '' else subdirectory
            # image
            with open(osp.join(self.output_path, filename+'.txt'), 'w') as fo:
                data_path = osp.join(self.data_root, subdirectory, 'image/*'+self.dtype, )
                for path in glob.glob(data_path):
                    fo.write('{0}\n'.format(path.split('/')[-1]))

    def generate_class_map(self, output_path=None):
        if output_path is None:
            output_path = self.data_root
        with open(self.label_class_path, 'r') as fi:
            text = csv.reader(fi)
            colors = []
            names = []
            for item in text:
                names.append(item[0])
                colors.append(item[1])
        color_map = {}
        color_name = {}
        r_color_map = {}
        r_color_name = {}
        for index in range(len(colors)):
            color = colors[index]
            name = names[index]
            color_map[color] = index
            r_color_map[index] = color
            color_name[color] = name
            r_color_name[name] = color
        with open(output_path+"/info.txt", 'w') as fo:
            fo.write(color_map.__str__() + '\n')
            fo.write(r_color_map.__str__() + '\n')
            fo.write(color_name.__str__() + '\n')
            fo.write(r_color_name.__str__() + '\n')
