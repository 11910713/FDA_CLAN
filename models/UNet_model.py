from numpy import full
import torch
import torch.nn.functional as F
import random
from torch import nn
from .base_model import BaseModel
from . import networks

class MVDGANv2Model(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_CE', type=float, default=1.0, help='weight for L1 loss')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # attributes for visualization 
        self.loss_names = ['G', 'CE'] # need modification
        if opt.phase == 'training':
            self.visual_names = ['input', 'color_source_label', 'color_pred_source', 'color_pred_target', 'color_target_label'] # 
        else:
            self.visual_names = ['target_input', 'ground_truth', 'color_target_label', 'mask', 'color_pred_target']
        # models
        self.model_names = ['G', 'D']
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, netG=opt.netG, gpu_ids=self.gpu_ids)
        self.netD = networks.define_D(opt.output_nc, 1, netD=opt.netD, gpu_ids=self.gpu_ids)
        self.source_id, self.target_id = 1, 0
        self.epoch_stop = opt.n_epochs + opt.n_epochs_decay
        if self.isTrain:  # define discriminators
            self.lambda_CE = opt.lambda_CE
            self.criterionCE = torch.nn.CrossEntropyLoss()
            self.optimizer_G =  torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
        
    def set_input(self, input, isTrain=None):
        # set sourcc input image and label
        self.input = input['source_input'].to(self.device)
        self.label = input['source_label'].to(self.device, dtype=torch.long)
        # set target input image
        self.image_paths = input['target_name']

    def compute_visuals(self):
        self.color_label = torch.unsqueeze(self.source_label, dim=1)
        self.color_pred = torch.unsqueeze(torch.argmax(self.pred_source, dim=1), dim=1)
        if not self.isTrain:
            self.mask = torch.unsqueeze(torch.argmax(self.pred_target, dim=1), dim=1)

    def optimize_parameters(self):
        self.optimizer_G.zero_grad()
        self.forward()
        self.set_requires_grad([self.netG], False)
        self.set_requires_grad([self.netD], True)
        self.forward_D()
        self.backward_D()
        self.optimizer_D.step()
        self.set_requires_grad([self.netG], True)
        self.set_requires_grad([self.netD], False)
        self.backward_G()
        self.optimizer_G.step()

    def set_epoch_num(self, epoch):
        self.curr_epoch = epoch

    def forward(self):
        # source image:
        self.pred_source_1, self.pred_source_2 = self.netG(self.source_input)
        self.pred_source = self.pred_source_1 + self.pred_source_2
        # target input:
        self.pred_target1, self.pred_target2 = self.netG(self.target_input)
        self.pred_target = self.pred_target1 + self.pred_target2
        self.weight_map = self.weighted_map(F.softmax(self.pred_target1, dim=1), F.softmax(self.pred_target2, dim=1))

    def forward_D(self):
        # soruce 
        source = self.pred_source.detach()
        self.D_out_s = self.netD(source)
        # target
        target = self.pred_target.detach()
        self.D_out_t = self.netD(target)

    def backward_D(self):
        # source
        self.loss_D_source = self.BCELoss(self.D_out_s, torch.full_like(self.D_out_s, self.source_id))
        # target
        weight_map = self.weight_map.detach()
        self.loss_D_target = self.BCELossWithWeight(self.D_out_t, torch.full_like(self.D_out_t, self.target_id))
        # self.loss_D_target = self.loss_D_target*self.alpha+self.loss_D_target*weight_map*self.beta
        self.loss_D_target *= weight_map
        self.loss_D_target = torch.mean(self.loss_D_target)

        self.loss_D = (self.loss_D_target+self.loss_D_source)
        self.loss_D.backward()
    
    def backward_G(self):
        # adv loss (source, target), segmentation loss (source, target), discrepancy loss(classifieres)
        # adv loss
        adv_source = self.netD(self.pred_source)
        adv_target = self.netD(self.pred_target)
        adv_loss_source = self.BCELoss(adv_source, torch.full_like(adv_source, self.target_id))
        adv_loss_target = self.BCELossWithWeight(adv_target, torch.full_like(adv_target, self.source_id)) 
        adv_loss_target *= self.weight_map
        adv_loss_target = torch.mean(adv_loss_target)
        self.loss_adv = (adv_loss_source + adv_loss_target) * self.lamda_adv 
        # segmentation loss
        self.loss_CE_1 = self.criterionCE(self.pred_source_1, self.source_label)
        self.loss_CE_2 = self.criterionCE(self.pred_source_2, self.source_label)
        self.loss_CE = (self.loss_CE_1 + self.loss_CE_2) * self.lambda_CE
        # discrepancy loss
        self.loss_discrepancy = self.discrepancyLoss(self.netG.module.classifier1.parameters(), self.netG.module.classifier2.parameters())*self.damping*self.lambda_dis
        # total loss
        self.loss_G = self.loss_adv + self.loss_CE + self.loss_discrepancy
        # backward
        self.loss_G.backward()

    def weighted_map(self, pred1, pred2):
        output = 1.0 - torch.sum((pred1 * pred2), 1).view(-1, 1, pred1.size(2), pred1.size(3)) / \
        (torch.norm(pred1, 2, 1) * torch.norm(pred2, 2, 1)).view(-1, 1, pred1.size(2), pred1.size(3))
        return output

    def validation(self, data):
        pass