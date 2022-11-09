from numpy import full
import torch
import torch.nn.functional as F
import random
from torch import nn
from .base_model import BaseModel
from . import networks

class WeightDiscrepancyLoss(nn.Module):
    def __init__(self, opt):
        super(WeightDiscrepancyLoss, self).__init__()
        self.w1 = None
        self.w2 = None
        self.opt = opt

    def forward(self, para1, para2):
        self.w1 = None
        self.w2 = None
        for (w1, w2) in zip(para1, para2):
            w1 = w1.view(-1)
            w2 = w2.view(-1)
            if self.w1 is None and self.w2 is None:
                self.w1 = w1
                self.w2 = w2
            else:
                self.w1 = torch.cat((self.w1, w1), 0)
                self.w2 = torch.cat((self.w2, w2), 0)
        discrepancy_loss = (torch.matmul(self.w1, self.w2) / (torch.norm(self.w1) * torch.norm(self.w2)) + 1)
        return discrepancy_loss

class MVDGANv2Model(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_CE', type=float, default=1.0, help='weight for L1 loss')
            parser.add_argument('--D_lr', type=float, required=True)
            parser.add_argument('--lambda_adv', type=float, default=0.1)
            parser.add_argument('--lambda_dis', type=float, default=2)
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # attributes for visualization 
        self.loss_names = ['G', 'CE', 'discrepancy', 'D', 'adv', 'D_target', 'D_source'] # need modification
        if opt.phase == 'training':
            self.visual_names = ['source_input', 'target_input', 'color_source_label', 'color_pred_source', 'color_pred_target', 'color_target_label'] # 
        else:
            self.visual_names = ['target_input', 'ground_truth', 'color_target_label', 'mask', 'color_pred_target']
        # models
        self.model_names = ['G', 'D']
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, netG=opt.netG, gpu_ids=self.gpu_ids)
        self.netD = networks.define_D(opt.output_nc, 1, netD=opt.netD, gpu_ids=self.gpu_ids)
        self.source_id, self.target_id = 1, 0
        self.epoch_stop = opt.n_epochs + opt.n_epochs_decay
        if self.isTrain:  # define discriminators
            self.alpha = 0.7
            self.beta = 0.3
            self.lamda_adv = opt.lambda_adv
            self.lambda_CE = opt.lambda_CE
            self.lambda_dis = opt.lambda_dis
            self.criterionCE = torch.nn.CrossEntropyLoss()
            self.discrepancyLoss = WeightDiscrepancyLoss(self.opt)
            self.BCELoss = nn.BCEWithLogitsLoss()
            self.BCELossWithWeight = nn.BCEWithLogitsLoss(reduction='none')
            self.optimizer_G =  torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.SGD(self.netD.parameters(), lr=opt.D_lr, momentum=0.9, weight_decay=0.0005)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        
    def set_input(self, input, isTrain=None):
        # set sourcc input image and label
        self.source_input = input['source_input'].to(self.device)
        self.source_label = input['source_label'].to(self.device, dtype=torch.long)
        # set target input image
        self.target_input = input['target_input'].to(self.device)
        self.target_label = input['target_label'].to(dtype=torch.long)
        self.ground_truth = torch.unsqueeze(self.target_label, dim=1)
        # # set image path
        self.image_paths = input['target_name']

        self.damping = 1-(self.curr_epoch/(self.epoch_stop+1))

    def compute_visuals(self):
        self.color_source_label = torch.unsqueeze(self.source_label, dim=1)
        self.color_target_label = torch.unsqueeze(self.target_label, dim=1)
        self.color_pred_source = torch.unsqueeze(torch.argmax(self.pred_source, dim=1), dim=1)
        self.color_pred_target = torch.unsqueeze(torch.argmax(self.pred_target, dim=1), dim=1)
        if not self.isTrain:
            self.mask = torch.unsqueeze(torch.argmax(self.pred_target, dim=1), dim=1)

    def optimize_parameters(self):
        self.optimizer_G.zero_grad()
        self.optimizer_D.zero_grad()
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