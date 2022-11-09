from numpy import full
import torch
import torch.nn.functional as F
import random
from torch import nn

from .base_model import BaseModel
from . import networks

Eposilon = 0.4
Lambda_local = 40
Lambda_adv = 0.001
Lambda_weight = 0.01


class WeightedBCEWithLogitsLoss(nn.Module):
    
    def __init__(self, size_average=True):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.size_average = size_average
        
    def weighted(self, input, target, weight, alpha, beta):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
                
        if weight is not None:
            loss = alpha * loss + beta * loss * weight

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
    
    def forward(self, input, target, weight, alpha, beta):
        if weight is not None:
            return self.weighted(input, target, weight, alpha, beta)
        else:
            return self.weighted(input, target, None, alpha, beta)

class WeightDiscrepancyLoss(nn.Module):
    def __init__(self, opt):
        super(WeightDiscrepancyLoss, self).__init__()
        self.w1 = None
        self.w2 = None
        self.opt = opt

    def forward(self, para1, para2, epoch):
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

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def lr_warmup(base_lr, iter, warmup_iter):
    return base_lr * ((float(iter) + 1) / warmup_iter)

class CLANModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--D_lr', type=float, required=True)
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # attributes for visualization 
        self.loss_names = ['CE', 'adv', 'discrepancy', 'D_target', 'D_source'] # need modification
        if opt.phase == 'training':
            self.visual_names = ['source_input', 'target_input', 'color_source_label', 'color_pred_source', 'color_pred_target', 'color_target_label'] # 
        else:
            self.visual_names = ['target_input', 'ground_truth', 'color_target_label', 'mask', 'color_pred_target']
        # models
        self.model_names = ['G', 'D']
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, netG=opt.netG, gpu_ids=self.gpu_ids)
        self.netD = networks.define_D(opt.output_nc, 1, netD=opt.netD, gpu_ids=self.gpu_ids)
        self.source_id, self.target_id = 0, 1



        if self.isTrain:  # define discriminators
            self.PREHEAT_STEPS = opt.n_epochs
            self.NUM_STEPS = opt.n_epochs + opt.n_epochs_decay
            self.criterionCE = torch.nn.CrossEntropyLoss()
            self.discrepancyLoss = WeightDiscrepancyLoss(self.opt)
            self.BCELoss = nn.BCEWithLogitsLoss()
            self.WeightedBCELoss = WeightedBCEWithLogitsLoss()
            self.optimizer_G = torch.optim.SGD(self.netG.module.optim_parameters(opt), lr=opt.lr, momentum=0.9, weight_decay=0.0005)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.D_lr, betas=(0.9, 0.999))
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

        self.interp_source = nn.Upsample(size=(self.source_input.size(dim=2), self.source_input.size(dim=3)), mode='bilinear', align_corners=True)
        self.interp_target = nn.Upsample(size=(self.target_input.size(dim=2), self.target_input.size(dim=3)), mode='bilinear', align_corners=True) 

    def compute_visuals(self):
        self.color_source_label = torch.unsqueeze(self.source_label, dim=1)
        self.color_target_label = torch.unsqueeze(self.target_label, dim=1)
        self.color_pred_source = torch.unsqueeze(torch.argmax(self.pred_source, dim=1), dim=1)
        self.color_pred_target = torch.unsqueeze(torch.argmax(self.pred_target, dim=1), dim=1)
        if not self.isTrain:
            self.mask = torch.unsqueeze(torch.argmax(self.pred_target, dim=1), dim=1)

    def optimize_parameters(self):
        self.forward()
        self.optimizer_D.step()
        self.optimizer_G.step()

    def set_epoch_num(self, epoch):
        self.curr_epoch = epoch

    def forward(self):        
        self.optimizer_G.zero_grad()
        self.optimizer_D.zero_grad()
        self.damping = 1 - self.curr_epoch / (self.NUM_STEPS + 1)
        self.set_requires_grad([self.netD], False)
        # source image:
        self.pred_source_1, self.pred_source_2 = self.netG(self.source_input)
        self.pred_source_1 = self.interp_source(self.pred_source_1)
        self.pred_source_2 = self.interp_source(self.pred_source_2)
        self.pred_source = self.pred_source_1 + self.pred_source_2
        
        if self.isTrain:
            self.loss_CE_1 = self.criterionCE(self.pred_source_1, self.source_label)
            self.loss_CE_2 = self.criterionCE(self.pred_source_2, self.source_label)
            self.loss_CE = (self.loss_CE_1 + self.loss_CE_2)
            self.loss_CE.backward()

        self.pred_target_1, self.pred_target_2 = self.netG(self.target_input)
        self.pred_target_1 = self.interp_target(self.pred_target_1)
        self.pred_target_2= self.interp_target(self.pred_target_2)
        self.weight_map = self.weighted_map(F.softmax(self.pred_target_1, dim=1), F.softmax(self.pred_target_2, dim=1))
        self.pred_target = self.pred_target_1 + self.pred_target_2
        self.D_out_t = self.interp_target(self.netD(F.softmax(self.pred_target, dim=1)))
        
        if self.isTrain:
            if self.curr_epoch > self.PREHEAT_STEPS:
                self.loss_adv = self.WeightedBCELoss(self.D_out_t, torch.full_like(self.D_out_t, self.target_id), self.weight_map, Eposilon, Lambda_local)
            else:
                self.loss_adv = self.BCELoss(self.D_out_t, torch.full_like(self.D_out_t, self.target_id))
            self.loss_adv = self.loss_adv * Lambda_adv * self.damping
            self.loss_adv.backward()

        if self.isTrain:
            self.loss_discrepancy = self.discrepancyLoss(self.netG.module.layer5.parameters(), self.netG.module.layer6.parameters(), self.curr_epoch) * Lambda_weight * 2
            self.loss_discrepancy.backward()

        self.set_requires_grad([self.netD], True)
        source = self.pred_source.detach()
        self.D_out_s = self.interp_source(self.netD(F.softmax(source, dim=1)))
        self.loss_D_source = self.BCELoss(self.D_out_s, torch.full_like(self.D_out_s, self.source_id)) 
        self.loss_D_source.backward()
        target = self.pred_target.detach()
        weight_map = self.weight_map.detach()
        self.D_out_t = self.interp_target(self.netD(F.softmax(target, dim=1)))
        if self.isTrain:
            if self.curr_epoch > self.PREHEAT_STEPS:
                self.loss_D_target = self.WeightedBCELoss(self.D_out_t, torch.full_like(self.D_out_t, self.target_id), weight_map, Eposilon, Lambda_local)
            else:
                self.loss_D_target = self.BCELoss(self.D_out_t, torch.full_like(self.D_out_t, self.target_id))
            self.loss_D_target.backward()

    def weighted_map(self, pred1, pred2):
        output = 1.0 - torch.sum((pred1 * pred2), 1).view(-1, 1, pred1.size(2), pred1.size(3)) / \
        (torch.norm(pred1, 2, 1) * torch.norm(pred2, 2, 1)).view(-1, 1, pred1.size(2), pred1.size(3))
        return output

    def validation(self, data):
        pass

    def update_learning_rate(self):
        old_lr = self.optimizers[0].param_groups[0]['lr']
        old_D_lr = self.optimizers[1].param_groups[0]['lr']
        print(self.curr_epoch, self.opt.lr)
        if self.curr_epoch < self.PREHEAT_STEPS:
            lr = lr_warmup(self.opt.lr, self.curr_epoch, self.PREHEAT_STEPS)
            D_lr = lr_warmup(self.opt.D_lr, self.curr_epoch, self.PREHEAT_STEPS)
        else:
            lr = lr_poly(self.opt.lr, self.curr_epoch, self.NUM_STEPS, 0.9)
            D_lr = lr_poly(self.opt.D_lr, self.curr_epoch, self.NUM_STEPS, 0.9)
        self.optimizers[0].param_groups[0]['lr'] = lr
        self.optimizers[1].param_groups[0]['lr'] = D_lr
        if len(self.optimizers[1].param_groups) > 1:
            self.optimizers[1].param_groups[1]['lr'] = D_lr * 10
        if len(self.optimizers[0].param_groups) > 1:
            self.optimizers[0].param_groups[1]['lr'] = lr * 10
        print('learning rate G: %.7f -> %.7f, D:%.7f -> %.7f' % (old_lr, lr, old_D_lr, D_lr))