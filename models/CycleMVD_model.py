from numpy import full
import torch
import torch.nn.functional as F
import random
from torch import full_like, nn
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

class CycleMVDModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_CE', type=float, default=1.0, help='weight for L1 loss')
            parser.add_argument('--D_lr', type=float, required=True)
            parser.add_argument('--lambda_adv', type=float, default=0.1)
            parser.add_argument('--lambda_dis', type=float, default=2)
            parser.add_argument('--lambda_con', type=float, default=1)
            parser.add_argument('--netG2m', type=str, default='unet_seg')
            parser.add_argument('--netG2i', type=str, default='unet_seg')
            parser.add_argument('--netDi', type=str, default='basic')
            parser.add_argument('--netDm', type=str, default='u2d')
            parser.add_argument('--lr_G2m', type=float, default=0.00025)
            parser.add_argument('--lr_G2i', type=float, default=0.0002)
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # define models
        self.model_names = ['G2m', 'Dm', 'G2i', 'Di']
        self.netG2m = networks.define_G(opt.input_nc, opt.output_nc, netG=opt.netG2m, gpu_ids=self.gpu_ids)
        self.netG2i = networks.define_G(1, 3, netG=opt.netG2i, gpu_ids=self.gpu_ids)
        self.netDm = networks.define_D(opt.output_nc, 8, netD=opt.netDm, gpu_ids=self.gpu_ids)
        self.netDi = networks.define_D(3, 8, netD=opt.netDi, gpu_ids=self.gpu_ids)
        self.real ,self.fake= 1, 0
        self.epoch_stop = opt.n_epochs + opt.n_epochs_decay
        # visualization
        self.loss_names = ['Dm', 'Di', 'CE', 'adv_Dm', 'adv_Di', 'con', 'con_target', 'con_source']
        self.visual_names = ['source_input', 'target_input', 'color_source_label', 'color_pred_source',
         'color_pred_target', 'color_target_label', 'reg_target', 'color_pred_reg_source', 'reg_source'] 
        
        if self.isTrain:
            # define loss
            self.BCELoss = nn.BCEWithLogitsLoss()
            self.DisLoss = WeightDiscrepancyLoss(opt);
            self.CELoss = nn.CrossEntropyLoss()
            self.L1Loss = nn.L1Loss()
            # define optimizor
            self.optimizer_G2m = torch.optim.Adam(self.netG2m.parameters(), lr=opt.lr_G2m, betas=(opt.beta1, 0.999))
            self.optimizer_Dm = torch.optim.Adam(self.netDm.parameters(), lr=opt.D_lr, betas=(opt.beta1, 0.999)) 
            self.optimizer_Di = torch.optim.Adam(self.netDi.parameters(), lr=opt.D_lr, betas=(opt.beta1, 0.999))
            self.optimizer_G2i = torch.optim.Adam(self.netG2i.parameters(), lr=opt.lr_G2i, betas=(opt.beta1, 0.999)) 
            self.optimizers.append(self.optimizer_G2m)
            self.optimizers.append(self.optimizer_Dm)
            self.optimizers.append(self.optimizer_G2i)
            self.optimizers.append(self.optimizer_Di)
            # hyper param
            self.lambda_CE = opt.lambda_CE
            self.lambda_adv = opt.lambda_adv
            self.lambda_con = opt.lambda_con

    def set_input(self, input, isTrain=None):
        # set sourcc input image and label
        self.source_input = input['source_input'].to(self.device)
        self.source_label = input['source_label'].to(self.device, dtype=torch.long)
        # set target input image and label
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
        self.color_pred_reg_source = torch.unsqueeze(torch.argmax(self.pred_on_reg_source, dim=1), dim=1)
        if not self.isTrain:
            self.mask = torch.unsqueeze(torch.argmax(self.pred_target, dim=1), dim=1)

    def optimize_parameters(self):
        self.optimizer_Dm.zero_grad()
        self.optimizer_Di.zero_grad()
        self.optimizer_G2i.zero_grad()
        self.optimizer_G2m.zero_grad()
        self.set_requires_grad([self.netDi, self.netDm], True)
        self.forward()
        self.foward_D()
        self.backward_D()
        self.optimizer_Di.step()
        self.optimizer_Dm.step()
        self.set_requires_grad([self.netDi, self.netDm], False)
        self.backward_G()
        self.optimizer_G2i.step()
        self.optimizer_G2m.step()

    def set_epoch_num(self, epoch):
        self.curr_epoch = epoch

    def forward(self):
        # source 
        # self.pred_source_1, self.pred_source_2 = self.netG2m(self.source_input) # MVD
        # self.pred_source = self.pred_source_1 + self.pred_source_2 # MVD
        self.pred_source = self.netG2m(self.source_input) # UNet
        self.reg_source = self.netG2i(torch.argmax(self.pred_source, dim=1, keepdim=True).float())
        self.pred_on_reg_source = self.netG2m(self.reg_source)
        # target image2mask
        # self.pred_target1, self.pred_target2 = self.netG2m(self.target_input) #MVD
        # self.pred_target = self.pred_target1 + self.pred_target2 #MVD
        self.pred_target = self.netG2m(self.target_input)
        # target mask2image
        self.reg_target = self.netG2i(torch.argmax(self.pred_target, dim=1, keepdim=True).float())

    def foward_D(self):
        # mask space
        Dm_source = self.pred_source.detach()
        self.Dm_pred_source = self.netDm(Dm_source)
        Dm_target = self.pred_target.detach()
        self.Dm_pred_target = self.netDm(Dm_target)
        # image space
        Di_target = self.target_input.detach()
        self.Di_pred_target = self.netDi(Di_target)
        Di_reg_target = self.reg_target.detach()
        self.Di_pred_reg_target = self.netDi(Di_reg_target)
    
    def backward_D(self):
        # Dm loss
        self.loss_Dm_source = self.BCELoss(self.Dm_pred_source, torch.full_like(self.Dm_pred_source, self.real))
        self.loss_Dm_target = self.BCELoss(self.Dm_pred_target, torch.full_like(self.Dm_pred_target, self.fake))
        self.loss_Dm = (self.loss_Dm_target+self.loss_Dm_source)
        self.loss_Dm.backward()  
        # Di loss
        self.loss_Di_target = self.BCELoss(self.Di_pred_target, torch.full_like(self.Di_pred_target, self.real))
        self.loss_Di_reg_target = self.BCELoss(self.Di_pred_reg_target, torch.full_like(self.Di_pred_reg_target, self.fake))
        self.loss_Di = self.loss_Di_reg_target + self.loss_Di_target
        self.loss_Di.backward()

    def backward_G(self):
        # mask: segmentation adv
        # image: consistency adv 

        # mask
        # segmentation
        # self.loss_CE_1 = self.CELoss(self.pred_source_1, self.source_label) #MVD
        # self.loss_CE_2 = self.CELoss(self.pred_source_2, self.source_label) # MVD
        # self.loss_CE = (self.loss_CE_1 + self.loss_CE_2) * self.lambda_CE # MVD
        self.loss_CE = self.CELoss(self.pred_source, self.source_label)
        # self.loss_CE.backward()

        # adv
        adv_Dm_source = self.netDm(self.pred_source)
        adv_Dm_target = self.netDm(self.pred_target)
        self.loss_adv_Dm = self.BCELoss(adv_Dm_source, torch.full_like(adv_Dm_source, self.fake)) + \
            self.BCELoss(adv_Dm_target, torch.full_like(adv_Dm_target, self.real))
        self.loss_adv_Dm *= self.lambda_adv * self.damping * 0.1
        # self.loss_adv_Dm.backward()

        # image
        # consistency
        self.loss_con_target = self.L1Loss(self.reg_target, self.target_input)*10
        self.loss_con_source = self.L1Loss(torch.argmax(self.pred_source, dim=1).float(), \
            torch.argmax(self.pred_on_reg_source, dim=1).float())
        self.loss_con = self.loss_con_source + self.loss_con_target
        self.loss_con *= self.lambda_con;
        # self.loss_reg.backward()

        # adv
        adv_Di_real = self.netDi(self.target_input)
        adv_Di_fake = self.netDi(self.reg_target)
        self.loss_adv_Di = self.BCELoss(adv_Di_real, torch.full_like(adv_Di_real, self.fake)) + \
            self.BCELoss(adv_Di_fake, torch.full_like(adv_Di_fake, self.real))
        self.loss_adv_Di *= self.lambda_adv
        # self.loss_adv_Di.backward()

        self.loss_G = self.loss_CE + self.loss_adv_Di + self.loss_adv_Dm + self.loss_con
        self.loss_G.backward()

    def validation(self, data):
        pass