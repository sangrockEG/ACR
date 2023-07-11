from cmath import isnan
import random
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pytorch_lightning as pl


# Image tools
from matplotlib import pyplot as plt

from tools import utils, pyutils
from tools.imutils import save_img, denorm, _crf_with_alpha, cam_on_image
from evaluation_precision import eval_in_script

# import resnet38d
from networks import resnet38d

class Exp(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.save_hyperparameters()
        self.automatic_optimization = False
        self.check_sanity = True

        self.args = args
        self.categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                           'bus', 'car', 'cat', 'chair', 'cow',
                           'diningtable', 'dog', 'horse', 'motorbike', 'person',
                           'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

        # Hyper-params
        self.grid = args.grid
        self.prob = args.prob
        self.W = args.W

        # Loss & Metric
        self.bce = nn.BCEWithLogitsLoss()
        self.L1 = nn.L1Loss()
        self.max_miou = 0
        self.right_count = 0
        self.wrong_count = 0

        self.build_framework(args.phase)

    def build_framework(self, phase):

        args = self.args

        # Network
        self.net_recon = resnet38d.Net_recon(args.D)

        self.net_recon.load_state_dict(resnet38d.convert_mxnet_to_torch('./pretrained/resnet_38d.params'), strict=False)

        self.net_main = resnet38d.Net_main(args.D)
        self.net_main.load_state_dict(resnet38d.convert_mxnet_to_torch('./pretrained/resnet_38d.params'), strict=False)

        if phase=='train':

            print('define optimizers')

            # Optimizer
            param_recon = self.net_recon.get_parameter_groups()
            self.opt_recon = utils.PolyOptimizer([
                {'params': param_recon[0], 'lr': 1 * args.lr, 'weight_decay': args.wt_dec},
                {'params': param_recon[1], 'lr': 2 * args.lr, 'weight_decay': 0},  # non-scratch bias
                {'params': param_recon[2], 'lr': 10 * args.lr, 'weight_decay': args.wt_dec},  # scratch weight
                {'params': param_recon[3], 'lr': 20 * args.lr, 'weight_decay': 0}  # scratch bias
            ],
                lr=args.lr, weight_decay=args.wt_dec, max_step=args.max_step)

            param_main = self.net_main.get_parameter_groups()
            self.opt_main = utils.PolyOptimizer([
                {'params': param_main[0], 'lr': 1 * args.lr, 'weight_decay': args.wt_dec},
                {'params': param_main[1], 'lr': 2 * args.lr, 'weight_decay': 0},  # non-scratch bias
                {'params': param_main[2], 'lr': 10 * args.lr, 'weight_decay': args.wt_dec},  # scratch weight
                {'params': param_main[3], 'lr': 20 * args.lr, 'weight_decay': 0}  # scratch bias
            ],
                lr=args.lr, weight_decay=args.wt_dec, max_step=args.max_step)

    def training_step(self, batch, batch_idx):
        
        tb = self.logger.experiment

        img = batch['img']  # B x 3 x H x W
        name = batch['name']
        label = batch['label']

        B = img.shape[0]
        H = img.shape[2]
        W = img.shape[3]
        C = 20  # Number of cls

        # Bring the strategy of class-specific erasing (OC-CSE)
        label_t, label_nt = self.split_label(label, B) # t:target, nt:non-target
        class_t = label_t.nonzero()[:,1]
            
        ################################### Train the recon network ###################################

        self.opt_recon.zero_grad()
        self.net_recon.train()
        self.net_main.eval()

        mask_small = torch.randint(0, 100, size=(B,1,self.grid,self.grid))<self.prob + self.current_epoch
        mask_grid = F.interpolate(mask_small.float(), size=[H,W], mode='nearest').cuda(self.device) # For VRF

        with torch.no_grad():
            cam_small, _, _ = self.net_main(img)
            cam_small = self.max_norm(cam_small)*label.view(B,20,1,1)
            cam = F.interpolate(cam_small, size=[H,W], mode='bilinear', align_corners=False)

            cam_t = cam[label_t==1].unsqueeze(1)
            segment_t_fuse_rev = cam_t.float()*mask_grid
            segment_t_fuse = 1 - segment_t_fuse_rev # Assuring virtual remnant

            cam_nt = 1-cam_t
            segment_nt_fuse_rev = cam_nt*mask_grid
            segment_nt_fuse = 1-segment_nt_fuse_rev # Assuring virtual remnant

            valid = (torch.sum(segment_t_fuse.view(B,-1),dim=-1))>50

        cam_small, pred, feat_small = self.net_recon(img)

        feat = F.interpolate(feat_small, size=[H,W], mode='bilinear', align_corners=False)
        
        feat_masked_t = feat * segment_t_fuse
        feat_masked_nt = feat * segment_nt_fuse

        # no-color
        img_recon_full = self.net_recon.recon_decoder(feat.detach(), use_tanh=False) # For decoder only
        img_recon_from_t = self.net_recon.recon_decoder(feat_masked_t, use_tanh=False)
        img_recon_from_nt = self.net_recon.recon_decoder(feat_masked_nt, use_tanh=False)

        loss_recon_cls = self.bce(pred,label)
        loss_recon_full = self.L1(img_recon_full, img)
        loss_recon_t =  self.masked_L1(img_recon_from_t, img, segment_t_fuse_rev, valid) # Apply loss on reverse region only
        loss_recon_nt = self.masked_L1(img_recon_from_nt, img, segment_nt_fuse_rev)

        loss_recon = self.W[0]*loss_recon_cls + self.W[1]*loss_recon_full + 0.3*loss_recon_t + 0.5*loss_recon_nt

        loss_recon.backward()

        if torch.isnan(loss_recon):
            print('reconNaN!')
        else:       
            self.opt_recon.step()

        ################################### Train the main network ###################################

        self.opt_main.zero_grad()
        self.net_main.train()
        self.net_recon.eval()

        with torch.no_grad():
            _, _, feat_small = self.net_recon(img)
            feat = F.interpolate(feat_small, size=[H,W], mode='bilinear', align_corners=False)

        cam_small, pred, _ = self.net_main(img)
        seg_pred = torch.argmax(F.interpolate(cam_small,(H,W),mode='bilinear',align_corners=False), dim=1).unsqueeze(1)

        cam_small = self.max_norm(cam_small)*label.view(B,20,1,1)
        cam = F.interpolate(cam_small, size=[H,W], mode='bilinear', align_corners=False)

        segment_t = cam[label_t==1].unsqueeze(1) # Soft
        segment_nt = 1-segment_t

        feat_t = feat.detach() * segment_t # Train main network only, while freezing recon network.
        feat_nt = feat.detach() * segment_nt

        img_recon_from_t = self.net_recon.recon_decoder(feat_t, use_tanh=False)
        img_recon_from_nt = self.net_recon.recon_decoder(feat_nt, use_tanh=False)

        eval_mask_for_t = (segment_t==0).float().detach() # Hard
        eval_mask_for_nt = segment_t.clone().detach() * (seg_pred==class_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)).float() # Soft

        loss_main_cls = self.bce(pred,label)
        loss_main_t = -self.masked_L1(img_recon_from_t, img, eval_mask_for_t)
        loss_main_nt = -self.masked_L1(img_recon_from_nt, img, eval_mask_for_nt)
        loss_main = loss_main_cls + self.args.W[2]*loss_main_t + self.args.W[3]*loss_main_nt  
        loss_main.backward()

        if torch.isnan(loss_main):
            print('mainNaN!')
        else:       
            self.opt_main.step()

        ######################################################################################################################
        ####################################################### Export #######################################################
        ######################################################################################################################

        losses = {}
        losses['loss_recon'] = loss_recon
        losses['loss_recon_cls'] = loss_recon_cls
        losses['loss_recon_full'] = loss_recon_full
        losses['loss_recon_mask'] = loss_recon_t
        losses['loss_main'] = loss_main
        losses['loss_main_cls'] = loss_main_cls
        losses['loss_main_t'] = loss_main_t
        losses['loss_main_nt'] = loss_main_nt

        ##

        self.count_rw(pred, label, label.shape[0])

        if batch_idx%50==0:
            key_list = list(losses.keys())
            for i, key in enumerate(key_list):
                tb.add_scalar('running/' + key, losses[key], self.global_step)

            acc = 100 * self.right_count / (self.right_count + self.wrong_count)
            self.log('running/acc', acc)

        if batch_idx%300==0:

            gt = label_t[0].cpu().detach().numpy()
            gt_cls = np.nonzero(gt)[0]

            img = denorm(img[0])
            img_masked_fuse = img*segment_t_fuse[0]
            img_masked_fuse_rev = img*segment_t_fuse_rev[0]

            img_t = img*segment_t[0]
            img_nt = img*segment_nt[0]

            img_recon_full = denorm(img_recon_full[0])
            img_recon_full = self.clip(img_recon_full)
            img_recon_from_t = denorm(img_recon_from_t[0])
            img_recon_from_t = self.clip(img_recon_from_t)
            img_recon_mask_rev = img_recon_from_t*segment_t_fuse_rev[0]

            img_recon_from_t = denorm(img_recon_from_t[0])
            img_recon_from_t = self.clip(img_recon_from_t)
            img_recon_from_t_rev = img_recon_from_t*eval_mask_for_t[0]

            img_recon_from_nt = denorm(img_recon_from_nt[0])
            img_recon_from_nt = self.clip(img_recon_from_nt)
            img_recon_from_nt_rev = img_recon_from_nt*eval_mask_for_nt[0]

            cam = cam[0,gt_cls[0]].cpu().detach().numpy()

            cam_temp = cam_on_image(img.cpu().detach().numpy(), cam)/255
            tb_img_1 = torch.cat((img, img_recon_full, torch.from_numpy(cam_temp).to(self.device)), dim=1)
            tb_img_2 = torch.cat((img_masked_fuse, img_recon_from_t, img_recon_mask_rev), dim=1)
            tb_img_3 = torch.cat((img_t, img_recon_from_t, img_recon_from_t_rev), dim=1)
            tb_img_4 = torch.cat((img_nt, img_recon_from_nt, img_recon_from_nt_rev), dim=1)
            
            tb_img = torch.cat((tb_img_1, tb_img_2, tb_img_3, tb_img_4), dim=2)            
            tb.add_image('0_train', tb_img, self.current_epoch)

        return losses
    
    def validation_step(self, batch, batch_idx):
        
        tb = self.logger.experiment

        name = batch[0][0]
        imgs = batch[1]
        label = batch[2]

        num_scales = len(imgs)//2
        img_ori = imgs[2]
        B = img_ori.shape[0]
        H = img_ori.shape[2]
        W = img_ori.shape[3]
        
        gt = label[0].cpu().detach().numpy()
        gt_cls = np.nonzero(gt)[0]

        self.net_main.eval()
        
        cam = torch.zeros(1,20,H,W).to(self.device)

        with torch.no_grad():
            for i in range(num_scales):
                img_temp = imgs[2*i]
                img_temp_flip = imgs[2*i+1]
                img_cat = torch.cat((img_temp, img_temp_flip), dim=0)
                
                cam_cat, _, _ = self.net_main(img_cat)

                cam_cat = F.interpolate(cam_cat, (H, W), mode='bilinear', align_corners=False)
                cam_cat = F.relu(cam_cat)
                
                cam_temp, cam_temp_flip = torch.split(cam_cat,(1,1),dim=0)
                cam_temp_flip = torch.flip(cam_temp_flip, (3,)) # Flip along horizontal axis
                
                cam += cam_temp
                cam += cam_temp_flip


        cam = cam * label.view(1, 20, 1, 1)
        cam = self.max_norm(cam)
        cam_np = cam.cpu().detach().numpy()[0] # C x H x W

        cam_dict = {}
        for i in range(20):
            if label[0, i] > 1e-5:
                cam_dict[i] = cam_np[i]

        ## Export
        # Save CAM dict
        np.save(tb.log_dir+'/dict/'+ name + '.npy', cam_dict)

        # Visualize CAM
        if not self.check_sanity:
            if batch_idx<50:
                input = denorm(img_ori[0])
                for c in gt_cls:
                    temp = cam_on_image(input.cpu().detach().numpy(), cam_np[c])
                    tb.add_image(name+'/'+self.categories[c], temp, self.current_epoch)

    def training_epoch_end(self, training_step_outputs):       
        
        phase = 'train'
        step_outputs = training_step_outputs

        key_list = list(step_outputs[0].keys())
        if 'loss' in key_list:
            key_list.remove('loss')
        loss_list = [0]*len(key_list)

        for output in step_outputs:
            for i, key in enumerate(key_list):
                loss_list[i] += output[key]

        for i, key in enumerate(key_list):
            self.log(phase + '/' + key, loss_list[i] / len(step_outputs))
        
        acc = 100 * self.right_count / (self.right_count + self.wrong_count)
        self.log(phase + '/acc', acc)

    def validation_epoch_end(self, val_step_outputs):

        tb = self.logger.experiment
        phase = 'val'

        if self.check_sanity:
            self.check_sanity = False
            return
        
        metric_dict = eval_in_script(eval_list='train', pred_dir=tb.log_dir+'/dict')
        th = metric_dict['th']
        miou = metric_dict['miou']
        mp = metric_dict['mp']
        mr = metric_dict['mr']

        print('Epo ' + str(self.current_epoch).zfill(2) + 
            ' : miou=' + str(round(miou,2)) + 
            ', mP=' + str(round(mp,2)) +
            ', mR=' + str(round(mr,2)) + 
            'at th ' + str(round(th,2))
            )

        if self.max_miou < miou:
            self.max_miou = miou
            print('New record!')

        self.log('val_miou', miou, self.current_epoch)
        tb.add_scalar('metrics/mIoU', miou, self.current_epoch)
        tb.add_scalar('metrics/mP', mp, self.current_epoch)
        tb.add_scalar('metrics/mR', mr, self.current_epoch)
        tb.add_scalar('metrics/th', th, self.current_epoch)


    ## INFER-RELATED
    
    def load_pretrained(self, load_path):
        self.load_pl_dict(self.net_main, load_path, 'net_main')

    def infer(self, batch, vis_path=None, cam_path=None, crf_path=None, alphas=[4,24]):

        # name, msf_img_list, label
        name = batch[0][0]
        imgs = batch[1]
        label = batch[2]

        num_scales = len(imgs)//2
        img_ori = imgs[2]
        B = img_ori.shape[0]
        H = img_ori.shape[2]
        W = img_ori.shape[3]
        
        with torch.no_grad():
            gt = label[0].cpu().detach().numpy()
            gt_cls = np.nonzero(gt)[0]

            self.net_main.eval()
            
            cam = torch.zeros(1,20,H,W).to(self.device)
        
            for i in range(num_scales):
                img_temp = imgs[2*i]
                img_temp_flip = imgs[2*i+1]
                img_cat = torch.cat((img_temp, img_temp_flip), dim=0)
                
                cam_cat, _, _ = self.net_main(img_cat)

                cam_cat = F.interpolate(cam_cat, (H, W), mode='bilinear', align_corners=False)
                cam_cat = F.relu(cam_cat)
                
                cam_temp, cam_temp_flip = torch.split(cam_cat,(1,1),dim=0)
                cam_temp_flip = torch.flip(cam_temp_flip, (3,)) # Flip along horizontal axis
                
                cam += cam_temp
                cam += cam_temp_flip

            cam = cam * label.view(1, 20, 1, 1)
            cam = self.max_norm(cam)
            cam_np = cam.cpu().detach().numpy()[0] # C x H x W

            cam_dict = {}
            for i in range(20):
                if label[0, i] > 1e-5:
                    cam_dict[i] = cam_np[i]

            ## Export
            if vis_path is not None:
                img_np = denorm(img_ori[0]).cpu().detach().data.permute(1, 2, 0).numpy()
                for c in gt_cls:
                    save_img(vis_path + '/' + name + '_cam_' + self.categories[c] + '.png', img_np, cam_np[c])
            
            if cam_path is not None:
                np.save(cam_path + '/' + name + '.npy', cam_dict)

            if crf_path is not None:
                for a in alphas:
                    crf_dict = _crf_with_alpha(cam_dict, name, alpha=a)
                    np.save(crf_path + '/' + str(a).zfill(2) + '/' + name + '.npy', crf_dict)

    ################ Functions #################   
    def max_norm(self, cam):
        N, C, H, W = cam.size()
        cam = F.relu(cam)
        max_v = torch.max(cam.view(N, C, -1), dim=-1)[0].view(N, C, 1, 1)
        min_v = torch.min(cam.view(N, C, -1), dim=-1)[0].view(N, C, 1, 1)
        cam = F.relu(cam - min_v - 1e-5) / (max_v - min_v + 1e-5)
        return cam

    def masked_L1(self, target, img, mask, valid=None):
        if valid==None:
            return self.L1(target*mask, img*mask)
        else:
            if valid.sum()==0:
                return 0
            else:
                return self.L1((target*mask)[valid], (img*mask)[valid])

    def clip(self, img):
        img[img<0] = 0
        img[img>1] = 1
        return img

    def split_label(self, label, B):
        label_mask= torch.zeros_like(label)
        for i in range(B):
            label_idx = torch.nonzero(label[i], as_tuple=False)
            rand_idx = torch.randint(0, len(label_idx), (1,))
            target = label_idx[rand_idx][0]
            label_mask[i, target] = 1
        label_remain = label - label_mask
        return label_mask, label_remain

    def count_rw(self, pred, label, bs):
        for b in range(bs):
            gt = label[b].cpu().detach().numpy()
            gt_cls = np.nonzero(gt)[0]
            num = len(np.nonzero(gt)[0])
            pred_t = pred[b].cpu().detach().numpy()
            pred_cls = pred_t.argsort()[-num:][::-1]

            for c in gt_cls:
                if c in pred_cls:
                    self.right_count += 1
                else:
                    self.wrong_count += 1

    def load_pl_dict(self, net, pl_dict_path, name):
        pl_dict= torch.load(pl_dict_path)['state_dict']
        len_name = len(name)
        temp_dict = {}
        for key in pl_dict.keys():
            if key[:len_name]==name:
                key_name = key[len_name+1:] # Remove "xxx." at the first of each key
                temp_dict[key_name] = pl_dict[key]
        net.load_state_dict(temp_dict, strict=True)

    def normalize_T(self, img):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        # Assume that img is in the range of [-1,1]
        img = (img+1)/2. # Now in [0,1]
        for i in range(3):
            img[:,i,:,:] = img[:,i,:,:] - mean[i]
            img[:,i,:,:] = img[:,i,:,:]/std[i]
        
        return img 
    ############################################

    def configure_optimizers(self):
        return None