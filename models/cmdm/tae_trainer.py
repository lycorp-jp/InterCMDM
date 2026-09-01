import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from os.path import join as pjoin
import torch.nn.functional as F

import torch.optim as optim

import time
import numpy as np
from collections import OrderedDict, defaultdict
from utils.utils import print_current_loss, WandbLogger
from models.losses import *
import os
import sys

from einops import rearrange

def def_value():
    return 0.0


class TAETrainer:
    def __init__(self, args, tae_model, transformer=None):
        self.opt = args
        self.tae_model = tae_model
        self.device = args.device
        
        if args.is_train:
            self.logger = WandbLogger(args)
        
        if transformer is not None:
            self.trans = transformer

    def forward(self, batch_data):
        motions = batch_data.detach().to(self.device).float()
        if self.opt.dataset_name == "interx":
            j, d = motions.shape[-2:]
            motions = rearrange(motions, 'b l j d -> b l (j d)')

        motions = rearrange(motions, 'b l d -> b l 1 d')
        pred_motion, loss_kl = self.tae_model(motions, need_loss=True)

        motions = rearrange(motions, 'b l 1 d -> b l d')
        pred_motion = rearrange(pred_motion, 'b l 1 d -> b l d')

        if self.opt.dataset_name == "interx":
            motions = rearrange(motions, 'b l (j d) -> b l j d', j=j, d=d)
            pred_motion = rearrange(pred_motion, 'b l (j d) -> b l j d', j=j, d=d)

        loss_rec = torch.nn.L1Loss()(motions, pred_motion)
        loss_explicit = torch.nn.L1Loss()(motions[..., :3], pred_motion[..., :3])
        loss_vel = torch.nn.L1Loss()(motions[..., 3:], pred_motion[..., 3:])
        loss_bn = torch.tensor(0.0)
        loss_geo = torch.tensor(0.0)
        loss_fc = torch.tensor(0.0)
        loss = loss_kl

        return loss, loss_rec, loss_explicit, loss_vel, loss_bn, loss_geo, loss_fc, loss_kl


    # @staticmethod
    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):

        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_tae_model.param_groups:
            param_group["lr"] = current_lr

        return current_lr

    def save(self, file_name, ep, total_it):
        state = {
            "tae_model": self.tae_model.state_dict(),
            "opt_tae_model": self.opt_tae_model.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        self.tae_model.load_state_dict(checkpoint['tae_model'])
        self.opt_tae_model.load_state_dict(checkpoint['opt_tae_model'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_loader, val_loader, test_loader, eval_wrapper, plot_eval=None):
        self.tae_model.to(self.device)

        total_iters = self.opt.max_epoch * len(train_loader)
        print(f'\nTotal Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))
        self.opt.warm_up_iter = len(train_loader)//4
        self.opt.log_every = len(train_loader)//10
        self.opt.save_latest = len(train_loader)//2
        print(f'Warm Up Iters: {self.opt.warm_up_iter}, Log Every: {self.opt.log_every} iters, Save every: {self.opt.save_latest} iters')
        
        self.opt.milestones = [int(total_iters*0.7), int(total_iters*0.85)]
        print(f"LR milestones: {self.opt.milestones}\n")

        self.opt_tae_model = optim.AdamW(self.tae_model.parameters(), lr=self.opt.lr, betas=(0.9, 0.99), weight_decay=self.opt.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt_tae_model, milestones=self.opt.milestones, gamma=self.opt.gamma)

        epoch = 0
        it = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)
            print("Load model epoch:%d iterations:%d"%(epoch, it))

        start_time = time.time()
        
        current_lr = self.opt.lr
        logs = defaultdict(def_value, OrderedDict())

        min_val_loss = np.inf
        min_fid = np.inf
        max_top1 = -np.inf

        if self.opt.do_eval:
                eval_file = pjoin(self.opt.eval_dir, 'evaluation_training.log')
        
        while epoch <= self.opt.max_epoch:
            epoch += 1
            self.tae_model.train()
            for i, batch_data in enumerate(train_loader):
                it += 1
                if it < self.opt.warm_up_iter:
                    current_lr = self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)
                loss, loss_rec, loss_explicit, loss_vel, loss_bn, loss_geo, loss_fc, loss_kl = self.forward(batch_data)
                self.opt_tae_model.zero_grad()
                loss.backward()
                clip_grad_norm_(self.tae_model.parameters(), max_norm=1.0)
                self.opt_tae_model.step()

                if it >= self.opt.warm_up_iter:
                    self.scheduler.step()
                
                logs['loss'] += loss.item()
                logs['loss_rec'] += loss_rec.item()
                logs['loss_explicit'] += loss_explicit.item()
                logs['loss_vel'] += loss_vel.item()
                logs['loss_bn'] += loss_bn.item()
                logs['loss_geo'] += loss_geo.item()
                logs['loss_fc'] += loss_fc.item()
                logs['loss_kl'] += loss_kl.item()
                logs['lr'] += self.opt_tae_model.param_groups[0]['lr']

                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    for tag, value in logs.items():
                        self.logger.add_scalar('Train/%s'%tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            print('Validation time:')
            self.tae_model.eval()
            val_loss_rec = []
            val_loss_explicit = []
            val_loss_vel = []
            val_loss_bn = []
            val_loss_geo = []
            val_loss_fc = []
            val_loss_kl = []
            val_loss = []
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    loss, loss_rec, loss_explicit, loss_vel, loss_bn, loss_geo, loss_fc, loss_kl = self.forward(batch_data)
                    val_loss.append(loss.item())
                    val_loss_rec.append(loss_rec.item())
                    val_loss_explicit.append(loss_explicit.item())
                    val_loss_vel.append(loss_vel.item())
                    val_loss_bn.append(loss_bn.item())
                    val_loss_geo.append(loss_geo.item())
                    val_loss_fc.append(loss_fc.item())
                    val_loss_kl.append(loss_kl.item())

            self.logger.add_scalar('Val/loss', sum(val_loss) / len(val_loss), it)
            self.logger.add_scalar('Val/loss_rec', sum(val_loss_rec) / len(val_loss_rec), it)
            self.logger.add_scalar('Val/loss_explicit', sum(val_loss_explicit) / len(val_loss_explicit), it)
            self.logger.add_scalar('Val/loss_vel', sum(val_loss_vel) / len(val_loss_vel), it)
            self.logger.add_scalar('Val/loss_bn', sum(val_loss_bn) / len(val_loss_bn), it)
            self.logger.add_scalar('Val/loss_geo', sum(val_loss_geo) / len(val_loss_geo), it)
            self.logger.add_scalar('Val/loss_fc', sum(val_loss_fc) / len(val_loss_fc), it)
            self.logger.add_scalar('Val/loss_kl', sum(val_loss_kl) / len(val_loss_kl), it)

            print('Validation Loss: %.5f Reconstruction: %.5f, Explicit: %.5f, Velocity: %.5f, Bone Length: %.5f, Geodesic: %.5f, Foot Contact: %.5f, Commit: %.5f' %
                  (sum(val_loss)/len(val_loss), sum(val_loss_rec)/len(val_loss), sum(val_loss_explicit)/len(val_loss), 
                   sum(val_loss_vel)/len(val_loss), sum(val_loss_bn)/len(val_loss), sum(val_loss_geo)/len(val_loss),
                   sum(val_loss_fc) / len(val_loss), sum(val_loss_kl)/len(val_loss)))

            if sum(val_loss) / len(val_loss) < min_val_loss:
                min_val_loss = sum(val_loss) / len(val_loss)
                self.save(pjoin(self.opt.model_dir, 'finest.tar'), epoch, it)
                print('Best Validation Model So Far!~')
            
            # Evaluation during training disabled for TAE (use eval_tae.py for offline evaluation)
            # if self.opt.do_eval:
            #     self.tae_model.eval()
            #     fid, mat, top1 = evaluation_during_training(self.opt, self.tae_model, test_loader, eval_wrapper, epoch, eval_file)
            #     self.logger.add_scalar('Test/FID', fid, epoch)
            #     self.logger.add_scalar('Test/Matching', mat, epoch)
            #     self.logger.add_scalar('Test/Top1', top1, epoch)
            #
            #     if fid < min_fid:
            #         min_fid = fid
            #         self.save(pjoin(self.opt.model_dir, 'best_fid.tar'), epoch, it)
            #         print('Best FID Model So Far!~')
            #     if top1 > max_top1:
            #         max_top1 = top1
            #         self.save(pjoin(self.opt.model_dir, 'best_top1.tar'), epoch, it)
            #         print('Best Top1 Model So Far!~')
            #     print('\n')