import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.networks.blocks.loss import hinge_loss, MA_GP


class Trainer:
    def __init__(self, data_loader, generator, discriminator, max_epoch, bs):
        self.train_loader, self.valid_loader = data_loader
        self.g = generator
        self.d = discriminator
        self.g_optim = torch.optim.Adam(self.g.parameters(), lr=0.0001, betas=(0.0, 0.9))
        self.d_optim = torch.optim.Adam(self.d.parameters(), lr=0.0004, betas=(0.0, 0.9))

        self.max_epoch = max_epoch
        self.bs = bs

        current_time = time.localtime()
        self.time_code = "".join([str(current_time.tm_year), str(current_time.tm_mon).zfill(2),
                                  str(current_time.tm_mday).zfill(2), str(current_time.tm_hour).zfill(2),
                                  str(current_time.tm_min).zfill(2)])

        self.writer = SummaryWriter(log_dir=rf"runs\{self.time_code}")
        Path(fr"runs\{self.time_code}\weights").mkdir(parents=True, exist_ok=True)

    def train(self):
        num_batch = self.train_loader.dataset.__len__()/self.bs
        for epoch in range(self.max_epoch):
            _t = time.time()
            t = _t

            g_loss = 0
            d_loss = 0

            i = 1
            for i, data in enumerate(self.train_loader):
                img, annot = data
                bs = img.shape[0]

                # sample = np.random.choice([0, 1], bs).astype(bool)
                # img_t = img.cuda()[sample].requires_grad_()
                # annot = annot[sample].requires_grad_()

                img_t = img.cuda().requires_grad_()
                annot = annot.requires_grad_()

                # Train discriminator
                img_f = self.g(annot)

                pred_t = self.d(img_t, annot)
                loss_t = hinge_loss(pred_t, 1)

                pred_m = self.d(torch.cat([img_t[1:], img_t[0:1]]), annot)
                loss_m = hinge_loss(pred_m, 0)

                pred_f = self.d(img_f.detach(), annot)
                loss_f = hinge_loss(pred_f, 0)

                loss_d = loss_t + (loss_m + loss_f)/2 + MA_GP(img_t, annot, pred_t)

                self.d_optim.zero_grad()
                loss_d.backward()
                self.d_optim.step()

                d_loss += loss_d.detach().cpu().item() * bs/self.bs

                # Train discriminator
                loss_g = -self.d(img_f, annot).mean()

                self.g_optim.zero_grad()
                loss_g.backward()
                self.g_optim.step()

                g_loss += loss_g.detach().cpu().item() * bs/self.bs

                # record
                # time_used = time.time() - t
                # print(time_used)
                # t = time.time()

            self.writer.add_scalar("D Loss", g_loss/num_batch, epoch)
            self.writer.add_scalar("G Loss", d_loss/num_batch, epoch)
            self.writer.flush()

    def validate(self):
        pass

    def save(self):
        directory = fr"runs\{self.time_code}\weights"
        torch.save(self.g.state_ditc(), fr"{directory}\g.pt")
        torch.save(self.d.state_ditc(), fr"{directory}\d.pt")

