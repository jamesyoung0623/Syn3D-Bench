import os
import torch
import torch.nn as nn
import torch.optim as optim

import random
import numpy as np
from tensorboardX import SummaryWriter
from utils.loss import hsic_bottleneck_loss
from .base_model import Model


class Trainer(nn.Module):
    def __init__(self, opt, train_loader, feature_dim: int = 512):
        super(Trainer, self).__init__()
        self.opt = opt
        self.setseed()

        self.best_acc = 0.0  # not used without val, but kept for compatibility
        self.total_steps = 0
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.device = torch.device(f'cuda:{opt.gpu_ids[0]}') if opt.gpu_ids and torch.cuda.is_available() else torch.device('cpu')

        # lambdas
        self.lambda_x = int(getattr(opt, 'lambda_x', 500))
        self.lambda_y = int(getattr(opt, 'lambda_y', 300))

        # training config
        self.epochs = 100
        self.lr = 1e-4

        # model / optim
        self.model = Model(feature_dim=feature_dim).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)

        # writer
        log_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.train_writer = SummaryWriter(log_dir)

        # loader
        self.train_loader = train_loader

    def setseed(self):
        torch.manual_seed(99)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(99)
        np.random.seed(99)
        random.seed(99)

    def save_networks(self, save_filename):
        save_path = os.path.join(self.save_dir, save_filename)
        state_dict = {"model": self.model.state_dict()}
        torch.save(state_dict, save_path)

    def run(self):
        for epoch in range(self.epochs):
            self.model.train()
            for x, y in self.train_loader:
                x = x.to(self.device)
                y = y.to(self.device).float().unsqueeze(1)

                self.optimizer.zero_grad()
                z, y_pred = self.model(x)

                loss_task = self.criterion(y_pred, y)
                loss_bottle = hsic_bottleneck_loss(z, x, y, self.lambda_x, self.lambda_y)
                loss = loss_task + loss_bottle

                loss.backward()
                self.optimizer.step()

                # logs
                self.train_writer.add_scalar('loss', loss.item(), self.total_steps)
                self.train_writer.add_scalar('loss_task', loss_task.item(), self.total_steps)
                self.train_writer.add_scalar('loss_bottle', loss_bottle.item(), self.total_steps)
                self.total_steps += 1

            # save every epoch (you can thin this if you like)
            run_tag = getattr(self.opt, "run_tag", "")
            fname = f"model_{run_tag}_epoch_{epoch}.pth" if run_tag else f"model_epoch_{epoch}.pth"
            self.save_networks(fname)

        self.train_writer.close()
