import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from tensorboardX import SummaryWriter

from .base_model import VIB


class _NullWriter:
    """Fallback writer when tensorboard logging cannot be initialized."""

    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass


class VIBTrainer(nn.Module):
    """
    Trainer for the Variational Information Bottleneck model.
    Loss = BCEWithLogits + beta * KL(N(mu, sigma^2) || N(0, I))
    """

    def __init__(self, opt, train_loader, beta: float = 8e-02, lr: float = 8e-03, feature_dim: int = 512):
        super(VIBTrainer, self).__init__()
        self.opt = opt
        self.setseed()

        self.best_acc = 0.0  # placeholder for compatibility
        self.total_steps = 0
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.device = torch.device(f'cuda:{opt.gpu_ids[0]}') if opt.gpu_ids and torch.cuda.is_available() else torch.device('cpu')

        # training config
        self.epochs = 100
        self.lr = lr
        self.beta = beta

        # model / optim
        self.model = VIB(feature_dim=feature_dim).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)

        # writer
        try:
            self.train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
        except PermissionError:
            print("[warn] SummaryWriter disabled (permission denied); continuing without TensorBoard logging.")
            self.train_writer = _NullWriter()

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

    @staticmethod
    def kl_divergence(mu, logvar):
        # KL(N(mu, sigma^2) || N(0, I)) averaged over batch
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    def run(self):
        for epoch in range(self.epochs):
            self.model.train()
            for x, y in self.train_loader:
                x = x.to(self.device)
                y = y.to(self.device).float().unsqueeze(1)

                self.optimizer.zero_grad()
                mu, logvar, y_pred = self.model(x, mode='train')

                loss_task = self.criterion(y_pred, y)
                loss_kl = self.kl_divergence(mu, logvar)
                loss = loss_task + self.beta * loss_kl

                loss.backward()
                self.optimizer.step()

                # logs
                self.train_writer.add_scalar('loss', loss.item(), self.total_steps)
                self.train_writer.add_scalar('loss_task', loss_task.item(), self.total_steps)
                self.train_writer.add_scalar('loss_kl', loss_kl.item(), self.total_steps)
                self.total_steps += 1

            # save every epoch
            self.save_networks(f'model_epoch_{epoch}.pth')

        self.train_writer.close()
