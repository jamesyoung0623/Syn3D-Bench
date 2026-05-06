import torch
import torch.nn as nn

def cuda(tensor, is_cuda):
    if is_cuda : return tensor.cuda()
    else : return tensor

class Model(nn.Module):
    def __init__(self, feature_dim=512, hidden_dim=64):
        super(Model, self).__init__()
        feature_dim = feature_dim
        hidden_dim = hidden_dim
        output_dim = 1
        
        self.encoder = nn.Linear(feature_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        z = self.encoder(x)
        y_pred = self.classifier(z)
        return z, y_pred
    
class VIB(nn.Module):
    def __init__(self, feature_dim=768, hidden_dim=64):
        super(VIB, self).__init__()
        feature_dim = feature_dim
        hidden_dim = hidden_dim
        output_dim = 1
        
        self.mu_encoder = nn.Linear(feature_dim, hidden_dim)
        self.logvar_encoder = nn.Linear(feature_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def reparameterize(self, mu, logvar, noise_ratio=0.1):
        std = logvar.mul(noise_ratio).exp()
        eps = torch.rand_like(std)

        return std.mul(eps) + mu

    def forward(self, x, mode='val'):
        if mode == 'train':
            mu = self.mu_encoder(x)
            logvar = self.logvar_encoder(x)
            logvar = torch.clamp(logvar, max=100)
            z = self.reparameterize(mu, logvar)
            y_pred = self.classifier(z)
            return mu, logvar, y_pred
        else:
            mu = self.mu_encoder(x)
            y_pred = self.classifier(mu)
            return mu, y_pred
        
def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)
