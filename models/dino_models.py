import torch
import torch.nn as nn


CHANNELS = {
    "dinov2_vitl14" : 1024,
}

class DINOModel(nn.Module):
    def __init__(self, name, num_classes=1):
        super(DINOModel, self).__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', name)
        self.fc = nn.Linear(CHANNELS[name], num_classes)
 

    def forward(self, x, return_feature=False):
        features = self.model.encode_image(x) 
        if return_feature:
            return features
        return self.fc(features)