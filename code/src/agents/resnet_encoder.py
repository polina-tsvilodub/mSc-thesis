import torch
import torch.nn as nn
import torchvision.models as models 

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """
        Initialize pretrained Resnet 50.
        """
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        # remove the last fully connected layer
        modules = list(resnet.children())[:-1] 
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.embed.weight.data.normal_(0., 0.02)
        self.embed.bias.data.fill_(0)
        
    def forward(self, images):
        features = self.resnet(images)
        # reshape features to shape (300, -1) - adapt to first dim
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        
        return features


class ResNetPreprocessor(nn.Module):
    def __init__(self):
        """
        Initialize pretrained Resnet 50.
        """
        super(ResNetPreprocessor, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        # remove the last fully connected layer
        modules = list(resnet.children())[:-1] 
        self.resnet = nn.Sequential(*modules)
        
    def forward(self, images):
        features = self.resnet(images)
        # reshape features to shape (300, -1) - adapt to first dim
        features = features.view(features.size(0), -1)
        
        return features
    
class EncoderMLP(nn.Module):
    def __init__(self, feature_size, embed_size):
        """
        Create the trainable projection layer on top of the resnet features.
        """
        super(EncoderMLP, self).__init__()
        self.embed = nn.Linear(feature_size, embed_size)
        self.embed.weight.data.normal_(0., 0.02)
        self.embed.bias.data.fill_(0)
        
    def forward(self, features):
        embedded_features = self.embed(features)
        
        return embedded_features        