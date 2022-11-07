import torch
import torch.nn as nn
from torchsummary import summary
from torchvision.models import efficientnet_v2_s
from torchvision.models import EfficientNet_V2_S_Weights

class BackBone(nn.Module):
    def __init__(self, number_of_classes):
        super().__init__()
        self.number_of_classes = number_of_classes

        #Use EfficientNet V2 with small weights as base
        pretrained = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        #Cut off last layers. Input (-1, 3, 128, 128), Output (-1, 64, 16, 16) 
        self.feature_extractor = nn.Sequential(*list(list(pretrained.children())[:-2][0].children())[:-4])
        for param in self.feature_extractor.parameters():
                param.requires_grad = False
        #Extra conv layers. Output (-1, 64, 8, 8)
        self.extra_layers = nn.Sequential(
                    nn.Conv2d(
                        in_channels=64,
                        out_channels=64,
                        kernel_size=3,
                        stride=1,
                        padding=1
                        ),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Conv2d(
                        in_channels=64,
                        out_channels=64,
                        kernel_size=3,
                        stride=1,
                        padding=1
                        ),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2))
        #Fully connected layers.
        self.fc1 = nn.Sequential(
                nn.Linear(4096, 128), 
                nn.LeakyReLU(),
                nn.Dropout(0.2))
        self.fc2 = nn.Sequential(
                nn.Linear(128, self.number_of_classes))

    def forward(self, x, return_activations=False, return_activations_and_output=False):
        x = self.feature_extractor(x)
        x = self.extra_layers(x)
        x = torch.flatten(x, start_dim = 1)
        a = self.fc1(x)
        a = nn.functional.normalize(a, p=2, dim=1) #L2-normalize features
        if return_activations:
            return a
        x = self.fc2(a)
        if return_activations_and_output:
            return a, x
        return x

    def get_normalized_weights(self):
        # Extract weights of last fully connected layer
        return nn.functional.normalize(self.fc2[0].weight, p=2, dim=1) #L2-normalize weights
