import torch.nn as nn
class CNN_Model(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=16, kernel_size=3, stride=1, padding=1, bias=False):
        super(CNN_Model, self).__init__()
        layers = []
        channels = 64

        # Initial convolution layer
        layers.append(nn.Conv2d(in_channels, channels, kernel_size, stride, padding, bias=bias))
        layers.append(nn.BatchNorm2d(channels))
        layers.append(nn.ReLU(inplace=True))

        # Adding the remaining layers
        for _ in range(num_layers - 2):  # subtract 2 to account for first and last layer
            layers.append(nn.Conv2d(channels, channels, kernel_size, stride, padding, bias=bias))
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU(inplace=True))

        # Final convolution layer to adjust to the output channels
        layers.append(nn.Conv2d(channels, out_channels, kernel_size, stride, padding, bias=bias))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)