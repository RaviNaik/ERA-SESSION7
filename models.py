from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

dropout_val = 0.1

class Model0(nn.Module):
    def __init__(self):
        super(Model0, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1) # 28 >> 28 | 1 >> 3
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # 28 >> 28 | 3 >> 5
        self.pool1 = nn.MaxPool2d(2, 2) # 28 >> 14 | 5 >> 6
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1) # 14 >> 14 | 6 >> 10
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1) # 14 >> 14 | 10 >> 14
        self.pool2 = nn.MaxPool2d(2, 2) # 14 >> 7 | 14 >> 16
        self.conv5 = nn.Conv2d(256, 512, 3) # 7 >> 5 | 16 >> 24
        self.conv6 = nn.Conv2d(512, 1024, 3) # 5 >> 3 | 24 >> 32
        self.conv7 = nn.Conv2d(1024, 10, 3) # 3 >> 1 | 32 >> 40

    def forward(self, x):
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = F.relu(self.conv6(F.relu(self.conv5(x))))
        x = self.conv7(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=0, bias=False), # 28 >> 26 || 1 >> 3
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=0, bias=False), # 26 >> 24 || 3 >> 5
            nn.ReLU()
        )

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 24 >> 12 || 5 >> 6

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=0, bias=False), # 12 >> 10 || 6 >> 10
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False), # 10 >> 10 || 10 >> 14
            nn.ReLU()
        )

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 10 >> 5 || 14 >> 16

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0, bias=False), # 5 >> 3 || 16 >> 24
            nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=10, kernel_size=3, padding=0, bias=False), # 3 >> 1 || 24 >> 32
            # nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=0, bias=False), # 28 >> 26 || 1 >> 3
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=0, bias=False), # 26 >> 24 || 3 >> 5
            nn.ReLU()
        )

        self.transition1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, padding=0, bias=False), # 24 >> 24 || 5 >> 5
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 24 >> 12 || 5 >> 6

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=0, bias=False), # 12 >> 10 || 6 >> 10
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, bias=False), # 10 >> 10 || 10 >> 14
            nn.ReLU()
        )

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 10 >> 5 || 14 >> 16

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=0, bias=False), # 5 >> 3 || 16 >> 24
            nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=3, padding=0, bias=False), # 3 >> 1 || 24 >> 32
            # nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.transition1(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

class Model3(nn.Module):
    def __init__(self):
        super(Model3, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=0, bias=False), # 28 >> 26 || 1 >> 3
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_val)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=0, bias=False), # 26 >> 24 || 3 >> 5
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_val)
        )

        self.transition1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, padding=0, bias=False), # 24 >> 24 || 5 >> 5
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 24 >> 12 || 5 >> 6

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=0, bias=False), # 12 >> 10 || 6 >> 10
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_val)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, bias=False), # 10 >> 10 || 10 >> 14
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_val)
        )

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 10 >> 5 || 14 >> 16

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=0, bias=False), # 5 >> 3 || 16 >> 24
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_val)
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=3, padding=0, bias=False), # 3 >> 1 || 24 >> 32
            # nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.transition1(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

class Model4(nn.Module):
    def __init__(self):
        super(Model4, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=0, bias=False), # 28 >> 26 || 1 >> 3
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_val)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=0, bias=False), # 26 >> 24 || 3 >> 5
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_val)
        )

        self.transition1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=1, padding=0, bias=False), # 24 >> 24 || 5 >> 5
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 24 >> 12 || 5 >> 6

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, padding=0, bias=False), # 12 >> 10 || 6 >> 10
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_val)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, padding=1, bias=False), # 10 >> 10 || 10 >> 14
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_val)
        )

        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 10 >> 5 || 14 >> 16

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=3, padding=0, bias=False), # 10 >> 8 || 14 >> 16
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_val)
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=0, bias=False), # 8 >> 6 || 16 >> 18
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_val)
        )

        self.gap = nn.AvgPool2d(kernel_size=6) # 6 >> 1 || 18 >> 28

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=1, padding=0, bias=False),
            # nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.transition1(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.gap(x)
        x = self.conv7(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

class Model5(nn.Module):
    def __init__(self):
        super(Model5, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, padding=0, bias=False), # 28 >> 26 || 1 >> 3
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_val)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=3, padding=0, bias=False), # 26 >> 24 || 3 >> 5
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_val)
        )

        self.transition1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=1, padding=0, bias=False), # 24 >> 24 || 5 >> 5
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 24 >> 12 || 5 >> 6

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, padding=0, bias=False), # 12 >> 10 || 6 >> 10
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_val)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, padding=1, bias=False), # 10 >> 10 || 10 >> 14
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_val)
        )

        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 10 >> 5 || 14 >> 16

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=3, padding=0, bias=False), # 10 >> 8 || 14 >> 16
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_val)
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=0, bias=False), # 8 >> 6 || 16 >> 18
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_val)
        )

        self.gap = nn.AvgPool2d(kernel_size=6) # 6 >> 1 || 18 >> 28

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=1, padding=0, bias=False),
            # nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.transition1(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.gap(x)
        x = self.conv7(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)







