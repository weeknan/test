import torch
import torch.nn as nn
from torch.nn.modules import padding

class C3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1a = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv2a = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.relu = nn.ReLU()
        self.bn = nn.Identity()
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))
        self.softmax = nn.Softmax(dim = 0)
    
    #x:4, 3, 16, 112, 112 | B C L H W
    def forward(self, x):
        print(x.size())
        #x: 4, 3, 16, 112, 112
        x = self.relu(self.bn(self.conv1a(x)))
        print(x.size())
        #x: 4, 64, 16, 112, 112
        x = self.pool1(x)
        print(x.size())
        #x: 4, 64, 16, 56, 56
        x = self.relu(self.bn(self.conv2a(x)))
        print(x.size())
        #x: 4, 128, 16, 56, 56
        x = self.pool2(x)
        print(x.size())
        #x: 4, 128, 8, 28, 28
        x = self.relu(self.bn(self.conv3a(x)))
        x = self.relu(self.bn(self.conv3b(x)))
        print(x.size())
        #x: 4, 256, 8, 28, 28
        x = self.pool3(x)
        print(x.size())
        #x: 4, 256, 4, 14, 14
        x = self.relu(self.bn(self.conv4a(x)))
        x = self.relu(self.bn(self.conv4b(x)))
        print(x.size())
        #x: 4, 512, 4, 14, 14
        x = self.pool4(x)
        print(x.size())
        #x: 4, 512, 2, 7, 7
        x = self.relu(self.bn(self.conv5a(x)))
        x = self.relu(self.bn(self.conv5b(x)))
        print(x.size())
        #x: 4, 512, 2, 7, 7
        x = self.pool5(x)
        #x: 4, 512, 1, 3, 3
        print(x.size())
        x = self.fc6(x.view(-1, 8192))
        x = self.fc7(x)
        print(x.size())
        return self.softmax(x)



def main():
    x = torch.rand(4, 3, 16, 112, 112)
    net = C3D()
    y = net(x)
    print(x.size())
    print(y.size())

if __name__ == '__main__':
    main()


