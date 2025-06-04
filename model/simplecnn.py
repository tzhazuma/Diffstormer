from ..utils.lib import *
class CNN2D(nn.Module):
    def __init__(self):
        super(CNN2D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True)
        )
        # torch.nn.init.
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True)
        )

        self.deConv1_1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.deConv1 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True)
        )

        self.deConv2_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.deConv2 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True)
        )

        self.deConv3_1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.deConv3 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True)
        )

        self.deConv4_1 = nn.Conv2d(32, 3, kernel_size=3, padding=1)

        self.deConv4 = nn.ReLU()

    def forward(self, input):
        conv1 = self.conv1(input)

        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        x = self.deConv1_1(conv4)
        x = x + conv3

        deConv1 = self.deConv1(x)

        x = self.deConv2_1(deConv1)
        x += conv2
        deConv2 = self.deConv2(x)

        x = self.deConv3_1(deConv2)
        x += conv1
        deConv3 = self.deConv3(x)

        x = self.deConv4_1(deConv3)
        x += input
        output = self.deConv4(x)

        return output
