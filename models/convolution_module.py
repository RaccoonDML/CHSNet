from torch import nn


class ConvBlock(nn.Module):
    """
    Normal Conv Block with BN & ReLU
    """

    def __init__(self, cin, cout, k_size=3, d_rate=1, batch_norm=True, res_link=False):
        super().__init__()
        self.res_link = res_link
        if batch_norm:
            self.body = nn.Sequential(
                nn.Conv2d(cin, cout, k_size, padding=d_rate, dilation=d_rate),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )
        else:
            self.body = nn.Sequential(
                nn.Conv2d(cin, cout, k_size, padding=d_rate, dilation=d_rate),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        if self.res_link:
            return x + self.body(x)
        else:
            return self.body(x)


class OutputNet(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.conv1 = ConvBlock(dim, 256, 3)
        self.conv2 = ConvBlock(256, 128, 3)
        self.conv3 = ConvBlock(128, 64, 3)
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.ReLU(True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x
