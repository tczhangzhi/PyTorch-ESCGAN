import torch
import torch.nn as nn


class ESCDiscriminator(nn.Module):
    def __init__(self,
                 in_channels=4,
                 grid_size=(9, 9),
                 hid_channels=64,
                 with_bn=True,
                 num_classes=3):
        super(ESCDiscriminator, self).__init__()

        self.in_channels = in_channels
        self.grid_size = grid_size
        self.hid_channels = hid_channels
        self.num_classes = num_classes
        self.with_bn = with_bn

        conv1 = [
            nn.Conv2d(in_channels,
                      hid_channels,
                      kernel_size=(3, 3),
                      stride=1,
                      padding=1),
            nn.LeakyReLU()
        ]
        if with_bn:
            conv1.insert(1, nn.BatchNorm2d(hid_channels))
        self.conv1 = nn.Sequential(*conv1)

        conv2 = [
            nn.Conv2d(hid_channels,
                      hid_channels * 2,
                      kernel_size=(3, 3),
                      stride=2,
                      padding=1),
            nn.LeakyReLU()
        ]
        if with_bn:
            conv2.insert(1, nn.BatchNorm2d(hid_channels * 2))
        self.conv2 = nn.Sequential(*conv2)

        conv3 = [
            nn.Conv2d(hid_channels * 2,
                      hid_channels * 2,
                      kernel_size=(3, 3),
                      stride=1,
                      padding=1),
            nn.LeakyReLU()
        ]
        if with_bn:
            conv3.insert(1, nn.BatchNorm2d(hid_channels * 2))
        self.conv3 = nn.Sequential(*conv3)

        conv4 = [
            nn.Conv2d(hid_channels * 2,
                      hid_channels * 4,
                      kernel_size=(3, 3),
                      stride=2,
                      padding=1),
            nn.LeakyReLU()
        ]
        if with_bn:
            conv4.insert(1, nn.BatchNorm2d(hid_channels * 4))
        self.conv4 = nn.Sequential(*conv4)

        self.feature_dim = self.get_feature_dim()

        self.adv_proj = nn.Sequential(
            nn.Linear(self.feature_dim, hid_channels * 8), nn.LeakyReLU(),
            nn.Dropout(p=0.3), nn.Linear(hid_channels * 8, hid_channels * 8),
            nn.LeakyReLU(), nn.Dropout(p=0.3), nn.Linear(hid_channels * 8, 1))
        self.aux_proj = nn.Sequential(
            nn.Linear(self.feature_dim, hid_channels * 8), nn.LeakyReLU(),
            nn.Dropout(p=0.3), nn.Linear(hid_channels * 8, hid_channels * 8),
            nn.LeakyReLU(), nn.Dropout(p=0.3),
            nn.Linear(hid_channels * 8, self.num_classes))

        self.init()

    def init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def get_feature_dim(self):
        with torch.no_grad():
            mock_eeg = torch.zeros(1, self.in_channels, *self.grid_size)

            mock_eeg = self.conv1(mock_eeg)
            mock_eeg = self.conv2(mock_eeg)
            mock_eeg = self.conv3(mock_eeg)
            mock_eeg = self.conv4(mock_eeg)

        return mock_eeg.flatten(start_dim=1).shape[-1]

    def forward(self, x, return_feature=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.flatten(start_dim=1)

        if return_feature:
            return self.adv_proj(x), self.aux_proj(x), x
            
        return self.adv_proj(x), self.aux_proj(x)