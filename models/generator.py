import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import griddata
from torcheeg.datasets.constants.emotion_recognition.deap import \
    DEAP_CHANNEL_LOCATION_DICT


def get_mask_from_channel_location_dict(channel_location_dict):
    location_array = np.array(list(channel_location_dict.values()))

    loc_x_list = []
    loc_y_list = []
    for _, (loc_y, loc_x) in channel_location_dict.items():
        loc_x_list.append(loc_x)
        loc_y_list.append(loc_y)

    width = max(loc_x_list) + 1
    height = max(loc_y_list) + 1

    grid_y, grid_x = np.mgrid[
        min(location_array[:, 0]):max(location_array[:, 0]):height * 1j,
        min(location_array[:, 1]):max(location_array[:, 1]):width * 1j, ]
    grid_y = grid_y
    grid_x = grid_x

    mock_eeg = np.ones((len(location_array), 1))
    mock_eeg = mock_eeg.transpose(1, 0)
    # timestep eeg channel
    outputs = []

    for timestep_split_y in mock_eeg:
        outputs.append(
            griddata(location_array,
                     timestep_split_y, (grid_x, grid_y),
                     method='cubic',
                     fill_value=0))
    outputs = np.array(outputs)
    return torch.tensor(outputs).float().unsqueeze(0)


class ECALayer(nn.Module):

    def __init__(self, kernel_size=3):
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1,
                              1,
                              kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2,
                              bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1,
                                              -2)).transpose(-1,
                                                             -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class ESCGenerator(nn.Module):

    def __init__(self,
                 hid_channels=128,
                 out_channels=128,
                 grid_size=(9, 9),
                 num_classes=3,
                 with_bn=True,
                 with_eca=False,
                 channel_location_dict=DEAP_CHANNEL_LOCATION_DICT):
        super(ESCGenerator, self).__init__()

        self.hid_channels = hid_channels
        self.out_channels = out_channels
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.channel_location_dict = channel_location_dict
        self.with_bn = with_bn
        self.with_eca = with_eca

        self.register_buffer(
            'mask',
            get_mask_from_channel_location_dict(self.channel_location_dict))

        conv1 = [
            nn.Conv2d(out_channels,
                      hid_channels,
                      kernel_size=(3, 3),
                      stride=1,
                      padding=1),
            nn.LeakyReLU()
        ]
        if with_bn:
            conv1.insert(1, nn.BatchNorm2d(hid_channels))
        if with_eca:
            conv1.append(ECALayer())

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
        if with_eca:
            conv2.append(ECALayer())
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
        if with_eca:
            conv3.append(ECALayer())

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
        if with_eca:
            conv4.append(ECALayer())

        self.conv4 = nn.Sequential(*conv4)

        self.label_embeding = nn.Embedding(num_classes,
                                           hid_channels * 4 * 3 * 3)

        deconv1 = [
            nn.ConvTranspose2d(hid_channels * 4,
                               hid_channels * 2,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               bias=True),
            nn.LeakyReLU()
        ]
        if with_bn:
            deconv1.insert(1, nn.BatchNorm2d(hid_channels * 2))
        self.deconv1 = nn.Sequential(*deconv1)

        deconv2 = [
            nn.ConvTranspose2d(hid_channels * 2,
                               hid_channels * 2,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True),
            nn.LeakyReLU()
        ]
        if with_bn:
            deconv2.insert(1, nn.BatchNorm2d(hid_channels * 2))
        self.deconv2 = nn.Sequential(*deconv2)

        deconv3 = [
            nn.ConvTranspose2d(hid_channels * 2,
                               hid_channels,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               bias=True),
            nn.LeakyReLU()
        ]
        if with_bn:
            deconv3.insert(1, nn.BatchNorm2d(hid_channels))
        self.deconv3 = nn.Sequential(*deconv3)

        deconv4 = [
            nn.ConvTranspose2d(hid_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True),
            nn.LeakyReLU()
        ]
        self.deconv4 = nn.Sequential(*deconv4)

    @property
    def feature_dim(self):
        with torch.no_grad():
            mock_eeg = torch.zeros(1, self.out_channels, *self.grid_size)

            mock_eeg = self.conv1(mock_eeg)
            mock_eeg = self.conv2(mock_eeg)
            mock_eeg = self.conv3(mock_eeg)
            mock_eeg = self.conv4(mock_eeg)

        return mock_eeg.flatten(start_dim=1).shape[-1]

    def forward(self, x, y):

        label_emb = self.label_embeding(y)
        label_emb = label_emb.view(-1, self.hid_channels * 4, 3, 3)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        z = self.conv4(x)

        x = label_emb * z

        # ablate control signal
        # x = self.deconv1(z)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)

        x = x * self.mask
        return x, z
