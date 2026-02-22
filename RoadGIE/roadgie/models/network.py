from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse


args = argparse.Namespace(num_class=1, dataset="chase")

W = 512
H = 512

hori_translation = torch.zeros([1, 1, W, W])
for i in range(W - 1):
    hori_translation[:, :, i, i + 1] = torch.tensor(1.0)
verti_translation = torch.zeros([1, args.num_class, H, H])
for j in range(H - 1):
    verti_translation[:, :, j, j + 1] = torch.tensor(1.0)
hori_translation = hori_translation.float()
verti_translation = verti_translation.float()


def Bilateral_voting(c_map, hori_translation, verti_translation):
    ####  bilateral voting and convert connectivity-based output into segmentation map. ####

    batch, class_num, channel, row, column = c_map.size()
    vote_out = torch.zeros([batch, class_num, channel, row, column]).cuda()
    # print(vote_out.shape,c_map.shape,hori_translation.shape)
    right = (
        torch.bmm(c_map[:, :, 4].contiguous().view(-1, row, column), hori_translation.view(-1, column, column))).view(
        batch, class_num, row, column)

    left = (torch.bmm(c_map[:, :, 3].contiguous().view(-1, row, column),
                      hori_translation.transpose(3, 2).view(-1, column, column))).view(batch, class_num, row, column)

    left_bottom = (torch.bmm(verti_translation.transpose(3, 2).view(-1, row, row),
                             c_map[:, :, 5].contiguous().view(-1, row, column))).view(batch, class_num, row, column)
    left_bottom = (
        torch.bmm(left_bottom.view(-1, row, column), hori_translation.transpose(3, 2).view(-1, column, column))).view(
        batch, class_num, row, column)
    right_above = (
        torch.bmm(verti_translation.view(-1, row, row), c_map[:, :, 2].contiguous().view(-1, row, column))).view(batch,
                                                                                                                 class_num,
                                                                                                                 row,
                                                                                                                 column)
    right_above = (torch.bmm(right_above.view(-1, row, column), hori_translation.view(-1, column, column))).view(batch,
                                                                                                                 class_num,
                                                                                                                 row,
                                                                                                                 column)
    left_above = (
        torch.bmm(verti_translation.view(-1, row, row), c_map[:, :, 0].contiguous().view(-1, row, column))).view(batch,
                                                                                                                 class_num,
                                                                                                                 row,
                                                                                                                 column)
    left_above = (
        torch.bmm(left_above.view(-1, row, column), hori_translation.transpose(3, 2).view(-1, column, column))).view(
        batch, class_num, row, column)
    bottom = (torch.bmm(verti_translation.transpose(3, 2).view(-1, row, row),
                        c_map[:, :, 6].contiguous().view(-1, row, column))).view(batch, class_num, row, column)
    up = (torch.bmm(verti_translation.view(-1, row, row), c_map[:, :, 1].contiguous().view(-1, row, column))).view(
        batch, class_num, row, column)
    right_bottom = (torch.bmm(verti_translation.transpose(3, 2).view(-1, row, row),
                              c_map[:, :, 7].contiguous().view(-1, row, column))).view(batch, class_num, row, column)
    right_bottom = (torch.bmm(right_bottom.view(-1, row, column), hori_translation.view(-1, column, column))).view(
        batch, class_num, row, column)

    vote_out[:, :, 0] = (c_map[:, :, 0]) * (right_bottom)
    vote_out[:, :, 1] = (c_map[:, :, 1]) * (bottom)
    vote_out[:, :, 2] = (c_map[:, :, 2]) * (left_bottom)
    vote_out[:, :, 3] = (c_map[:, :, 3]) * (right)
    vote_out[:, :, 4] = (c_map[:, :, 4]) * (left)
    vote_out[:, :, 5] = (c_map[:, :, 5]) * (right_above)
    vote_out[:, :, 6] = (c_map[:, :, 6]) * (up)
    vote_out[:, :, 7] = (c_map[:, :, 7]) * (left_above)

    pred_mask, _ = torch.max(vote_out, dim=2)
    ###
    # vote_out = vote_out.view(batch,-1, row, column)
    return pred_mask, vote_out


# -----------------------------------------------------------------------------
# DecoderBlock（方向感知结构增强模块）
# -----------------------------------------------------------------------------
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters, BatchNorm, inp=False):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.bn1 = BatchNorm(in_channels // 4)
        self.relu1 = nn.ReLU()
        self.inp = inp

        self.deconv1 = nn.Conv2d(in_channels // 4, in_channels // 4, (1, 9), padding=(0, 4))
        self.deconv2 = nn.Conv2d(in_channels // 4, in_channels // 4, (9, 1), padding=(4, 0))
        self.deconv3 = nn.Conv2d(in_channels // 4, in_channels // 4, (9, 1), padding=(4, 0))
        self.deconv4 = nn.Conv2d(in_channels // 4, in_channels // 4, (1, 9), padding=(0, 4))

        self.bn2 = BatchNorm(in_channels)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels, n_filters, 1)
        self.bn3 = BatchNorm(n_filters)
        self.relu3 = nn.ReLU()

        self._init_weight()

    def forward(self, x, inp=False):
        x = x.to(dtype=torch.float32)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x1 = self.deconv1(x)
        x2 = self.deconv2(x)
        x3 = self.inv_h_transform(self.deconv3(self.h_transform(x)))
        x4 = self.inv_v_transform(self.deconv4(self.v_transform(x)))

        x = torch.cat((x1, x2, x3, x4), dim=1)

        if self.inp:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def h_transform(self, x):
        shape = x.size()
        x = F.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)
        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = F.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
        x = x[..., 0: shape[-2]]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = F.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = F.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)

# -----------------------------------------------------------------------------
# Wrapper：用于替代 Conv2d，适配 UNet 接口
# -----------------------------------------------------------------------------
class DecoderConvWrapper(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, do_activation=True):
        super().__init__()
        self.block = DecoderBlock(in_channels, out_channels, BatchNorm=nn.BatchNorm2d, inp=False)

    def forward(self, x):
        return self.block(x)


# -----------------------------------------------------------------------------
# Blocks
# -----------------------------------------------------------------------------

class Conv2d(nn.Module):
    """ Perform a 2D convolution

    inputs are [b, c, h, w] where 
        b is the batch size
        c is the number of channels 
        h is the height
        w is the width
    """
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int, 
                 padding: int,
                 do_activation: bool = True, 
                 ):
        super(Conv2d, self).__init__()

        conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        lst = [conv]

        if do_activation:
            lst.append(nn.PReLU()) ## PCX modified
            # lst.append(nn.ReLU())

        self.conv = nn.Sequential(*lst)

    def forward(self, x):
        # x is [B, C, H, W]
        return self.conv(x)
    
# -----------------------------------------------------------------------------
# Network
# -----------------------------------------------------------------------------


class _UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 features: List[int] = [64, 64, 64, 64, 64],
                 conv_kernel_size: int = 3,
                 conv: Optional[nn.Module] = None,
                 conv_kwargs: Dict[str,Any] = {}
                 ):
        """
        UNet (but can switch out the Conv)
        """
        super(_UNet, self).__init__()

        self.in_channels = in_channels

        padding = (conv_kernel_size - 1) // 2

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of U-Net
        for i, feat in enumerate(features):

            if i in [0, 1, 2, 3]:
                self.downs.append(
                    conv(
                        in_channels, feat, kernel_size=conv_kernel_size, padding=padding, **conv_kwargs
                    )
                )
            else:
                self.downs.append(
                    conv(
                        in_channels, feat, kernel_size=conv_kernel_size, padding=padding, **conv_kwargs
                    )
                )
            in_channels = feat

        # Up part of U-Net
        for i, feat in enumerate(reversed(features)):

            if i in [3]:
                self.ups.append(nn.UpsamplingBilinear2d(scale_factor=2))
                self.ups.append(
                    DecoderConvWrapper(
                        # Factor of 2 is for the skip connections
                        feat * 2, feat, kernel_size=conv_kernel_size, padding=padding, **conv_kwargs
                    )
                )
            else:
                self.ups.append(nn.UpsamplingBilinear2d(scale_factor=2))
                self.ups.append(
                    conv(
                        # Factor of 2 is for the skip connections
                        feat * 2, feat, kernel_size=conv_kernel_size, padding=padding, **conv_kwargs
                    )
                )

        self.bottleneck = conv(
            features[-1], features[-1], kernel_size=conv_kernel_size, padding=padding, **conv_kwargs
            )
        self.final_conv = conv(
            features[0], out_channels, kernel_size=1, padding=0, do_activation=False, **conv_kwargs
            )

        self.gradcam_activation = None

    def forward(self, x: torch.Tensor):

        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        # x.retain_grad()
        # self.gradcam_activation = x

        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

            # if idx == 2:
            #     x.retain_grad()
            #     self.gradcam_activation = x

        # x.retain_grad()
        # self.gradcam_activation = x

        logit = self.final_conv(x)

        # hori_translationn = hori_translation.repeat(x.shape[0], 1, 1, 1).cuda()
        # verti_translationn = verti_translation.repeat(x.shape[0], 1, 1, 1).cuda()
        # output_test = F.sigmoid(logit)
        # class_pred = output_test.view([x.shape[0], -1, 8, 512, 512])
        # pred = torch.where(class_pred > 0.5, 1, 0)
        # pred, _ = Bilateral_voting(pred.float(), hori_translationn, verti_translationn)
        #
        # return logit, pred
        return logit
    

class UNet(_UNet):
    """
    Unet with normal conv blocks

    input shape: B x C x H x W
    output shape: B x C x H x W 
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(conv=Conv2d, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)
        