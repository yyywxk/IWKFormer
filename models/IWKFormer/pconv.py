import numpy as np

import torch
from torch import nn
import torch.nn.functional as F


class PartialConv(nn.Conv2d):

    def __init__(self, *args, **kwargs) -> None:
        kwargs['padding_mode'] = 'reflect'
        super().__init__(*args, bias=True, **kwargs)
        for m in self.modules():
            if hasattr(m, 'bias'):
                m.bias.data.fill_(0)

        self.partial_conv = nn.Conv2d(
            1,
            1,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=False,
            padding_mode='reflect',
        )
        self.partial_conv.weight.data.fill_(1 / np.prod(self.kernel_size))
        self.partial_conv.requires_grad_(False)
        return

    def _forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x (torch.Tensor): (B, C, H, W)
            mask (torch.Tensor): (1, 1, H, W)

        Return:
        '''
        x = mask * x
        with torch.no_grad():
            w = self.partial_conv(mask)
            w.clamp_(min=1e-8)
            w.reciprocal_()
            w *= mask

        x = super().forward(x)
        x *= w
        return x

    def forward(
            self,
            x: torch.Tensor,
            mask: torch.Tensor,
            #padding_mode: str='reflect') -> torch.Tensor:
            padding_mode: str='zero') -> torch.Tensor:

        x = x * mask
        #print(padding_mode)
        if padding_mode == 'zero':
            x = super().forward(x)
            x = x * mask
            return x

        with torch.no_grad():
            b, c, h, w = x.size()
            x_pad = x.view(b * c, 1, h, w)
            x_pad = self.partial_conv(x_pad)
            x_pad = x_pad.view(b, c, h, w)

            weight = self.partial_conv(mask)
            weight.clamp_(min=1e-4)
            weight.reciprocal_()
            weight_after = weight * mask

            void = (weight > np.prod(self.kernel_size) + 1).float()
            weight *= (1 - void) * (1 - mask)

        x = x + x_pad * weight
        x = super().forward(x)
        x = x * weight_after
        return x


# https://github.com/NVIDIA/partialconv
class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False
        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = False
        super(PartialConv2d, self).__init__(*args, **kwargs)
        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0],
                                                 self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])

        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * \
                             self.weight_maskUpdater.shape[3]
        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)
            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)
                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2],
                                          input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in
                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride,
                                            padding=self.padding, dilation=self.dilation, groups=1)

                # for mixed precision training, change 1e-8 to 1e-6
                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)
        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)
        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)
        if self.return_mask:
            return output, self.update_mask
        else:
            return output