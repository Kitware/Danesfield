###############################################################################
# Copyright Kitware Inc. and Contributors
# Distributed under the Apache License, 2.0 (apache.org/licenses/LICENSE-2.0)
# See accompanying Copyright.txt and LICENSE files for details
###############################################################################

import torch
from . import unet


class ExtensionUNet(unet.UNet):
    """
    Note input shapes should be a power of 2.

    In this case there will be a ~188 pixel difference between input and output
    dims, so the input should be mirrored with

    Example:
        >>> from clab.live.unet2 import *  # NOQA
        >>> from torch.autograd import Variable
        >>> B, C, W, H = (4, 3, 256, 256)
        >>> B, C, W, H = (4, 3, 572, 572)
        >>> n_classes = 11
        >>> inputs = Variable(torch.rand(B, C, W, H))
        >>> labels = Variable((torch.rand(B, W, H) * n_classes).long())
        >>> self = UNet2(in_channels=C, n_classes=n_classes)
        >>> outputs = self.forward(inputs)
        >>> print('inputs.shape = {!r}'.format(inputs.shape))
        >>> print('outputs.shape = {!r}'.format(outputs.shape))
        >>> print(np.array(inputs.shape) - np.array(outputs.shape))

    Example:
        >>> from clab.torch.models.unet import *  # NOQA
        >>> from torch.autograd import Variable
        >>> B, C, W, H = (4, 5, 480, 360)
        >>> n_classes = 11
        >>> inputs = Variable(torch.rand(B, C, W, H))
        >>> labels = Variable((torch.rand(B, W, H) * n_classes).long())
        >>> self = UNet(in_channels=C, n_classes=n_classes)
        >>> outputs = self.forward(inputs)
        >>> print('inputs.shape = {!r}'.format(inputs.shape))
        >>> print('outputs.shape = {!r}'.format(outputs.shape))
        >>> print(np.array(inputs.shape) - np.array(outputs.shape))
    """

    def __init__(self, **kwargs):
        super(ExtensionUNet, self).__init__(**kwargs)

        self.final
        self.final2 = torch.nn.Conv2d(
            self.final.in_channels, self.n_classes, 1)

    def raw_forward2(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)

        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)
        final2 = self.final2(up1)
        return final, final2

    def forward(self, inputs):
        """
            >>> from clab.torch.models.unet import *  # NOQA
            >>> import torch
            >>> from torch.autograd import Variable
            >>> B, C, W, H = (4, 5, 256, 256)
            >>> n_classes = 11
            >>> inputs = Variable(torch.rand(B, C, W, H))
            >>> labels = Variable((torch.rand(B, W, H) * n_classes).long())
            >>> self = UNet(in_channels=C, n_classes=n_classes)
            >>> outputs = self.forward(inputs)
        """
        # Is there a way to miror so that we have enough input pixels?
        # so we can crop off extras after?
        if isinstance(inputs, (list, tuple)):
            assert len(inputs) == 1, 'unet2 only takes single branch input'
            inputs = inputs[0]

        mirrored = inputs
        mirrored, crop_wh = self.prepad(inputs)

        final1, final2 = self.raw_forward2(mirrored)

        cropped1 = self.postcrop(final1, crop_wh)
        # cropped2 = self.postcrop(final2, crop_wh)

        # return cropped1, cropped2

        return cropped1
