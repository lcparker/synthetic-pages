import logging
from pathlib import Path
import warnings
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import SkipConnection

logging.basicConfig(level=logging.INFO)

class InputCacheModule(nn.Module):
    def __init__(self):
        super().__init__()
        self._cache = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._cache = x
        return x

    def pop(self) -> Union[torch.Tensor, None]:
        temp = self._cache
        self._cache = None
        return temp


class UNet(nn.Module):
    """
    NB I have no idea where this code is from, I stole it from Tim's notebooks...

    Enhanced version of UNet which has residual units implemented with the ResidualUnit class.
    The residual part uses a convolution to change the input dimensions to match the output dimensions
    if this is necessary but will use nn.Identity if not.
    Refer to: https://link.springer.com/chapter/10.1007/978-3-030-12029-0_40.

    Each layer of the network has a encode and decode path with a skip connection between them. Data in the encode path
    is downsampled using strided convolutions (if `strides` is given values greater than 1) and in the decode path
    upsampled using strided transpose convolutions. These down or up sampling operations occur at the beginning of each
    block rather than afterwards as is typical in UNet implementations.

    To further explain this consider the first example network given below. This network has 3 layers with strides
    of 2 for each of the middle layers (the last layer is the bottom connection which does not down/up sample). Input
    data to this network is immediately reduced in the spatial dimensions by a factor of 2 by the first convolution of
    the residual unit defining the first layer of the encode part. The last layer of the decode part will upsample its
    input (data from the previous layer concatenated with data from the skip connection) in the first convolution. this
    ensures the final output of the network has the same shape as the input.

    Padding values for the convolutions are chosen to ensure output sizes are even divisors/multiples of the input
    sizes if the `strides` value for a layer is a factor of the input sizes. A typical case is to use `strides` values
    of 2 and inputs that are multiples of powers of 2. An input can thus be downsampled evenly however many times its
    dimensions can be divided by 2, so for the example network inputs would have to have dimensions that are multiples
    of 4. In the second example network given below the input to the bottom layer will have shape (1, 64, 15, 15) for
    an input of shape (1, 1, 240, 240) demonstrating the input being reduced in size spatially by 2**4.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        channels: sequence of channels. Top block first. The length of `channels` should be no less than 2.
        strides: sequence of convolution strides. The length of `stride` should equal to `len(channels) - 1`.
        kernel_size: convolution kernel size, the value(s) should be odd. If sequence,
            its length should equal to dimensions. Defaults to 3.
        up_kernel_size: upsampling convolution kernel size, the value(s) should be odd. If sequence,
            its length should equal to dimensions. Defaults to 3.
        num_res_units: number of residual units. Defaults to 0.
        act: activation type and arguments. Defaults to PReLU.
        norm: feature normalization type and arguments. Defaults to instance norm.
        dropout: dropout ratio. Defaults to no dropout.
        bias: whether to have a bias term in convolution blocks. Defaults to True.
            According to `Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_,
            if a conv layer is directly followed by a batch norm layer, bias should be False.
        adn_ordering: a string representing the ordering of activation (A), normalization (N), and dropout (D).
            Defaults to "NDA". See also: :py:class:`monai.networks.blocks.ADN`.

    Examples::

        from monai.networks.nets import UNet

        # 3 layer network with down/upsampling by a factor of 2 at each layer with 2-convolution residual units
        net = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(4, 8, 16),
            strides=(2, 2),
            num_res_units=2
        )

        # 5 layer network with simple convolution/normalization/dropout/activation blocks defining the layers
        net=UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(4, 8, 16, 32, 64),
            strides=(2, 2, 2, 2),
        )

    .. deprecated:: 0.6.0
        ``dimensions`` is deprecated, use ``spatial_dims`` instead.

    Note: The acceptable spatial size of input data depends on the parameters of the network,
        to set appropriate spatial size, please check the tutorial for more details:
        https://github.com/Project-MONAI/tutorials/blob/master/modules/UNet_input_size_constrains.ipynb.
        Typically, when using a stride of 2 in down / up sampling, the output dimensions are either half of the
        input when downsampling, or twice when upsampling. In this case with N numbers of layers in the network,
        the inputs must have spatial dimensions that are all multiples of 2^N.
        Usually, applying `resize`, `pad` or `crop` transforms can help adjust the spatial size of input data.

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
        dimensions: Optional[int] = None,
        output_bottleneck: bool = False,
    ) -> None:

        super().__init__()

        if len(channels) < 2:
            raise ValueError("the length of `channels` should be no less than 2.")
        delta = len(strides) - (len(channels) - 1)
        if delta < 0:
            raise ValueError(
                "the length of `strides` should equal to `len(channels) - 1`."
            )
        if delta > 0:
            warnings.warn(
                f"`len(strides) > len(channels) - 1`, the last {delta} values of strides will not be used."
            )
        if dimensions is not None:
            spatial_dims = dimensions

        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering
        self.bottleneck_cache = None

        def _create_block(
            inc: int,
            outc: int,
            channels: Sequence[int],
            strides: Sequence[int],
            kernels: Sequence[int],
            is_top: bool,
        ) -> nn.Module:
            """
            Builds the UNet structure from the bottom up by recursing down to the bottom block, then creating sequential
            blocks containing the downsample path, a skip connection around the previous block, and the upsample path.

            Args:
                inc: number of input channels.
                outc: number of output channels.
                channels: sequence of channels. Top block first.
                strides: convolution stride.
                is_top: True if this is the top block.
            """
            c = channels[0]
            s = strides[0]
            k = kernels[0]

            subblock: nn.Module

            if len(channels) > 2:
                subblock = _create_block(
                    c, c, channels[1:], strides[1:], kernels[1:], False
                )  # continue recursion down
                upc = c * 2
            else:
                # the next layer is the bottom so stop recursion, create the bottom layer as the sublock for this layer
                subblock = self._get_bottom_layer(c, channels[1])
                upc = c + channels[1]

            down = self._get_down_layer(
                inc, c, s, k, is_top
            )  # create layer in downsampling path
            up = self._get_up_layer(
                upc, outc, s, k, is_top
            )  # create layer in upsampling path

            return self._get_connection_block(down, up, subblock)

        self.model = _create_block(
            in_channels,
            out_channels,
            self.channels,
            self.strides,
            self.kernel_size,
            True,
        )

        if output_bottleneck:
            neck = self.model
            self.bottleneck_cache = InputCacheModule()
            while hasattr(neck[1], "submodule") and isinstance(
                neck[1].submodule, nn.Sequential
            ):
                neck = neck[1].submodule
            neck.append(self.bottleneck_cache)

    def _get_connection_block(
        self, down_path: nn.Module, up_path: nn.Module, subblock: nn.Module
    ) -> nn.Module:
        """
        Returns the block object defining a layer of the UNet structure including the implementation of the skip
        between encoding (down) and decoding (up) sides of the network.

        Args:
            down_path: encoding half of the layer
            up_path: decoding half of the layer
            subblock: block defining the next layer in the network.
        Returns: block for this layer: `nn.Sequential(down_path, SkipConnection(subblock), up_path)`
        """
        return nn.Sequential(down_path, SkipConnection(subblock), up_path)

    def _get_down_layer(
        self,
        in_channels: int,
        out_channels: int,
        strides: int,
        kernels: int,
        is_top: bool,
    ) -> nn.Module:
        """
        Returns the encoding (down) part of a layer of the network. This typically will downsample data at some point
        in its structure. Its output is used as input to the next layer down and is concatenated with output from the
        next layer to form the input for the decode (up) part of the layer.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        mod: nn.Module
        if self.num_res_units > 0:

            mod = ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=kernels,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                adn_ordering=self.adn_ordering,
            )
            return mod
        mod = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=kernels,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering,
        )
        return mod

    def _get_bottom_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        """
        Returns the bottom or bottleneck layer at the bottom of the network linking encode to decode halves.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
        """
        return self._get_down_layer(in_channels, out_channels, 1, 3, False)

    def _get_up_layer(
        self,
        in_channels: int,
        out_channels: int,
        strides: int,
        kernels: int,
        is_top: bool,
    ) -> nn.Module:
        """
        Returns the decoding (up) part of a layer of the network. This typically will upsample data at some point
        in its structure. Its output is used as input to the next layer up.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        conv: Union[Convolution, nn.Sequential]

        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=kernels,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_top and self.num_res_units == 0,
            is_transposed=True,
            adn_ordering=self.adn_ordering,
        )

        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=kernels,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_top,
                adn_ordering=self.adn_ordering,
            )
            conv = nn.Sequential(conv, ru)

        return conv

    def pop_bottleneck(self) -> Union[torch.Tensor, None]:
        if self.bottleneck_cache is not None:
            return self.bottleneck_cache.pop()
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x


def get_device():
    from torch import device
    from torch.cuda import is_available as cuda_is_available

    device = device("cuda" if cuda_is_available() else "cpu")
    logging.getLogger(__name__).info(f"Using device: {device}")
    return device


def make_network(weights_path: str | Path | None = None) -> UNet:
    import torch
    from torch import nn

    device = get_device()

    network = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(64, 128, 320, 512, 768, 1024),
        strides=(4, 2, 2, 2, 2),
        kernel_size=(7, 3, 3, 3, 3),
        output_bottleneck=True,
        dropout=0.1,
    )

    network.to(device)

    new_output_conv = nn.Sequential(
        nn.ConvTranspose3d(
            128,
            64,
            kernel_size=(3, 3, 3),
            stride=(2, 2, 2),
            padding=(1, 1, 1),
            output_padding=(1, 1, 1),
        ),
        nn.ConvTranspose3d(
            64,
            20,
            kernel_size=(3, 3, 3),
            stride=(2, 2, 2),
            padding=(1, 1, 1),
            output_padding=(1, 1, 1),
        ),
        nn.ConvTranspose3d(
            20,
            20,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            output_padding=(0, 0, 0),
        ),
    )

    # NOTE I don't know whether this block does anything since the state dict
    # is loaded immediately after, but just copying tim's code
    nn.init.kaiming_uniform_(new_output_conv[0].weight)
    nn.init.kaiming_uniform_(new_output_conv[1].weight)
    nn.init.kaiming_uniform_(new_output_conv[2].weight)
    nn.init.constant_(new_output_conv[0].bias, 0.0)
    nn.init.constant_(new_output_conv[1].bias, 0.0)
    nn.init.constant_(new_output_conv[2].bias, 0.0)

    network.model[2].conv = new_output_conv.to(device)  # type: ignore

    if weights_path:
        network.load_state_dict(torch.load(weights_path, weights_only=True, map_location=device))
    else:
        raise ValueError(f"No weights path provided.")

    return network.to(device)
