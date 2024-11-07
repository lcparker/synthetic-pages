import logging
from unet import UNet
from pathlib import Path

logging.basicConfig(level=logging.INFO)


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
        from scrolls.weights_init import InitWeights_He

        network.apply(InitWeights_He())
    return network.to(device)


if __name__ == "__main__":

    logging.getLogger(__name__).log(logging.INFO, "Running as script")
    logging.getLogger(__name__).log(logging.INFO, "Making network")

    network = make_network()

    logging.getLogger(__name__).log(logging.INFO, f"Network weights: {network}")

    # import torch
    # random_input = torch.rand((1,512,512,512))
    # out = network(random_input)
    # logging.getLogger(__name__).log(logging.INFO, f"Network infer values: {out}")
    # logging.getLogger(__name__).log(logging.INFO, f"Network infer shape: {out.shape}")
