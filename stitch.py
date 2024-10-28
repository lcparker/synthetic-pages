import logging
from unet import UNet
from pathlib import Path

logging.basicConfig(level=logging.INFO)

def get_device():
    from torch import device
    from torch.cuda import is_available as cuda_is_available
    device = device('cuda' if cuda_is_available() else 'cpu')

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
        # num_res_units=2
    )
    new_output_conv = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=(3,3,3), stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(1, 1, 1)),
            nn.ConvTranspose3d(64, 16, kernel_size=(3,3,3), stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(1, 1, 1)),
            nn.ConvTranspose3d(16, 16, kernel_size=(3,3,3), stride=(1, 1, 1), padding=(1, 1, 1), output_padding=(0, 0, 0)))
    network.model[2].conv = new_output_conv.to(device)

    if weights_path:
        network.load_state_dict(torch.load(weights_path))
    else:
        from scrolls.weights_init import InitWeights_He # TODO fix import here
        network.apply(InitWeights_He())
    return network.to(device)

# load surrounding 8 cubes of a given coordinate
# need to parse command line, download if not already there (maybe? or just error?)

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
