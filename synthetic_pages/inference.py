#!/usr/bin/env python3

import torch
from synthetic_pages.nrrd_file import Nrrd
import argparse
from pathlib import Path

def setup_device(force_cpu: bool) -> torch.device:
    if force_cpu:
        return torch.device('cpu')
    
    if not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        return torch.device('cpu')
    
    try:
        device = torch.device('cuda')
        torch.zeros(1).to(device)
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        return device
    except Exception as e:
        print(f"Error initializing CUDA: {e}")
        print("Falling back to CPU")
        return torch.device('cpu')

def load_model(weights_path: str, device: torch.device):
    try:
        from synthetic_pages.stitch import make_network
        network = make_network(weights_path)
        network.to(device)
        network.eval()
        return network
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {weights_path}: {str(e)}")

def infer(input_nrrd: Nrrd, weights_path: str, force_cpu: bool = False) -> Nrrd:
    device = setup_device(force_cpu)
    
    network = load_model(weights_path, device)
    
    try:
        with torch.no_grad():
            volume = torch.from_numpy(input_nrrd.volume).to(device=device).float()[None, None, ...]
            print(f"Shape of volume = {volume.shape}")
            logits = network(volume)
            logits = torch.nn.functional.softmax(logits, dim=1)
            predictions = logits.argmax(dim=1).squeeze(0).cpu().numpy()
            return Nrrd(predictions, input_nrrd.metadata)
    except Exception as e:
        raise RuntimeError(f"Error during inference: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Run inference on NRRD volume')
    parser.add_argument('input_path', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('weights', type=str)
    parser.add_argument('--force-cpu', action='store_true')
    
    args = parser.parse_args()
    
    try:
        input_nrrd = Nrrd.from_file(args.input_path)
        output_nrrd = infer(input_nrrd, args.weights, args.force_cpu)
        output_nrrd.write(args.output_path)
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
