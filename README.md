This repository is home to the code I'm developing as part of working on the [vesuvius challenge](scrollprize.org).

Here, you will find:
* Base classes that wrap useful functionality from other libraries, such as `Mesh` and `Nrrd`
* Algorithms to generate synthetic training data (of labelled instances of scroll cubes) for training instance segmentation neural networks (bezier surface generation in `main.py`)
* Scripts that can be used to download raw data (TIFs) of the scrolls, convert them to cubes, and infer them (this is useful for doing data processing on the cloud)
* `label.py`, an early-days volume annotation tool that I'm hoping to use for post-processing instance-segmentation predictions, stitching them into larger segments for flattening

This code, as written, is solely for my own use. I've made no effort to provide environment setup instructions or links to other software I use, and much of the code is in an unfinished state.
