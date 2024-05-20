"""
The plan:

I want to create a program that generates a volume with N synthetic pages.
It should output the data and the ground truth labels.

The pages should be non-overlapping. 

They should be in 3D.


How do you generate N structured pages? Should be able to do something with splines?

The question is
* how do you generate realistic crumples for pages st they're not all just the exact same?
  and don't intersect?

My thoughts
* generate N straight pages (planes) oriented in a random direction in the scene
* deform them using control points (bezier curves or something like FFD)
* apply random local distortions/variations as like a masked kernel 
(so there's a heatmap over the whole volume that you scale the kernel with,
identity most places except where there are local variations)
"""

import numpy as np


np.zeros((256,256,256))
