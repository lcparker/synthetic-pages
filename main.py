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
from scipy.interpolate import make_interp_spline

def make_spline():
    xs = np.linspace(0,1,10)
    control_points = np.random.rand(len(xs))* 4 + 5
    spline = make_interp_spline(xs, control_points) # cubic b-spline fit to control points
    return spline


# make 2d spline

# position mesh in space

# compute distances to mesh

"""
but is that really what we want? i don't think so

we want to define a 2d sheet and then get the set of distances from it to form a layer.



we should be able to do something like
* generate a plane mesh
* deform the plane mesh to be a spline
* for the grid compute distance to the surface
"""
