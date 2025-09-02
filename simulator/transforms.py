"""
Some utility functions for the image simulator.

@author: Sebastian Lotter <sebastian.g.lotter@fau.de>
"""
import numpy as np
from skimage import transform as tf
from skimage.draw import polygon
import os

PROJECT_PATH = os.path.dirname(os.path.dirname(__file__))


def init_transform(src, tgt):
    pass
    ## Step 4A
    # Estimate the homography transform using the scource and target cordinates provided.
    # Refer: https://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.ProjectiveTransform
    #print(type(src))
    src = np.array(src)
    tgt = np.array(tgt)
    tform = tf.ProjectiveTransform()
    tform.estimate(src, tgt)
    #estimate(src, tgt)
    # return tform
    return tform

def project(coords, tform):

    """
    Apply the homography transformation given by 'tform' on the cordinates given by 'coords'
    and return transformed cordinates.

    Input:
    coords -- list containing the coordinates
    tform  -- homography transform
    """
    pass

    ## Step 4B
    # Obtain the transformed cordinates for the given input cordinates using the transformation.
    # # Refer: https://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.ProjectiveTransform

    transformed_coords = tform.__call__(coords)


    # return transformed_coords

    return transformed_coords
