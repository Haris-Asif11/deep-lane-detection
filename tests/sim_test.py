import os
import sys
from functools import partial

import matplotlib.image as mpimg
import numpy as np
from matplotlib import pyplot as plt


import simulator.colors as colors
import simulator.filters as filters
import simulator.layers as layers
from pipelines import PROJECT_PATH, simulate_road_img
from utils.experiment import read_config
from simulator.transforms import init_transform, project
from utils.file_handling import get_image_list

TEST_FIG_PATH = os.path.join(PROJECT_PATH,'tests','figs')
CONFIG_FOLDER = os.path.join(PROJECT_PATH,'tests','sim_test_configs')

test_register = {
    "constant": [colors.color_w_constant_color, "output_test1.png"],
    "noisy": [colors.color_w_noisy_color, "output_test2.png"],
    "road": [layers.StraightRoadLayer, "output_test3.png"],
    "lanes": [layers.LaneLayer, "output_test4.png"],
    "sky": [layers.SkyLayer, "output_test5.jpg"],
    "tilt": [filters.TiltRoadFilter, "output_test6.jpg"],
    "shift": [filters.ShiftRoadFilter, "output_test7.jpeg"],
    "width": [filters.LaneWidthFilter, "output_test8.JPG"],

}


def test_type(params):
    height = params["simulator"]["height"]
    width = params["simulator"]["width"]
    bin_mask = np.ones((height, width))
    layer_config = params["simulator"]["layer_configs"]
    for layers in layer_config.values():
        if layers["layer_key"] == "BackgroundLayer":
            color_fct = layers["layer_params"]["color_fct"]
            func = partial(test_register[color_fct['key']][0],
                           **color_fct['params'])
            img = func(bin_mask)

            if type(img) != np.ndarray:
                raise TypeError("Output must be a numpy array.")

            if img.shape != (height, width, 3):
                raise TypeError("Output dimensions incorrect.")

            if img.dtype != np.uint8:
                raise TypeError("Output must be 8-bit unsigned int.")
        else:
            continue


def test_color(params, func):
    filename = test_register[func][1]
    test_path = os.path.join(TEST_FIG_PATH, filename)

    test_type(params)

    img1, _ = simulate_road_img(params["simulator"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(img1)
    ax1.title.set_text('Your Result')

    img2 = mpimg.imread(test_path)
    ax2.imshow(img2)
    ax2.title.set_text('Expected Result')
    [i.axis('off') for i in [ax1, ax2]]

    plt.show()


def test_homography(params):
    transform_cordinates = params["simulator"]["layer_configs"]["1"]["layer_params"][
        "transform_coordinates"]
    transform_matrix = init_transform(**transform_cordinates)

    l2_norm = np.linalg.norm(transform_matrix.params, 2)
    l1_norm = np.linalg.norm(transform_matrix.params, 1)

    if not (abs(l2_norm - 2.74738977853) < 1e-6 and abs(
            l1_norm - 3.04545454545) < 1e-6):
        raise ValueError(
            "Homography Test: FAIL; init_transform function implementation incorrect.")

    tgt = project(transform_cordinates['src'], transform_matrix)

    error = np.linalg.norm(tgt - np.array(transform_cordinates['tgt']))
    if error > 1e-6:
        raise ValueError(
            "Homography Test: FAIL; project function implementation incorrect.")

    print('Homography Test: PASS')


def test_road(params, func):
    filename = test_register[func][1]
    test_path = os.path.join(TEST_FIG_PATH, filename)

    img1, _ = simulate_road_img(params["simulator"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(img1)

    ax1.title.set_text('Your Result')

    img2 = mpimg.imread(test_path)
    ax2.imshow(img2)
    ax2.title.set_text('Expected Result')
    [i.axis('off') for i in [ax1, ax2]]

    plt.show()


def test_filter(params, func):
    filename = test_register[func][1]
    test_path = os.path.join(TEST_FIG_PATH, filename)

    img1, _ = simulate_road_img(params["simulator"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(img1)

    ax1.title.set_text('Your Result')

    img2 = mpimg.imread(test_path)
    ax2.imshow(img2)
    ax2.title.set_text('Expected Result')
    [i.axis('off') for i in [ax1, ax2]]

    plt.show()

def test_image_types():
    """Tests if the get_image_list is able to return images
    of types jpeg, jpg and png"""
    print('Testing correct files')
    img_list =  get_image_list("data/input_data/Real_Lane_Predict")
    image_types=[i.rsplit('.')[-1] for i in img_list]
    expected_types = ('jpg','png','JPG','Png','jpeg','PNG','JPEG','Jpg')
    for expected_type in expected_types:
        if expected_type in image_types:
            print(f'PASS: .{expected_type} image listed')
        else:
            print(f'FAIL: .{expected_type} image not listed')
            return 1
    print('\nTesting incorrect files')
    unexpected_types = ('json','h5')
    for unexpected_type in unexpected_types:
        if expected_type in image_types:
            print(f'PASS: .{unexpected_type} image not listed')
        else:
            print(f'FAIL: .{unexpected_type} image listed')
            return 1


if __name__ == '__main__':
    config_path = os.path.join(CONFIG_FOLDER, 'test_filter_tilt.json')

    params = read_config(config_path)

    test_road(params, "sky")
