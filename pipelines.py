"""
Simulation pipeline and some helper functions.

"""
import numpy as np
import os
import os.path
import json
from copy import deepcopy
import numpy as np
import random
from skimage.io import imsave

# User imports
from utils.experiment import write_config, read_config
from utils.file_handling import write_detections



# User imports
from simulator.colors import COLOR_FCT_REGISTRY
from simulator.filters import CONFIG_FILTER_REGISTRY
from simulator.layers import LAYER_REGISTRY, merge_layers


DEFAULT_SEED = 42
DEFAULT_CONFIG_FILENAME = 'config.json'
PROJECT_PATH = os.path.dirname(__file__)
DEFAULT_SIM_CONFIG_PATH = os.path.join(PROJECT_PATH, 'sim_config.json')


def init_rng(seed):
    """
    To make results reproducible fix seed of RNG
    """
    random.seed(seed)


    
def simulate_road_img(config):
    """
    Main function of the simulator. Creates an artificial cartoon-like image and a corresponding
    point cloud of on-lane pixels.

    An image consists of several overlayed layers, each one of them having a particular shape
    and beeing able to render itself with a configured color function (see colors.py). The
    configuration of the various layers (e.g. background layer, road layer) and their composition
    is completely up to the user. A example configuration is provided in config.py.

    In order to introduce sufficient variance for the neural net to learn in a meaningful manner,
    there exist 'filters' (see filters.py) that vary the layer configuration in a pseudo-random
    fashion. They tilt, for example, the road or shift it randomly. An example configuration of
    such filters can also be found in config.py.

    Input:
    config     -- configuration of the image layers, see config.py for an example configuration

    Output:
    img        -- the artificial image
    detections -- a listing of on-lane pixels
    """
    layers = []
    detections = np.ndarray((5,0), dtype=np.int32)
    params = deepcopy(config)

    # For each layer configured in the config
    for layer in params['layers']:
        # If it has no 'prob' attribute, or the probablity of appearing in the image
        # is bigger than a drawn random number
        if 'prob' not in layer or layer['prob'] > random.random():
            try:
                # Get the configuration for this layer
                layer_config = params['layer_configs'][layer['config_id']]
            except KeyError:
                raise

            for f in layer.get('filters', []):
                # Apply the filters defined for this layer
                layer_config = CONFIG_FILTER_REGISTRY[f['key']](**f['params']).filter(layer_config)

            try:
                # Initialize layer
                img_layer = LAYER_REGISTRY[layer_config['layer_key']](params['height'], params['width'], **layer_config['layer_params'])

                # Add layer to layer stack
                layers.append(img_layer)

                # If the layer should be serialized, add its features to the list of detections
                if layer.get('serialize'):
                    detections = img_layer.to_point_cloud()
            except KeyError:
                raise

    # Merge image
    img = merge_layers(layers)

    return img, detections


def simulation_pipeline(params, batch_size, dataset_name, seed):
    """
    Main loop of the image simulator. Simulates images and saves them
    to the input data folder together with the configuration and
    the corresponding detections.

    Input:
    params       -- global parameters like paths, filenames etc.
    batch_size   -- number of images to simulate
    dataset_name -- name of the dataset to be produced
    seed         -- seed for the RNG (optional)
    """
    init_rng(seed or DEFAULT_SEED)

    dataset_path = setup_data_dir(params, dataset_name)
    sim_cfg = read_config(DEFAULT_SIM_CONFIG_PATH)
    detections = {}

    # Serialize config to dataset folder
    write_config({
        'global': params,
        'simulator': sim_cfg,
        'others': {
            'seed': seed
        }
    }, os.path.join(dataset_path, DEFAULT_CONFIG_FILENAME))

    # simulator main loop
    for i in range(batch_size):
        # Simulate image
        img, detection = simulate_road_img(sim_cfg['simulator'])

        # Store results
        filename = f"{params['img_file_prefix']}{i}{params['img_file_suffix']}"
        imsave(os.path.join(dataset_path, filename), img)

        detections[str(filename)] = detection

    write_detections(detections, os.path.join(dataset_path, params['detections_file_name']))


def setup_data_dir(params, dataset_name):
    dataset_path = os.path.join(os.path.abspath(params['input_data_path']), dataset_name)

    try:
        os.makedirs(dataset_path)
    except OSError as e:
        raise Exception(("Directory %s already exists. Please choose an unused dataset " +
                "identifier") % (dataset_path,))

    return dataset_path

    


def label_true_negatives(dataset_path, detections_file_name):
    """
    In order to include real world images, that do not contain roads, in the training,
    they are treated as regular dataset without detections/on-road pixels.
    """
    detections = {}
    for filename in os.listdir(dataset_path):
        detection=np.array([[],[],[],[],[]], dtype=np.int32)
        detections[filename] = detection

    write_detections(detections, os.path.join(dataset_path, detections_file_name))
