"""
functions for writing/reading data to/from disk
"""

import glob
import json

import os
import shutil
from datetime import datetime


def read_config(cfg_path):
    with open(cfg_path, 'r') as f:
        params = json.load(f)

    return params


def write_config(params, cfg_path, sort_keys=False):
    with open(cfg_path, 'w') as f:
        json.dump(params, f, indent=2, sort_keys=sort_keys)


def save_train_config(params, augmentation, seed, dataset_path,size):
    params['Network']['total_dataset_number'] += 1
    dataset_key = 'Training_Dataset_' + str(
        params['Network']['total_dataset_number'])
    # If augmentation is applied
    if augmentation:
        augs = {}
        if augmentation.__class__.__name__ =='Compose':
            for i,transform in enumerate(augmentation.transforms):
                name = transform.__class__.__name__
                augs[name+'_prob'] = transform.probability
        else:
            name = augmentation.__class__.__name__
            augs[name + '_prob'] = augmentation.probability


    else:
        augs = None
    dataset_info = {
        'name': os.path.basename(dataset_path),
        'path': dataset_path,
        'size': size,
        'augmentation_applied': augs,
        'seed': seed
    }
    params[dataset_key] = dataset_info
    write_config(params, params['cfg_path'], sort_keys=True)


def save_model_config(train):
    train.model_info['network_name'] = train.net.__class__.__name__
    train.model_info['optimiser'] = train.optimiser.__class__.__name__
    train.model_info['loss_function'] = train.loss_function.__class__.__name__
    train.model_info['optimiser_params'] = train.optimiser.defaults
    train.model_info['lane_to_nolane_weight_ratio'] = (train.loss_function.weight[0]/train.loss_function.weight[1]).item()
    train.model_info['num_epochs'] = len(train.val_f1_history)

    train.model_info['trained_time'] = "{:%B %d, %Y, %H:%M:%S}".format(
        datetime.now())

    cfg_path = train.exp.params['cfg_path']
    write_config(train.exp.params, cfg_path, sort_keys=True)


class Experiment:
    def __init__(self, name, new=False, overwrite=False):
        self.exp_name = name
        if new:
            try:
                self.create_experiment()
            except:
                if overwrite:
                    self.delete_experiment()
                    self.create_experiment()
                else:
                    raise NameError(
                        'Experiment {} already exists. Use another name or set '
                        'overwrite=True'.format(self.exp_name))
        else:
            self.open_experiment()

    def create_experiment(self):
        self.params = read_config('./config.json')
        self.params['experiment_name'] = self.exp_name
        self.create_experiment_folders()
        cfg_file_name = self.params['experiment_name'] + '_config.json'
        cfg_path = os.path.join(self.params['network_output_path'],
                                cfg_file_name)
        self.params['cfg_path'] = cfg_path
        write_config(self.params, cfg_path)

    def create_experiment_folders(self):

        path_keynames = ["network_output_path", "output_data_path",
                         "tf_logs_path"]
        for key in path_keynames:
            self.params[key] = os.path.join(self.params[key],
                                            self.params['experiment_name'])
            os.makedirs(self.params[key])

    def open_experiment(self):
        """Open Existing Experiments"""

        default_params = read_config('./config.json')
        cfg_file_name = self.exp_name + '_config.json'
        cfg_path = os.path.join(default_params['network_output_path'],
                                self.exp_name, cfg_file_name)
        self.params = read_config(cfg_path)

    def delete_experiment(self):
        """Delete Existing Experiment folder"""

        default_params = read_config('./config.json')
        cfg_file_name = self.exp_name + '_config.json'
        cfg_path = os.path.join(default_params['network_output_path'],
                                self.exp_name, cfg_file_name)

        params = read_config(cfg_path)

        path_keynames = ["network_output_path", "output_data_path",
                         "tf_logs_path"]
        for key in path_keynames:
            shutil.rmtree(params[key])

    def del_unused_models(self):
        """Deletes all models (pth file) except the best model from given
        experiment network path """

        default_params = read_config('./config.json')
        cfg_file_name = self.exp_name + '_config.json'
        cfg_path = os.path.join(default_params['network_output_path'],
                                self.exp_name, cfg_file_name)
        if not os.path.exists(cfg_path):
            return 0
        params = read_config(cfg_path)

        if 'best_model_full_path' in params:
            pth_files = glob.glob(
                os.path.join(self.params['network_output_path'], '*.pth'))
            pth_files.remove(self.params['best_model_full_path'])



            [os.remove(pth_file) for pth_file in pth_files]
            print('Deleted {} files from exp: {}'.format(len(pth_files),self.exp_name))

