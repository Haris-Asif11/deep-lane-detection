import os

import numpy as np
import torch
from skimage.io import imread
from torch.utils.data import Dataset

from pipelines import DEFAULT_SIM_CONFIG_PATH,simulation_pipeline
from utils.experiment import read_config, save_train_config
from utils.file_handling import read_detection,get_image_list

sc = read_config(DEFAULT_SIM_CONFIG_PATH)['simulator']
HEIGHT = sc['height']
WIDTH = sc['width']
DEFAULT_DATA_PATH = read_config('./config.json')['input_data_path']


#####################################################################################
# HELPER FUNCTIONS


def simulator_run_check(dataset_path, dataset_name, size, seed, params):
    """Checks if theres is dataset in the given path and if there is not
    dataset, then runs simulator pipeline to generate a dataset using the
    other parameters provided. """

    ##  Step 1a:  Remove the pass statement below and fill in the missing code.
    #pass
    # Check if the given path exists. If it does not exist run the simulator
    # using given arguments: params and size.
    # Refer the function simulation_pipeline() in pipelines.py
    
    if os.path.isdir(dataset_path) == False:
        simulation_pipeline(params, size, dataset_name, seed)

def get_label_array(dataset_path, detections_file_name, image_file_path):
    """
    Read the detection file and return the label array. and obtain the pixel
    positions of the lane for the given image_filename. Using the detections,
    generate a label array by setting those pixel positions to 1.
    """

    label = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    detection_file_path = os.path.join(dataset_path, detections_file_name)

    ## Step 1b:  Remove the pass statement below and fill in the missing code.

    # Check if the detection file exists in the path. If it does not exist, return a
    # zero valued label array.
    
    if os.path.isfile(detection_file_path) == False:
        return label 

    # Read the detection file to obtain the pixel positions of the lane for
    # the given image. Refer the function read_detection() in file_handling.py.
    # One of the inputs for read_detection() is the image_filename. But you
    # are given image_file_path, which contains the full path. To
    # extract/split the filename from its full path, you may use the
    # appropriate function from os.path module or any other suitable string
    # module.
    
    filename = os.path.basename(image_file_path)
    #print(filename)
    detections = read_detection(detection_file_path, filename) 
    
    # Using the detections, generate a label array by setting those pixel
    # positions to 1. You have already done this before in the simulator test
    # notebook.
    
    for i in range(detections.shape[1]):
        y = detections[0][i] #rows
        x = detections[1][i] #columns
        label[y][x] = 1

    return label


def convert_numpy_to_tensor(np_img):
    """Convert a numpy image to tensor image"""

    ## Step 1c: Remove the pass statement below and fill in the missing code.
    #pass
    # Check if np_imp is 2d array or 3d array
    # Labels will be 2d arrays (in binary) and input images will be 3d arrays (in RGB)
    if np_img.ndim == 2:
        np_img = np_img.astype('int64') #orig 'int_'
        tensor = torch.from_numpy(np_img)

    # For 2d array:
    # Convert to tensor using appropriate function from Pytorch.
    # Ensure that the datatype is 64 bit integer (long)



    # For 3d array:
    # Numpy array is in H x W x C format, where H:Height, W:Width, C:Channel
    # But Pytorch tensors use C X H X W format.
    # Transpose the numpy array to tensor format.
    
    if np_img.ndim == 3:
        np_img = np_img.astype('float32')
        np_img = np_img/255
        np_img1= np_img.transpose((2,0,1))
        tensor = torch.from_numpy(np_img1)
    

    # Convert to tensor using appropriate function from pytorch.
    # Ensure that the datatype is 32 bit float and
    # the values are normalized from range [0,255] to range [0,1]


    # Return the tensor.
    return tensor


def convert_tensor_to_numpy(tensor_img):
    """Convert the tensor image to a numpy image"""

    # # Step 1d: Remove the line below i.e. np_img = tensor_img,
    # and complete the missing code
    #np_img = tensor_img


    # Numpy conversion: Pytorch has a function to do this. But the given
    # tensor may be in gpu (i.e. tensor_img.device attribute will be cuda if it is
    # in the GPU). Such tensors need to be brought back to cpu before numpy
    # conversions can be done. Refer to the appropriate function in the pytorch
    # documentation.
    
    if tensor_img.is_cuda:
        tensor_img = tensor_img.to("cpu")


    # For 2d array:
    # Return the np_img array without any further action.
    np_img = tensor_img.numpy()
    #if np_img.ndim == 2:
        #return np_img

    # For 3d array:
    # np_img image is now in  C X H X W
    # transpose this array to H x W x C
    
    if np_img.ndim == 3:
        #np_img = np.transpose(np_img).copy()
        np_img = np_img.transpose((1,2,0))
        np_img = np_img * 255
        np_img = np_img.astype('uint8')


    # Ensure that the datatype is 8 bit unsigned int and the values are in range
    # from 0 to 255.
    
    return np_img


#####################################################################################

class ImageDataset(Dataset):
    """
    Class representing Image Dataset This class is used to represent both
    Simulation datasets and true negative real world image datasets.
    Depending on the mode specified by the user,the Dataset return labels for train
    and test modes. User also has the option of choosing smaller sample from a folder
    containing large number of images by setting the size parameter

    Refer: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """

    def __init__(self, dataset_name, cfg_path, size=None, mode='Train',
                 dataset_parent_path=DEFAULT_DATA_PATH
                 , augmentation=None, seed=1):
        """
        Args:
            dataset_name (string): Folder name of the dataset.
            mode (string):
                Nature of operation to be done with the data.
                Possible inputs are Predict, Train, Test
                Default value: Train
            dataset_parent_path (string):
                Path of the folder where the dataset folder is present
                Default: DEFAULT_DATA_PATH from config.json
            size (int):
                Number of images to be generated by the simulator. Ignored otherwise.
            cfg_path (string):
                Config file path of your experiment
            augmentation(Augmentation object):
                Augmentation to be applied on the dataset. Augmentation is
                passed using the object from Compose class (see augmentation.py)
            seed (int):
                Seed used for random functions
                Default:1

        """
        params = read_config(cfg_path)

        self.detections_file_name = params['detections_file_name']

        self.mode = mode
        self.dataset_path = os.path.join(dataset_parent_path, dataset_name)
        self.size = size
        self.augmentation = augmentation

        # Check if the directory exists
        simulator_run_check(self.dataset_path, dataset_name, self.size, seed, params)

        # Get image list and store them to a list
        dataset_folder = os.path.abspath(self.dataset_path)
        self.img_list = get_image_list(dataset_folder)
        self.size = len(self.img_list)

        # Save the dataset information in config file
        if self.mode == 'Train':
            save_train_config(params, self.augmentation, seed, self.dataset_path,
                              self.size)

    def __len__(self):
        """Returns length of the dataset"""
        return self.size

    def __getitem__(self, idx):
        """
        Using self.img_list and the argument value idx, return images and
        labels(if applicable based on Mode) The images and labels are
        returned in torch tensor format

        """
        image_file_path = self.img_list[idx]
        


        # # Step 1e: Delete the below line i.e. img, label = None, None and
        # complete the function implementation as indicated by the TO DO
        # comment blocks.
        #img, label = None, None


        #  TO DO: Read images using filename available in image_file_path
        # Hint: Use imread from scipy or any other appropriate library function.
        # img = ...
        #img = imread(os.path.basename(image_file_path))
        img = imread(image_file_path)
        



        # Some images (usually png) have 4 channels i.e. RGBA where A is the
        # alpha channel. Since the network will use images with 3 channels.
        # Remove the 4th channel in case of RGBA images so that array has a
        # shape height x width x 3.
        img = img[:, :, :3]
        


        # TO DO: Obtain label array from the detection file
        # Hint: Use the helper function from step 1b.
        # label = ...
        
        label = get_label_array(self.dataset_path, self.detections_file_name, image_file_path)


        # Apply augmentation if applicable
        if self.augmentation is not None:
            img, label = self.augmentation((img, label))

        #  TO DO: Convert image and label to tensor and return image,label
        
        img = convert_numpy_to_tensor(img)
        label = convert_numpy_to_tensor(label)
        
        return img,label
        
        










