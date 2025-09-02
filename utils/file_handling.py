import os
import h5py
import numpy as np

PROJECT_PATH = os.path.dirname(os.path.dirname(__file__))


def get_image_list(folder, path_parent=PROJECT_PATH):
    ## Step 7
    # Get the full path by joining the path_parent parameter  with folder parameter.
    # Ref: https://www.geeksforgeeks.org/python-os-path-join-method/

    # full_path = ...

    # Obtain the list of files in the given full_path and store this list in variable "files"
    # Option 1: https://www.geeksforgeeks.org/python-os-listdir-method/
    # Option 2: https://www.geeksforgeeks.org/glob-filename-pattern-matching/



    # From the list "files", filter out image files, i.e. files ending with jpg, png and jpeg
    # and store them in separate list named "image_files".
    # Note: you also require to handle cases like '.PNG','.Jpg', etc
    # The "image_files" list should contain the full image path and not just the image filename.
    # for example ['/home/mlisp-1/lab/data/sim_image/1.jpg', ...]. You may use the os.path.join function
    # if required.

    full_path = os.path.join(path_parent, folder)
    files = os.listdir(full_path)
    image_files = [os.path.join(full_path, file) for file in files if
                   file.lower().endswith(('jpg', 'png', 'jpeg'))]

    return image_files



def write_detections(detections, detections_file_path):
    """
    Write detections.h5 file that contains detections 
    using the efficient hdf format.
    
    Layout:
    { 'image1.png': np.array((2,N)),
      'image2.png': np.array((2,N))... }
      
    """
    with h5py.File(detections_file_path, 'w') as f:
        for d in detections:
            # Compression factor with 'poor, but fast' lzf compression almost
            # factor 10 for 1 dataset, ~factor 35 for 100 datasets
            f.create_dataset(d, data=detections[d], compression='lzf')


def read_detection(detections_file_path, dataset_key):
    """
    Read detections from detections.h5 file.
    """
    with h5py.File(detections_file_path, 'r') as f:
        ds = f[dataset_key]
        detection = np.zeros(ds.shape, ds.dtype)
        if ds.size != 0:
            ds.read_direct(detection)

    return detection