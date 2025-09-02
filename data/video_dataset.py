import os

import fnmatch
from skimage.transform import resize
import skvideo.io
import numpy as np

from torch.utils.data import Dataset

from data.image_dataset import convert_numpy_to_tensor

from pipelines import DEFAULT_SIM_CONFIG_PATH
from utils.experiment import read_config

params = {
    'input_data_path': './data/input_data'
}
sc = read_config(DEFAULT_SIM_CONFIG_PATH)['simulator']

class VideoWriter:
    def __init__(self, exp):

        filename = "outputvideo.mp4"
        parent_path = exp.params['output_data_path']
        self.full_path = os.path.join(parent_path, filename)
        self.writer = skvideo.io.FFmpegWriter(self.full_path,
            outputdict={'-vcodec': 'libx264'},
        )
        self.writer._proc = None

    def __enter__(self):
        return self

    def write_frame(self,image,overlay_image):
        # Generate video frame by concatenating input and
        # overlay image horizontally.
        frame = (np.concatenate((image, overlay_image),axis=1))
        self.writer.writeFrame(frame)


    def __exit__(self, exc_type, exc_val, exc_tb):
        self.writer.close()


        print('done')




class VideoDataset(Dataset):
    """Video Dataset."""

    def __init__(self, dir=params['input_data_path'],
                 folder_name='Video',
                 video_file_name=None):

        self.length = 0
        folder_path = os.path.join(dir, folder_name)
        search_pattern = video_file_name or '*.mp4'

        for file_name in os.listdir(folder_path):
            if fnmatch.fnmatch(file_name, search_pattern):
                self.file_name = os.path.join(folder_path, file_name)
                self.videogen = skvideo.io.vreader(self.file_name)
                self.vid = list(self.videogen)
                self.length = len(self.vid)
        if self.length == 0:
            raise IOError(
                '{0} Video not found in {1}'.format(video_file_name or '',
                                                    folder_path))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = self.vid[idx]

        img = resize(img, (sc['height'], sc['width'], 3)) * 255

        img = convert_numpy_to_tensor(img)
        label = []
        return img, label