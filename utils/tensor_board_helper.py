import torch
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

from utils.visualization import get_overlay_image

class TB():
    def __init__(self, path):
        # Tensor board writer
        self.writer = SummaryWriter(path)

        # Tensor board validation image list
        self.img_list = []
        # Image counter
        self.counter = 0
        # Epoch counter
        self.epoch = 1

    def add_img(self, images, predicted_labels, lane_probability):
        # Only every 5th image will be displayed on tensorboard upto a maximum
        self.counter += 1
        if self.counter % 5 == 0:
            image = images[0].cpu()
            predicted_labels = predicted_labels[0].cpu()
            lane_probability = lane_probability[0].cpu()
            overlay_img = get_overlay_image(image, predicted_labels)

            prob = torch.zeros((3, image.shape[1], image.shape[2]))
            prob[0, :, :] = lane_probability



            self.img_list.extend([image, prob, overlay_img])
            

    def push_image_list(self):
        image_grid = make_grid(self.img_list, nrow=3)
        self.writer.add_image('Input-Segmentation Probability-Overlay',
                              image_grid, self.epoch)

    def push_results(self, mode, loss):
        self.writer.add_scalar(mode + '_Loss', loss, self.epoch)
        if mode == "Test":
            self.push_image_list()

        self.reset(mode)

    def reset(self, mode="Train"):
        self.img_list = []
        self.counter = 0
        if mode == "Test":
            self.img_list = []
            self.epoch += 1
