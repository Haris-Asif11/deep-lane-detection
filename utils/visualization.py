import matplotlib.pyplot as plt
from IPython.display import Markdown as md
import torch
import numpy as np

def get_overlay_image(input_tensor, predicted_labels):
    """Returns overlay image

    input
    input_tensor: 3d tensor in C x H x W format (dtype:  float32)
    predicted_labels: 2d tensor in H x W format (dtype:  long or int64)

    output
    overlay_image : 3d tensor in C x H x W format (dtype: float32)

    """

    # Create overlay_image such that the pixels detected as lane are red in
    # color and the other pixels are same as input.
    # Since the tensors are in the range [0,1],
    # the rgb values of a red color pixel in the tensor would be (1,0,0)

    # You can implement this function faster without using for loops!



    ## Step 5a: Delete the line below and complete the missing code
    ##overlay_image = input_tensor
    ##overlay_image[0, torch.transpose(predicted_labels == 1, 0, 1)] = 1.0
    ##overlay_image[1, torch.transpose(predicted_labels == 1, 0, 1)] = 0.0
    ##overlay_image[2, torch.transpose(predicted_labels == 1, 0, 1)] = 0.0
    overlay_image = input_tensor.clone();
    overlay_image[0, (predicted_labels==1)]= 1.0
    overlay_image[1, (predicted_labels==1)]= 0.0
    overlay_image[2, (predicted_labels==1)] = 0.0
    
    


    return overlay_image


def play_notebook_video(file_path):
    """Plays video in ipython using markdown"""
    return md('<video controls src="{0}"/>'.format(file_path))


def display_output(image, prob_img, overlay_img):
    """
    Displays the output using matplotlib
    """
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 7))
    ax1.set_title("Image")
    ax1.imshow(image)
    ax2.set_title("Probability Map")
    ax2.imshow(prob_img, cmap='gist_gray')
    ax3.set_title("Overlay Image")
    ax3.imshow(overlay_img)
    [i.axis('off') for i in [ax1, ax2, ax3]]
    plt.show()

def display_input(image, label):
    """
        Displays the network training input using matplotlib
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 7))
    ax1.imshow(image)
    ax1.set_title("Image")
    ax2.imshow(label)
    ax2.set_title("Labels")
    [i.axis('off') for i in [ax1, ax2]]

    plt.show()
