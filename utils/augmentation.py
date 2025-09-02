import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image, ImageOps, ImageFilter

from utils.experiment import read_config

DEFAULT_PARAMS = read_config('./config.json')


################################################################################
# HELPER FUNCTIONS : PIL <-> NUMPY CONVERSION

def convert_to_PIL(image, label):
    image = Image.fromarray(image, mode='RGB')
    # mode L => 8 bit unsigned int (grayscale) images
    label = Image.fromarray(label, mode='L')

    return image, label


def convert_to_numpy(image, label):
    image = np.array(image)
    label = np.array(label, dtype=np.uint8)

    return image, label


################################################################################

class VerticalFlip(object):
    """Mirror(Vertical flip)  both image and label"""

    def __init__(self, probability=DEFAULT_PARAMS['vertical_flip_prob']):
        self.probability = probability

    def __call__(self, input_imgs):
        image, label = input_imgs
        # Limit the number of augmented images in dataset by applying augmentation
        # only if random number generated is less than self.probability
        if random.random() < self.probability:
            image = np.flip(image, 1).copy()  # Flip image
            label = np.flip(label, 1).copy()  # Flip label
        return image, label


class GaussianBlur(object):
    """
    Apply blur on the input image using a gaussian filter. The label is not modified.
    """

    def __init__(self, probability=DEFAULT_PARAMS['blur_prob']):
        self.probability = probability

    def __call__(self, input_imgs):
        image, label = input_imgs
        # Limit the number of augmented images in dataset by applying augmentation
        # only if random number generated is less than self.probability
        if random.random() < self.probability:
            radius = random.uniform(0.25, 3)
            image, label = convert_to_PIL(image, label)
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))

            image, label = convert_to_numpy(image, label)
        return image, label


class Rotate(object):
    """
    Rotate both image and label rotation angle by randomly sampling an angle
     from a uniform distribution with interval [-max_rotation, +max_rotation]
    """

    def __init__(self, probability=DEFAULT_PARAMS['rotate_prob']):
        self.probability = probability

    def __call__(self, input_imgs):
        image, label = input_imgs

        max_rotation = DEFAULT_PARAMS['max_rot']
        # Augmentation is only performed if random number generated is less than
        # self.probability. This limits the number of augmented images in the
        # dataset.
        if random.random() < self.probability:
            ## Step 2a: Remove the pass statement below and complete the missing code.
            #pass
            # Obtain angle using suitable random function
            angle = random.uniform(-1*max_rotation, max_rotation)

            # Convert to PIL
            image, label = convert_to_PIL(image, label)


            # Rotate both image and label using suitable PIL function.
            image = image.rotate(angle)
            label = label.rotate(angle)


            #Convert back to numpy
            image, label = convert_to_numpy(image, label)


        return image, label


class GaussianNoise(object):
    """Add Gaussian noise to image only - Gaussian noise is added pixel- and
    channelwise to image - value added added to each channel of each pixel is
    drawn from a normal distribution of mean DEFAULT_PARAMS['mean'] and
    standard deviation DEFAULT_PARAMS['std'] """

    def __init__(self, probability=DEFAULT_PARAMS['gaussian_prob']):
        self.probability = probability

    def __call__(self, input_imgs):
        image, label = input_imgs

        mean = DEFAULT_PARAMS['mean']
        std = DEFAULT_PARAMS['std']
        # Augmentation is only performed if random number generated is less than
        # self.probability. This limits the number of augmented images in the
        # dataset.
        if random.random() < self.probability:
            # Step 2b: Remove the pass statement below and complete the missing code.
            #pass

            # Convert the image to datatype float
            image = image.astype('float32')

            # Create a noise array using random functions
            noise_size = image.shape
            noise = np.random.normal(mean, std, noise_size)


            # Add the noise array to the image.
            image = image.copy() + noise


            # Ensure that the pixel values are between 0 and 255. If not, clip appropriately.
            image[image < 0] = 0
            image[image > 255] = 255


            # Convert the image back to 8 bit unsigned integer
            image = image.astype('uint8')

        return image, label


class ColRec(object):
    """Add colored rectangles of height params['y_size'] and width
       params['x_size'] to image only
        - number  of rectangles is specified by params['num_rec']
        - position is drawn randomly from a uniform distribution
        - value of each color channel is drawn randomly from a uniform
          distribution"""

    def __init__(self, probability=DEFAULT_PARAMS['colrec_prob']):
        self.probability = probability

    def __call__(self, input_imgs):
        image, label = input_imgs

        n_rectangle = random.randint(1, DEFAULT_PARAMS['num_rec'])
        #n_rectangle = 100
        y_size = DEFAULT_PARAMS['y_size']
        x_size = DEFAULT_PARAMS['x_size']

        # Augmentation is only performed if random number generated is less than
        # self.probability. This limits the number of augmented images in the
        # dataset.
        if random.random() < self.probability:
            for i in range(n_rectangle):
                # Step 2c: Remove the pass statement below and complete the missing code.
                #pass


                # Select a random (y,x) pixel position for top left corner of the rectangle.
                # Since image size is 256 x 256, select point such that
                # x is a random value between 0 and 255-x_size
                # y is a random value between 0 and 255-y_size
                # This is done so that the rectangle does not go outside the image.
                # Note: pixel values must be integers!
                y = random.randint(0, 255-x_size)
                x = random.randint(0, 255-y_size)
                



                # The rectangle will be in the location [y : y+y_size, x : x+x_size]
                # In the input image array, assign a random color to all the pixels inside this
                # rectangle.
                colour_R = random.randint(0, 255)
                colour_G = random.randint(0,255)
                colour_B = random.randint(0,255) #pixel values are integers - 8 bit unsigned integers
                image[y:y+y_size, x:x+x_size, 0] = colour_R
                image[y:y+y_size, x:x+x_size, 1] = colour_G
                image[y:y+y_size, x:x+x_size, 2] = colour_B
                



                # Similarly, for the label array, assign pixel value = 0 for
                # all pixels inside this rectangle
                label[y:y+y_size, x:x+x_size] = 0


        return image, label


class ZoomIn(object):
    """ - from the original image and label crop a squared box
        - height and width of the box is uniformly drawn from
          [255 * DEFAULT_PARAMS['box_size_min'], 255 * DEFAULT_PARAMS['box_size_max'])
        - position of the box is drawn randomly from a uniform distribution
        - cropped is resized to PIL image of size 256x256"""

    def __init__(self, probability=DEFAULT_PARAMS['zoomin_prob']):
        self.probability = probability

    def __call__(self, input_imgs):
        image, label = input_imgs

        # Apply augmentation only if random number generated is less than
        # probability specified in params

        box_size_min = DEFAULT_PARAMS['box_size_min']
        box_size_max = DEFAULT_PARAMS['box_size_max']
        box_size = np.int_(random.uniform(box_size_min, box_size_max) * 255)

        # Augmentation is only performed if random number generated is less than
        # self.probability. This limits the number of augmented images in the
        # dataset.
        if random.random() < self.probability:
            y = random.randint(0, 255-box_size)
            x = random.randint(0, 255-box_size)
            
            # Step 2d: Remove the pass statement below and complete the missing code.
            #pass

            # Similar to previous step, select a random (y,x) pixel position for
            # top left corner of the box.
            # Since image size is 256 x 256, select point such that
            # x is a random value between 0 and 255-box_size
            # y is a random value between 0 and 255-box_size
        
     





            # Now we know the location of the box.
            # Define a list  such that [top_left_y, top_left_x, bottom_left_y, bottom_left_x]

            # box = ...
            top_left_y = y
            top_left_x = x
            bottom_right_y = y+box_size
            bottom_right_x = x+box_size
            box = [top_left_y, top_left_x, bottom_right_y, bottom_right_x] #check needed
               
             
           
            # Crop both the image and label using box list and resize it back to
            # (height = 256,width =256)
            # Hint: Refer PIL library for suitable functions. Remember to convert
            # the result back to numpy array before returning the result.
            image, label = convert_to_PIL(image, label)
            image = image.crop(box)
            label = label.crop(box)
            #image = image.resize((3,256,256))
            image = image.resize((256,256))
            label = label.resize((256,256))
            image, label = convert_to_numpy(image, label)
        

        return image, label


class ZoomOut(object):
    """ A larger black image is created based on the zoom constraints. The image
     is placed in this image at random position. This new image is resize to
     256 x 256."""

    def __init__(self, probability=DEFAULT_PARAMS['zoomout_prob']):
        self.probability = probability

    def __call__(self, input_imgs):
        image, label = input_imgs

        zoom_min = DEFAULT_PARAMS['zoomfac_min']
        zoom_max = DEFAULT_PARAMS['zoomfac_max']
        zoomed_size = np.int_(random.uniform(zoom_min, zoom_max) * 255)
        print("zoom min: ", zoom_min)
        print("zoom max: ", zoom_max)
        print("zoomed size: ", zoomed_size )
        # Augmentation is only performed if random number generated is less than
        # self.probability. This limits the number of augmented images in the
        # dataset.
        if random.random() < self.probability:
            # Step 2e: Remove the pass statement below and complete the missing code.
            #pass




            # Create a large 3D black image (zero valued array) of
            # size zoomed_size x zoomed_size
            # black_image_3d = ...
            black_image_3d = np.zeros((zoomed_size, zoomed_size, 3), dtype = 'uint8') #numpy image is HxWxC



            # For the label, we use a 2D black image of same spatial dimension.
            # black_image_2d =
            black_image_2d = np.zeros((zoomed_size, zoomed_size), dtype = 'uint8')




            # Similar to previous step, select a random (y,x) pixel position for
            # top left corner of the image/label inside the black image.
            # Since inpput image size is 256 x 256, select point such that
            # x is a random value between 0 and zoomed_size-255
            # y is a random value between 0 and zoomed_size-255
            y = random.randint(0, zoomed_size-255)
            x = random.randint(0, zoomed_size-255)




            # Now we know the location of the image. Replace the pixel values
            # at this location in the 3d black image with the input image and
            # 2d black image with label respectively.
            black_image_3d[y:y+256, x:x+256, :] = image.copy()
            #black_image_3d = black_image_3d.astype('uint8')
            #imgplot = plt.imshow(black_image_3d)
            #print(black_image_3d.shape)
            #print(black_image_3d[1][1][1])
            #print(image[30][30][1])
            black_image_2d[y:y+256, x:x+256] = label.copy()


            # Use PIL library to resize the image and label to 256 x 256 as before.
            image, label = convert_to_PIL(black_image_3d, black_image_2d)
            image = image.resize((256,256)).copy()
            label = label.resize((256,256)).copy()
            image, label = convert_to_numpy(image, label)
            
           

        return image, label
        
