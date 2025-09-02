"""
This module provides implementations of some basis layer types
that together may form a simulated image.

@author: Sebastian Lotter <sebastian.g.lotter@fau.de>
"""
from functools import partial
import numpy as np
import copy


# User imports
from simulator.colors import COLOR_FCT_REGISTRY
from simulator.draw import draw_polygon
from simulator.transforms import init_transform, project

def merge_layers(layers):
    img = np.zeros((layers[0].fmask.shape) + (3,), dtype=np.uint8)

    for layer in layers:
        rendered_layer = layer.render()
        idx = layer.fmask > 0
        img[idx,:] = rendered_layer[idx,:]

    return img


class Layer():
    """
    A Layer is defined by several attributes that control the dimensions,
    its shape and how the layer is rendered. Each layer has an associated
    shape, represented by a numpy array bitmask 'fmask'
    (from {0,1}^(height x width)), e.g. [[0,1,1,0],[0,1,0,0]],
    which defines the support of the feature it represents.
    Every layer has a height and a width and a color function used to
    render it.

    The Layer class is the abstract base class for implementations of
    different layer types.
    """
    def __init__(self, height, width):
        self.dims = (height, width)

    def render(self):
        return self.color_fct(self.fmask)

    def to_point_cloud(self):
        raise NotImplementedError("Feature not implemented: Serialization of layer type %s" % (type(self).__name__,))


class BackgroundLayer(Layer):
    """
    Represents background of an image. The 'fmask' of a BackgroundLayer
    is an all-one matrix of the dimensions of the image.
    """
    def __init__(self, height, width, color_fct):
        """

        Input:
        height -- height of the image
        width  -- width of the image
        color_fct -- the dictionary containing the color function information obtained
                     from the config file. This dictionary has 2 keys,
                        1. key: key of the coloring function in COLOR_FCT_REGISTRY,
                        2. params: parameter to be passed to the coloring function

        """


        Layer.__init__(self, height, width)


        # Step 1: Fill in the missing code using the instructions in the comments.
        # raise NotImplementedError(
        #     'Complete missing code in __init__ function in BackgroundLayer class [layers.py]')

        # Initialize a class member named fmask with a 2d numpy array with given
        # height and width. All elements in the array must be initialized to one.
        
        self.fmask = np.ones((height, width)) #check order of arguments
        





        # Using color_fct["key"], obtain the color function from the COLOR_FCT_REGISTRY.
        # COLOR_FCT_REGISTRY is a dictionary that is defined at end of the color.py file.
        
        color_function = COLOR_FCT_REGISTRY[str(color_fct["key"])] #no self?
        #raise
        # Initialize a class member named color_fct to bind color function to given parameters. 
        # Use a partial function.
        # refer the partial function example in the problem set document.
        # Also refer : https://www.geeksforgeeks.org/partial-functions-python/
        
        # The first input of the partial function will be the color_function from the previous step.
        # The next inputs of the partial function will be obtained by unpacking the dictionary color_fct['params']
        

        # If you are not familiar with ** notation for unpacking dictionaries,
        # refer: https://realpython.com/python-kwargs-and-args/
        
        self.color_fct = partial(color_function, **color_fct["params"]) #check needed
        

       


class StraightRoadLayer(Layer):
    """
    Models a straight road consisting of the road itself and at least on lane (meaning
    the lane markings). The road is defined in non-perspective coordinates (bird-view)
    and then projected into perspective view via a homography transformation. This
    transformation is defined by two arrays of coordinates [src_coords],[tgt_coords]. For
    details see for example https://en.wikipedia.org/wiki/Homography.

    All coordinates are scaled to [0;1], such that the size of the image can be easily
    changed without need to adjust the layer definition.

    For an example configuration, see config.py.

    The StraightRoadLayer is a composite layer, consisting of a RoadLayer and several
    LaneLayers. Its bitmask (fmask) is the superposition of all bitmasks of those layers.
    """
    def __init__(self, height, width, color_fcts, road_left_cord, road_width, lane_left_cords, lane_widths, transform_coordinates):
        """ 
        Initializes the combined straight road class and lane classes

        Input:
        self            -- it's me
        height          -- layer height
        width           -- layer width
        color_fcts      -- list of functions used to color the road and the lanes
        road_left_cord  -- [[road_lbx, road_lby], [road_ltx, road_lty]] left boundary of the road; b: bottom, t: top, l:left
        road_width      -- width of the road
        lane_left_cords -- [[[lane_1_lbx, lane_1_lby],[lane_1_ltx, lane_1_lty]],...,
                           [lane_n_ltx, lane_n_lty]]] left boundaries of the lanes; b: bottom, t: top, l:left
        lane_widths     -- array of lane widths
        transform_coordinates
                        -- array of coordinates defining the homography projection
        """
        Layer.__init__(self, height, width)


        # Initialize transform matrix.
        transform_matrix = init_transform(**transform_coordinates)


        # Initialize the RoadLayer class object
        self.road_layer = RoadLayer(height, width, road_left_cord, road_width,
                                    transform_matrix, color_fcts[0])


        # Initialize the Lane class object for each lane
        self.lane_layers = []
        for (idx, lane_left_cord) in enumerate(lane_left_cords):
            self.lane_layers.append(LaneLayer(height, width, lane_left_cord,
                                              lane_widths[idx], transform_matrix,
                                              color_fcts[idx+1]))


        # Merge the fmask of the road and the lanes
        self.fmask = merge_masks(map(lambda l: l.fmask, [self.road_layer]+self.lane_layers))



    def render(self):
        # Merge road and lane sublayers and render them (see simulator/draw.py)

        return merge_layers([self.road_layer]+self.lane_layers)


    def to_point_cloud(self):
        """
        Serialize road layer to list of indizes of on-lane pixels
        
        The 2 output dimensions are intended for
        1. y coordinate of pixel
        2. x coordinate of pixel

        Output:
        point_cloud -- [[vertical_idx],[horizontal_idx]]
                       meaning here: [[y],[x]]
        """

        # Initialize output data structure
        point_cloud = np.ndarray((2,0), dtype=np.int32)

        # For each lane layer:
        #   Transform its bitmask to a point cloud
        #   Append the result to the output
        if hasattr(self, 'lane_layers'):
            for (idx, lane_layer) in enumerate(self.lane_layers):
                #Make use of below function fmask_to_point_cloud in order to get variables with the indices of on-lane pixels
                [y, x] = fmask_to_point_cloud(lane_layer.fmask)
                #Conversion to required shape
                point_cloud = np.concatenate((
                        point_cloud,
                        np.vstack((y,x))
                    ), axis=1)

        return point_cloud


class RoadLayer(Layer):
    """
    Layer representing the road itself, without lanes.
    """
    def __init__(self, height, width, road_left_cord,road_width, tform, color_fct):
        """
        Initialize RoadLayer

        Input:
        height      -- layer height
        width       -- layer width
        road_coords -- coordinates defining the road polygon
        road_width  -- width of the road
        tform       -- transform used to project the road to perspective view
        color_fct   -- partial function to color layer
        """

        ## Step 6

        # Using the left coordinates given in the list 'road_left_cord' and road width,
        # generate the right side coordinates.
        
        #print(road_left_cord)
        #print(type(road_left_cord))
        road_right_cord = copy.deepcopy(road_left_cord)
        road_right_cord[0][1] = road_right_cord[0][1] + road_width
        road_right_cord[1][1] = road_right_cord[1][1] + road_width
        #print(road_left_cord)
        #print(road_right_cord)
        road_cord = [road_left_cord[0], road_left_cord[1], road_right_cord[1], road_right_cord[0], road_left_cord[0]]
        #print(road_cord)

        # Store the both the left and right coordinates  in a list named 'road_cord',
        # in the cyclical order as shown below:
        # road_cord = [[lbx,lby], [ltx,lty], [rtx,rty],[rbx,rby], [lbx,lby] ]
        # where b: bottom, t: top, r:right, l:left
        # It is very important to follow this kind of cyclical order because we will
        # use these coordinates to generate a polygon in the image in later part of
        # the code.

        # Project coordinates to perspective view using the project function that you
        # implemented in step 4.
        proj_coords = project(road_cord, tform)

        # Initialize bitmask via draw_polygon function
        # Check the function signature and provide inputs to the function accordingly;
        self.fmask = draw_polygon(height, width, proj_coords)

        # Bind color function to input parameters using the color_fct from COLOR_FCT_REGISTRY.
        # Follow the same procedure as you did in Step 1 for Background layer
        #print(type(color_fct))
        color_function = COLOR_FCT_REGISTRY[str(color_fct["key"])]
        self.color_fct = partial(color_function, **color_fct["params"])
        pass

class LaneLayer(RoadLayer,object):
    """
    Layer representing the lanes/lane markings on a road.
    """
    def __init__(self, height, width, lane_coords,lane_width, tform, color_fct):
        """ 
        Initialize LaneLayer

        Input:
        height      -- layer height
        width       -- layer width
        lane_coords -- coordinates defining the lane polygons
        tform       -- transform used to project the lanes to perspective view
        color_fct   -- partial function to color layer
        """

        ## Step 7
        # The implementation of this class is identical to that of the RoadLayer Class
        # from the previous step.


        # Short answer: Think of a one line solution using the super function.
       



        # Longer answer: Pay attention to the variable names if you decide to copy-paste!
        # Obtain the right side coordinates and complete lane coordinates to form a polygon.
        #print(lane_coords)
        lane_left_cord = copy.deepcopy(lane_coords)
        lane_right_cord = copy.deepcopy(lane_left_cord)
        lane_right_cord[0][1] = lane_right_cord[0][1] + lane_width
        lane_right_cord[1][1] = lane_right_cord[1][1] + lane_width
        lane_coords = [lane_left_cord[0], lane_left_cord[1], lane_right_cord[1], lane_right_cord[0], lane_left_cord[0]]

        # Project coordinates to perspective view using the project function.

        proj_coords = project(lane_coords, tform)

        # Initialize bitmask via draw_polygon function
        self.fmask = draw_polygon(height, width, proj_coords)
        
        

        # Bind color function to input parameters using the color_fct from COLOR_FCT_REGISTRY.
        # Follow the same procedure as you did in Step 1 for Background layer
        
        color_function = COLOR_FCT_REGISTRY[str(color_fct["key"])]
        self.color_fct = partial(color_function, **color_fct["params"])
        
        pass


class SkyLayer(Layer):
    """
    This layer type is intended to add a sky-like polygon to the image. However, there is no
    implementation-wise limitation to sky-like shapes, all sorts of polygons can be added to
    the image via this layer.
    """
    def __init__(self, height, width, color_fct, shape):

        ## Step 8

        # Define the class variable 'fmask' using the draw_polygon function from draw.py
        # Check the function signature and provide inputs to the function accordingly

        
        self.fmask = draw_polygon(height, width, shape)
        
        # Bind color function to input parameters using the color_fct from COLOR_FCT_REGISTRY.
        # Follow the same procedure as you did in Step 1 for Background layer
        
        color_function = COLOR_FCT_REGISTRY[str(color_fct["key"])]
        self.color_fct = partial(color_function, **color_fct["params"])

        pass



# Helper functions
"""
Merge several bitmasks to one
"""
def merge_masks(masks):
    masks = list(masks)
    tgt_mask = np.zeros(masks[0].shape, dtype=np.uint8)

    for (idx, mask) in enumerate(masks, start=1):
        tgt_mask[mask>0] = idx

    return tgt_mask


"""
Transform a bitmask to a point cloud, i.e. list of indices of non-zero elements

Input:
fmask -- bitmask, np.array

Output:
point cloud, tuple of arrays containing the indices of non-zero components of 'fmask'
"""
def fmask_to_point_cloud(fmask):
    return np.nonzero(fmask)


# Public API
# Exporting a registry instead of the functions allows us to change the
# implementation whenever we want.
LAYER_REGISTRY = {
    'BackgroundLayer'         : BackgroundLayer,
    'SkyLayer'                : SkyLayer,
    'StraightRoadLayer'       : StraightRoadLayer
}


