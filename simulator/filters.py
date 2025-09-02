"""
This module contains filters that can be used to randomize the configuration
of single layers. For example, using the TiltRoadFilter, the tilt of the road
(and the lanes) of a StraightRoadLayer can be varied randomly.

@author: Sebastian Lotter <sebastian.g.lotter@fau.de>
"""
import random
import numpy as np

class ConfigFilter():
    """
    Base class for config-randomizing filters

    Every ConfigFilter has the ability to filter a given layer configuration,
    modifying parameters in a pseudo-random fashion.
    """
    def filter(self, config):
        return config


class TiltRoadFilter(ConfigFilter):
    """
    Tilts a straight road uniformly within [lb;ub].
    """
    def __init__(self, lb, ub):
        # Draw a tilt uniformly from [lb;ub] and save it as attribute
        self.tilt = random.uniform(lb,ub)

        
    def filter(self, config):
        """
        Tilt road and lanes by modifying the coordinates defined in the 'config' using self.tilt
        """

        # Store the road left cordinates and left coordinates of the lanes in
        # variables 'road_left' and 'lane_left' respectively.

        road_left = config['layer_params']['road_left_cord']
        lanes_left = config['layer_params']['lane_left_cords']
        #print(type(lanes_left))
        #print(lanes_left)


        ## Step 10
        # update each coordinate pair in the following manner:
        # x_new  = x_old
        # y_new  = (1-x_old)*tilt +y_old
        
        
        
        for i in range(len(road_left)):
            road_left[i][1] = (1-road_left[i][0])*self.tilt + road_left[i][1] 
        

        # Do this for all coordinates of road_left and lane_left.
        # Attention: lanes_left is a list containing coordinates for each lane.
        # Refer the config file (test_filter_tilt.json) to see how these coordinates are represented.
        for i in range(len(lanes_left)):
            lanes_left[i][0][1] = (1-lanes_left[i][0][0])*self.tilt + lanes_left[i][0][1]
            lanes_left[i][1][1] = (1-lanes_left[i][1][0])*self.tilt + lanes_left[i][1][1]


        # Store the new coordinates of road in config['layer_params']['road_left_cord'] and
        #  new coordinates of lanes in config['layer_params']['lane_left_cords']
        config['layer_params']['road_left_cord'] = road_left
        config['layer_params']['lane_left_cords'] = lanes_left
        



        
        return config


class ShiftRoadFilter(ConfigFilter):
    """
    Shifts a straight road within [lb;ub].
    """
    def __init__(self, lb, ub):
        self.shift = random.uniform(lb,ub)

            
    def filter(self, config):
        ## Step 11

        # Refer to the example in TiltRoadFilter. In this class, the coordinates have
        # to be updated in the following manner:
        # x_new  = x_old
        # y_new  = shift + y_old
        # All other steps are the same.
        
        road_left = config['layer_params']['road_left_cord']
        lanes_left = config['layer_params']['lane_left_cords']
        
        
        for i in range(len(road_left)):
            road_left[i][1] = self.shift + road_left[i][1]
            
        for i in range(len(lanes_left)):
            lanes_left[i][0][1] = self.shift + lanes_left[i][0][1]
            lanes_left[i][1][1] = self.shift + lanes_left[i][1][1]
            
        config['layer_params']['road_left_cord'] = road_left
        config['layer_params']['lane_left_cords'] = lanes_left

        return config




class LaneWidthFilter(ConfigFilter):
    """
    Varies lane width within [width-lb;width+ub].
    """
    def __init__(self, lb, ub):
        ## Step 12 A

        # Randomly select a number between lb and ub from a uniform distribution and store the
        # result in self.delta_width

        self.delta_width = random.uniform(lb, ub)



        pass

    def filter(self, config):
        ## Step 12 B

        # Use drawn delta_width to modify lane widths
        # config['layer_params'][lane_widths] contains a list with
        # the lane_widths of each lane. Update each element of the
        # list by incrementing with self.delta_width.
        # Store the new list into config['layer_params'][lane_widths].
        
        lanes_w = config['layer_params']['lane_widths']
        
        
        for i in range(len(lanes_w)):
            lanes_w[i] = lanes_w[i] + self.delta_width
            
        config['layer_params']['lane_widths'] = lanes_w
       
        return config


## Color filter functions and classes
def get_lane_color():
    # choose between white or yellow
    if random.random()>0.5:
        return [240,240,15]
    else:
        return [240,240,240]

def get_road_color():
    # choose between different shades of gray
    value = int(100 * random.random())
    return [value,value,value]


class ConstantColorFilter(ConfigFilter):
    """
    Picks random color from ([r-dr;r+dr],[g-dg;g+dg],[b-db;b+db]) to vary color
    of constant color function.
    """
    def __init__(self, dr, dg, db):

        self.dr =  dr
        self.dg = dg
        self.db =  db

    def filter(self, config):
        # Modify color defined in the config



        if 'color_fct' in config['layer_params'] and config['layer_params']['color_fct']['key'] == 'constant':

            dist_r = random.randint(-1 * self.dr, self.dr)
            dist_g = random.randint(-1 * self.dg, self.dg)
            dist_b = random.randint(-1 * self.db, self.db)

            config['layer_params']['color_fct']['params']['color'][0] += dist_r
            config['layer_params']['color_fct']['params']['color'][1] += dist_g
            config['layer_params']['color_fct']['params']['color'][2] += dist_b
        elif 'color_fcts' in config['layer_params']:
            color_fcts = config['layer_params']['color_fcts']
            for i in range(len(color_fcts)):
                if color_fcts[i]['key'] == 'constant':

                    dist_r = random.randint(-1 * self.dr, self.dr)
                    dist_g = random.randint(-1 * self.dg, self.dg)
                    dist_b = random.randint(-1 * self.db, self.db)
                    if i:
                        col = get_lane_color()
                    else:
                        col = get_road_color()

                    color_fcts[i]['params']['color'][0] = max(0,min(dist_r+col[0],255))
                    color_fcts[i]['params']['color'][1] = max(0,min(dist_g+col[1],255))
                    color_fcts[i]['params']['color'][2] = max(0,min(dist_b+col[2],255))
            config['layer_params']['color_fcts'] = color_fcts


        return config




class RandomColorMeanFilter(ConfigFilter):
    """
    Picks random color from ([r-dr;r+dr],[g-dg;g+dg],[b-db;b+db]) to vary mean
    of random color function.
    """
    def __init__(self, dr, dg, db):
        self.dr = dr
        self.dg = dg
        self.db = db

    def filter(self, config):
        # Modify color defined in the config
        if 'color_fct' in config['layer_params'] and \
                config['layer_params']['color_fct']['key'] == 'noisy':

            dist_r = random.randint(-1 * self.dr, self.dr)
            dist_g = random.randint(-1 * self.dg, self.dg)
            dist_b = random.randint(-1 * self.db, self.db)

            config['layer_params']['color_fct']['params']['mean'][0] += dist_r
            config['layer_params']['color_fct']['params']['mean'][1] += dist_g
            config['layer_params']['color_fct']['params']['mean'][2] += dist_b
        elif 'color_fcts' in config['layer_params']:
            color_fcts = config['layer_params']['color_fcts']
            for i in range(len(color_fcts)):
                if color_fcts[i]['key'] == 'noisy':
                    dist_r = random.randint(-1 * self.dr, self.dr)
                    dist_g = random.randint(-1 * self.dg, self.dg)
                    dist_b = random.randint(-1 * self.db, self.db)
                    if i:
                        col = get_lane_color()
                    else:
                        col = get_road_color()

                    color_fcts[i]['params']['mean'][0] = max(0,min(dist_r +col[0],255))
                    color_fcts[i]['params']['mean'][1] = max(0,min(dist_g +col[1],255))
                    color_fcts[i]['params']['mean'][2] = max(0,min(dist_b +col[2],255))
            config['layer_params']['color_fcts'] = color_fcts

        return config


# Public API
# Exporting a registry instead of the functions allows us to change the
# implementation whenever we want.
CONFIG_FILTER_REGISTRY = {
    'ShiftRoadFilter'       : ShiftRoadFilter,
    'TiltRoadFilter'        : TiltRoadFilter,
    'LaneWidthFilter'       : LaneWidthFilter,
    'ConstantColorFilter'   : ConstantColorFilter,
    'RandomColorMeanFilter' : RandomColorMeanFilter
}


