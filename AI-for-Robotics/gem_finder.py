"""
 === Introduction ===

   The assignment is broken up into two parts.
   Part A:
        Create a SLAM implementation to process a series of landmark (gem) measurements and movement updates.
        The movements are defined for you so there are no decisions for you to make, you simply process the movements
        given to you.
        Hint: A planner with an unknown number of motions works well with an online version of SLAM.
    Part B:
        Here you will create the action planner for the robot.  The returned actions will be executed with the goal
        being to navigate to and extract a list of needed gems from the environment.  You will earn points by
        successfully extracting the list of gems from the environment. Extraction can only happen if within the
        minimum distance of 0.15.
        Example Actions (more explanation below):
            'move 3.14 1'
            'extract B 1.5 -0.2'
    Note: All of your estimates should be given relative to your robot's starting location.
    Details:
    - Start position
      - The robot will land at an unknown location on the map, however, you can represent this starting location
        as (0,0), so all future robot location estimates will be relative to this starting location.
    - Measurements
      - Measurements will come from gems located throughout the terrain.
        * The format is {'landmark id':{'distance':0.0, 'bearing':0.0, 'type':'D'}, ...}
      - Only gems that have not been collected and are within the horizon distance will return measurements.
    - Movements
      - Action: 'move 1.570963 1.0'
        * The robot will turn counterclockwise 90 degrees and then move 1.0
      - Movements are stochastic due to, well, it being a robot.
      - If max distance or steering is exceeded, the robot will not move.
    - Needed Gems
      - Provided as list of gem types: ['A', 'B', 'L', ...]
      - Although the gem names aren't real, as a convenience there are 26 total names, each represented by an
        upper case letter of the alphabet (ABC...).
      - Action: 'extract'
        * The robot will attempt to extract a specified gem from the current location..
      - When a gem is extracted from the terrain, it no longer exists in the terrain, and thus won't return a
        measurement.
      - The robot must be with 0.15 distance to successfully extract a gem.
      - There may be gems in the environment which are not required to be extracted.
    The robot will always execute a measurement first, followed by an action.
    The robot will have a time limit of 5 seconds to find and extract all of the needed gems.
"""

from typing import Dict, List
import numpy as np
from matrix import matrix
from robot import truncate_angle
from copy import deepcopy

class SLAM:
    """Create a basic SLAM module.
    """

    def __init__(self):
        """Initialize SLAM components here.
        """
        # TODO
        self.measurement_noise = .5
        self.motion_noise      = .1
        self.steering_noise    = 0
        self.landmark_indices  = {}
        self.x = 0
        self.y = 0
        self.bearing = 0
        self.iter_count = -1
        
        ## initialize Omega matrix and Xi vector.  X and Y will be "interlaced" row-wise.
        self.O  = [[0 for i in range(1 * 2)] for i in range(1 * 2)]
        self.Xi = [ 0 for i in range(1 * 2)]
        self.mu = None
        
        ## initialize first position
        self.O[0][0] = 1 # x
        self.O[1][1] = 1 # y
        self.Xi[0]   = 0.0 # x
        self.Xi[1]   = 0.0 # y
    
    def update_landmark_indices(self, measurements):
        landmark_ids = list(sorted(measurements.keys()))
        for lid in landmark_ids:
            if lid not in list(self.landmark_indices.keys()):
                
                ## add to landmark id index dict
                self.landmark_indices[lid] = len(list(self.landmark_indices.keys()))
                
                ## expand the O matrix
                self.O += [[0.0 for i in range(len(self.O))] for ii in range(2)] ## add rows
                for i in range(len(self.O)): ## add columns
                    self.O[i] = self.O[i] + [0.0,0.0]
                    
                ## expand the Xi matrix
                self.Xi += [0.0,0.0]
        

    # Provided Functions
    def get_coordinates_by_landmark_id(self, landmark_id: str):
        """
        Retrieves the x, y locations for a given landmark

        Args:
            landmark_id: The id for a processed landmark

        Returns:
            the coordinates relative to the robots frame with an initial position of 0.0
        """
        # TODO:
        
        landmark_idx = self.landmark_indices[landmark_id]

        return self.mu.tolist()[(1 + landmark_idx) * 2][0], self.mu.tolist()[(1 + landmark_idx) * 2 + 1][0]

    def process_measurements(self, measurements: Dict):
        """
        Process a new series of measurements.

        Args:
            measurements: Collection of measurements
                in the format {'landmark id':{'distance':0.0, 'bearing':0.0, 'type':'B'}, ...}

        Returns:
            x, y: current belief in location of the robot
        """
        # TODO:
        
        ## SLAM Implementation for movement steps

        self.iter_count += 1

        ## add any new landmarks to the landmark index association dict
        self.update_landmark_indices(measurements)

        ## Get O and Xi matrices
        O  = deepcopy(self.O)
        Xi = deepcopy(self.Xi)

        ## motion and measurement multipliers
        motn_mult = 1 / self.motion_noise
        meas_mult = 1 / self.measurement_noise
        
        ## ----- LANDMARKS -----
        landmark_ids = list(sorted(measurements.keys()))
        
        for lmid in landmark_ids:
        
            ## indexing
            idx = 0
            li  = 1 + self.landmark_indices[lmid] ## index of the landmark in the matrix

            ## x and y are interlaced, so we skip 1 ahead to get the actual index
            li *= 2

            ## process measurement
            # this will get the estimated x,y coordinates of the landmark based on the measurement
            xchange, ychange = _get_xy_estimate(measurements[lmid]['distance'], measurements[lmid]['bearing'])
            dxm = xchange
            dym = ychange
            
            ## update Xi
            # x
            Xi[idx] -= dxm * meas_mult
            Xi[li] += dxm * meas_mult
            # y
            Xi[idx+1] -= dym * meas_mult
            Xi[li+1] += dym * meas_mult

            ## update Omega
            for b in range(2):
                O[idx+b][idx+b] += 1 * meas_mult
                O[li+b][li+b] += 1 * meas_mult
                O[idx+b][li+b] -= 1 * meas_mult
                O[li+b][idx+b] -= 1 * meas_mult
        
        # ## Matrix computations, not online since just for landmarks and not robot position
        O  = np.matrix(O)
        Xi = np.matrix([[x] for x in Xi])
        
        mu = np.dot( np.linalg.inv( O ) , Xi )
        
        ## update state of mu
        self.mu = mu
        
        ## Convert O and Xi back into lists
        O  = [o.tolist()[0] for o in O]
        Xi = [x.tolist()[0][0] for x in Xi]
        
        ## Update state of O and Xi
        self.O  = deepcopy(O)
        self.Xi = deepcopy(Xi)
        
        ## update state of x and y
        self.x = self.mu.tolist()[0][0]
        self.y = self.mu.tolist()[1][0]

        return self.x, self.y
    
    def process_movement(self, steering: float, distance: float):
        """
        Process a new movement.

        Args:
            steering: amount to turn
            distance: distance to move

        Returns:
            x, y: current belief in location of the robot
        """
        # TODO:

        ## SLAM Implementation for movement steps

        ## Get SLAM matrices
        O  = deepcopy(self.O)
        Xi = deepcopy(self.Xi)

        ## motion and measurement multipliers
        motn_mult = 1 / self.motion_noise
        meas_mult = 1 / self.measurement_noise
        
        ## ---- MOTION ----
        ## insert 2 extra rows and columns for x1 in O matrix
        O = O[:2] + [[0.0 for i in range(len(O))] for o in range(2)] + O[2:]
        for j in range(len(O)):
            O[j] = O[j][:2] + [0.0,0.0] + O[j][2:]
        
        ## insert extra positions for x1 in Xi matrix
        Xi = Xi[:2] + [0.0,0.0] + Xi[2:]

        ## indexing
        idx = 0
        
        ## process movement
        self.bearing = truncate_angle(self.bearing + np.random.normal(loc=steering, scale=self.steering_noise))
        xchange, ychange = _get_xy_estimate(distance, self.bearing)
        dx = xchange
        dy = ychange

        ## update Xi
        # x
        Xi[idx] -= dx * motn_mult ## this x
        Xi[idx+2] += dx * motn_mult ## next x
        # y
        Xi[idx+1] -= dy * motn_mult ## this y
        Xi[idx+1+2] += dy * motn_mult ## next y
        
        ## update Omega
        # main diagonal
        for b in range(4):
            O[idx+b][idx+b] += 1 * motn_mult
        ## aux diagonals
        for b in range(2):
            O[idx+b][idx+b+2] -= 1 * motn_mult
            O[idx+b+2][idx+b] -= 1 * motn_mult
            
        ## Online SLAM matrix equations
        A = np.matrix( [o[2:] for o in O[:2]] )
        B = np.matrix( [o[:2] for o in O[:2]] )
        C = np.matrix( [[x] for x in Xi[:2]] )

        Op = np.matrix( [o[2:] for o in O[2:]] )
        Xip = np.matrix( [[x] for x in Xi[2:]] )
        
        O  = np.matrix(O)
        Xi = np.matrix([[x] for x in Xi])

        O  = Op  - np.dot( np.dot( np.transpose(A), np.linalg.inv(B) ), A )
        Xi = Xip - np.dot( np.dot( np.transpose(A), np.linalg.inv(B) ), C )

        ## Get Mu
        mu = np.dot( np.linalg.inv( O ) , Xi )
        
        ## update state of mu
        self.mu = mu
        
        ## Convert O and Xi back into lists
        O  = [o.tolist()[0] for o in O]
        Xi = [x.tolist()[0][0] for x in Xi]
        
        ## Update state of O and Xi
        self.O  = deepcopy(O)
        self.Xi = deepcopy(Xi)

        ## update state of x and y
        self.x = mu.tolist()[0][0]
        self.y = mu.tolist()[1][0]

        return self.x, self.y
        #return 0, 0


class GemExtractionPlanner:
    """
    Create a planner to navigate the robot to reach and extract all the needed gems from an unknown start position.
    """

    def __init__(self, max_distance: float, max_steering: float):
        """
        Initialize your planner here.

        Args:
            max_distance: the max distance the robot can travel in a single move.
            max_steering: the max steering angle the robot can turn in a single move.
        """
        # TODO
        self.max_distance = max_distance
        self.max_steering = max_steering
        self.collected = []
        self.observed_measurements = {}
        self.random_search_iter = 0
        self.random_search_sub_iter = 0
        self.search_grid = [
            #(1,1),(1,1),(-1,1),(-1,1),(-1,-1),(-1,-1),(1,-1),(1,-1),
            (2,2),(2,2),(2,2),(-2,2),(-2,2),(-2,2),(-2,-2),(-2,-2),(-2,-2),(2,-2),(2,-2),(2,-2),
            (3.75,3.75),(3.75,3.75),(3.75,3.75),(-3.75,3.75),(-3.75,3.75),(-3.75,3.75),(-3.75,-3.75),(-3.75,-3.75),(-3.75,-3.75),(3.75,-3.75),(3.75,-3.75),(3.75,-3.75),
            (4.75,4.75),(4.75,4.75),(4.75,4.75),(4.75,4.75),(-4.75,4.75),(-4.75,4.75),(-4.75,4.75),(-4.75,4.75),(-4.75,-4.75),(-4.75,-4.75),(-4.75,-4.75),(-4.75,-4.75),(4.75,-4.75),(4.75,-4.75),(4.75,-4.75),(4.75,-4.75),
            (5.5,5.5),(5.5,5.5),(5.5,5.5),(5.5,5.5),(-5.5,5.5),(-5.5,5.5),(-5.5,5.5),(-5.5,5.5),(-5.5,-5.5),(-5.5,-5.5),(-5.5,-5.5),(-5.5,-5.5),(5.5,-5.5),(5.5,-5.5),(5.5,-5.5),(5.5,-5.5),
            (6,6),(6,6),(6,6),(6,6),(6,6),(-6,6),(-6,6),(-6,6),(-6,6),(-6,6),(-6,-6),(-6,-6),(-6,-6),(-6,-6),(-6,-6),(6,-6),(6,-6),(6,-6),(6,-6),(6,-6)
        ]

        self.mapping = SLAM()

    def next_move(self, needed_gems: List[str], measurements: Dict):
        """Next move based on the current set of measurements.
        Args:
            needed_gems: List of gems remaining which still need to be found and extracted.
            measurements: Collection of measurements from gems in the area.
                                {'landmark id': {
                                                    'distance': 0.0,
                                                    'bearing' : 0.0,
                                                    'type'    :'B'
                                                },
                                ...}
        Return: action: str, points_to_plot: dict [optional]
            action (str): next command to execute on the robot.
                allowed:
                    'move 1.570963 1.0'  - Turn left 90 degrees and move 1.0 distance.
                    'extract B 1.5 -0.2' - [Part B] Attempt to extract a gem of type D from your current location.
                                           This will succeed if the specified gem is within the minimum sample distance.
            points_to_plot (dict): point estimates (x,y) to visualize if using the visualization tool [optional]
                            'self' represents the robot estimated position
                            <landmark_id> represents the estimated position for a certain landmark
                format:
                    {
                        'self': (x, y),
                        '<landmark_id_1>': (x1, y1),
                        '<landmark_id_2>': (x2, y2),
                        ....
                    }
        """
        # TODO

        #print('------------------------------------------------------------------------')
        self.mapping.process_measurements(measurements)

        ## Initialize target list
        targets = []

        ## track estimated x and y for each gem seen, as we may not see them on future iterations after the robot travels out of range to collect other gems
        for lmid in list(measurements.keys()):
            od = measurements[lmid]['distance']
            ob = measurements[lmid]['bearing']
            tp = measurements[lmid]['type']
            ex, ey = _get_xy_estimate(od, ob)
            ex += self.mapping.x
            ey += self.mapping.y
            if lmid not in self.observed_measurements.keys():
                self.observed_measurements[lmid] = [(ex,ey,tp)]
            else:
                self.observed_measurements[lmid].append((ex,ey,tp))

        ## calculate heading and distance to each measured target, using the first observation available of each gem's location
        for lmid in list(self.observed_measurements.keys()):
            m = self.observed_measurements[lmid][0]
            if m[2] in needed_gems and m[2] not in self.collected:
                aim_heading  = np.arctan2(m[1] - self.mapping.y, m[0] - self.mapping.x) ## radian of heading we want to aim, (y,x input)
                aim_distance = np.linalg.norm( np.array([m[0],m[1]]) - np.array([self.mapping.x,self.mapping.y]) )
                targets.append( ( aim_distance, aim_heading, m[2], m[0], m[1] ) )


        ## random search if no targets.  tell it to go to random points and hopefully sense gems on the way
        # search_flag = False
        # if not len(targets):
        #     search_flag = True
        #     directions = ((1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1),(1,0))
        #     dist = self.random_search_iter + 1
        #     sx = directions[self.random_search_sub_iter][0] * dist
        #     sy = directions[self.random_search_sub_iter][1] * dist
        #     aim_heading  = np.arctan2(sy - self.mapping.y, sx - self.mapping.x) ## radian of heading we want to aim, (y,x input)
        #     aim_distance = np.linalg.norm( np.array([sx,sy]) - np.array([self.mapping.x,self.mapping.y]) )
        #     targets.append( ( aim_distance, aim_heading, str((sx,sy)), sx, sy ) )
        #     print(self.random_search_iter, self.random_search_sub_iter, sx, sy)

        # random search if no targets.  tell it to go to random points and hopefully sense gems on the way
        search_flag = False
        if not len(targets):
            search_flag = True
            aim_heading = self.mapping.bearing
            if self.random_search_iter in (2,4):
                aim_heading = self.mapping.bearing + np.pi / 4
            aim_distance = .5
            targets.append( ( aim_distance, aim_heading, 'search', 0, 0 ) )
            self.random_search_iter += 1
            if self.random_search_iter == 4:
                self.random_search_iter = 0

        ## sort targets by distance
        targets = sorted(targets, key=lambda x: x[0]) ## sort by distance

        ## firt element will be target with least distance
        target = targets[0]
        print('target: ', target)

        ## distance to target, or minimum robot travel distance if too far
        dist  = min(self.max_distance, target[0])
        ## target bearing, minus robot's current bearing to get steering instruction
        steer = target[1] - self.mapping.bearing
        steer = truncate_angle(steer)
        
        ## truncate steering to pi / 2 as that is robot's max steering
        if steer < 0:
            steer = max(- np.pi / 2, steer)
        if steer >= 0:
            steer = min(np.pi / 2, steer)

        ## Search iteration
        if search_flag and dist <= .15:
            #print("FOUND IN SEARCH")
            self.random_search_iter += 1

        ## Issue command
        ## If close to target, extract.  Otherwise move
        if dist <= .15 and target[2] in needed_gems:
            self.collected.append( target[2] )
            command = 'extract {0} {1} {2}'.format(target[2], np.round(self.mapping.x, 4), np.round(self.mapping.y, 4))
        else:
            ## move command - if steering is feasible, steer then move, otherwise just steer on this turn
            if target[1] - self.mapping.bearing > np.pi / 2:
                command = 'move {0} {1}'.format(steer, 0)
                self.mapping.process_movement(steer, 0)
            else:
                command = 'move {0} {1}'.format(steer, dist)
                self.mapping.process_movement(steer, dist)
           
        
        ## return command and optional points to plot
        print(command)
        return command, {'robot': (self.mapping.x, self.mapping.y), 'target': (target[3], target[4])}

def _get_xy_estimate(distance, bearing):
    "Gets coordinates of a point given a distance and a bearing"

    y = distance * np.sin(bearing)
    x = distance * np.cos(bearing)

    return (x,y)

def who_am_i():
    # Please specify your GT login ID in the whoami variable (ex: jsmith123).
    whoami = 'sshepherd35'
    return whoami
