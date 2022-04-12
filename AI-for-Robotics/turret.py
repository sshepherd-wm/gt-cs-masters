######################################################################
# This file copyright the Georgia Institute of Technology
#
# Permission is given to students to use or modify this file (only)
# to work on their assignments.
#
# You may NOT publish this file or make it available to others not in
# the course.
#
######################################################################
from builtins import object
import numpy as np


class Turret(object):
    """The laser used to defend against invading Meteorites."""

    def __init__(self, init_pos, arena_contains_fcn, max_angle_change,
                 initial_state):
        """Initialize the Turret."""
        self.x_pos = init_pos['x']
        self.y_pos = init_pos['y']

        self.bounds_checker = arena_contains_fcn
        self.current_aim_rad = initial_state['h']
        self.max_angle_change = max_angle_change

        ## adding meteorite observations
        self.meteorite_estimates = {}
        self.meteorite_observations = {}

        ## iteration count
        self.count_iter = 0

        ## defense variables
        self.fire_at_will = 0
        self.aim_history = []
        self.give_ups = []

        ## kalman contant inputs
        self.dt = .1

        # function
        self.F = np.matrix([
            [1,0,self.dt,0],
            [0,1,0,self.dt],
            [0,0,1,0],
            [0,0,0,1]
        ])

        # measurement function projecting from 4 dimensions into 2 because can only observe x and y and not velocity
        self.H = np.matrix([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # measurement uncertainty
        self.R = np.matrix([
            [0.1,0],
            [0,0.1]
        ])

        # identity matrix
        self.I = np.matrix([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    def get_meteorite_observations(self, meteorite_locations):
        """Observe the locations of the Meteorites.

        self - reference to the current object, the Turret.
        meteorite_locations - a list of observations of meteorite locations.

        Each observation in meteorite_locations is a tuple (i, x, y), where i
        is the unique ID for an meteorite, and x, y are the x, y locations
        (with noise) of the current observation of that meteorite at this
        timestep. Only meteorites that are currently 'in-bounds' will appear in
        this list, so be sure to use the meteorite ID, and not the
        position/index within the list to identify specific meteorites. (The
        list may change in size as meteorites move in and out of bounds.)

        In this function, store the meteorite_locations for use in predicting
        the meteorites' future locations, and update the various components of
        the Kalman Filter so that the turret can do that prediction in
        do_kf_estimate_meteorites().

        Returns: None
        """
        # TODO: Update the Turret's estimate of where the meteorites are
        # located at the current timestep

        ## Dictionary of locations keyed by meteorite id
        self.meteorite_observations = {m[0]: (m[1],m[2]) for m in meteorite_locations} # copy new locations into new variable
        #print(self.new_observed_locations)

        pass

    def get_laser_action(self):
        """Compute possible laser changes and return one.

        Return the change in the laser's aim angle, in radians.
        The laser can aim in the range [0.0, pi].
        The maximum angular distance the aim angle can change in a given
        timestep is max_angle_change; larger change angles will be clamped to
        the maximum angle change.
        If the laser is aimed at 0.0 rad, it will point horizontally to the
        right; if it is aimed at pi rad, it will point to the left.
        Note that if the aim angle change returned is 0.0 (no change from
        previous timestep's angle), the laser will be turned ON to fire at
        meteorites.
        If the value returned from this function is nonzero, the laser will
        turn OFF, as the laser can't move and fire at the same time.

        Returns: Float (desired change in laser aim angle, in radians)
        """
        # TODO: Update the change in the laser aim angle, in radians, based
        # on where the meteorites are currently.
        # Note that the laser is ON if the laser angle we return here does not
        # change between timesteps. The laser turns OFF when we change its
        # aim angle.

        if self.fire_at_will:
            self.fire_at_will = 0 ## reset fire flag
            return 0.0 ## return 0 indicating fire

        ## run estimates for the lowest targets
        num_targets = 20
        lowest_targets = [t for t in self._sort_targets() if t[0] not in self.give_ups][0:num_targets]
        for idx in [t[0] for t in lowest_targets]:
            ## Update estimate for this meteorite using the kalman filter
            self._kf(idx, self.meteorite_observations[idx])

        ## potential target coordinates based on velocity of closest current estimated targets
        targets = [
            (idx,
            self.meteorite_estimates[idx][0].tolist()[0][0], # x estimate
            self.meteorite_estimates[idx][0].tolist()[1][0], # y estimate
            ) for idx in [t[0] for t in lowest_targets]
            ]

        idx,x,y = targets[0] ## take the first target

        ## get radian calculation
        radian_aim = np.abs( (np.arctan2(x,y + 1) - np.pi / 2) )

        ## change in radians
        radian_change = radian_aim - self.current_aim_rad

        ## give up on this target for future iterations if we've aimed at it for 8 consecutive times
        if (len(self.aim_history) >= 8) and (sum(self.aim_history[-8:]) / 8 == self.aim_history[-1]):
                self.give_ups.append(idx)
                #print("giving up:" + str(self.give_ups))

        ## check if radian change exceeds max change
        if np.abs(radian_change) > self.max_angle_change:
            radian_change = (radian_change / np.abs(radian_change)) * self.max_angle_change ## if so, limit the change to the max change
            self.fire_at_will = 0 ## not ready to fire next round
        
        ## update aim for new target next round
        if radian_change != self.current_aim_rad:
            self.fire_at_will = 0

        self.fire_at_will = 1 ## ready to fire next round

        #print(final_target, self.fire_at_will, self.current_aim_rad, radian_aim, radian_change, self.max_angle_change)

        ## update aim
        self.current_aim_rad += radian_change

        #print('aim: {0}'.format((x,y)))

        self.aim_history.append(idx)
        return radian_change

    def do_kf_estimate_meteorites(self):
        """Estimate the locations of meteorites one timestep in the future.

        Returns: tuple of tuples containing:
            (meteorite ID, meteorite x-coordinate, meteorite y-coordinate)
        """
        # TODO: Use a KF to estimate the positions of the meteorites one
        # timestep in the future.

        ## Get the top 20 lowest targets
        lowest_targets = self._sort_targets()[0:20]

        ## Iterate through new meteorite observations
        for idx in sorted(self.meteorite_observations.keys()):
        #for idx in [t[0] for t in lowest_targets]:
            ## Update estimate for this meteorite using the kalman filter
            self._kf(idx, self.meteorite_observations[idx])

        ## increment iteration count
        self.count_iter += 1

        ## evaluate
        #self._evaluate_predictions()

        preds = tuple( (idx, self.meteorite_estimates[idx][0].tolist()[0][0], self.meteorite_estimates[idx][0].tolist()[1][0]) for idx in self.meteorite_observations.keys() )
        #print(preds[0])

        return preds
        #return tuple((idx, self.meteorite_observations[idx][0], self.meteorite_observations[idx][1]) for idx in self.meteorite_observations.keys())
        #return tuple((idx, 0, 0) for idx in self.meteorite_observations.keys())



    def _kf(self, idx, observation):
        """Kalman filter implementation for a particular meteorite and observation"""

        # constant inputs
        F = self.F
        H = self.H
        R = self.R
        I = self.I

        # initial uncertainty
        P = np.matrix([
            [1000, 0, 0, 0],
            [0, 1000, 0, 0],
            [0, 0, 1000, 0],
            [0, 0, 0, 1000]
        ])

        # external motion
        u = np.matrix([[0],[0],[0],[0]])

        ## get previous estimate or initialize if new
        if idx not in list(self.meteorite_estimates.keys()):
            x = np.matrix([[0],[0],[0],[0]]) ## initialize everything at zero if starting at beginning
        else:
            x = self.meteorite_estimates[idx][0] ## or grab previous estimate if starting mid-simulation
            P = self.meteorite_estimates[idx][1]

        ## MEASUREMENT
        z = np.matrix([[observation[0]],[observation[1]]]) #,[0],[0] ])
        y = z - (H * x)
        S = H * P * np.transpose(H) + R
        K = P * np.transpose(H) * np.linalg.inv(S) ## Kalman gain
        
        x = x + (K * y)
        P = (I - (K * H)) * P

        ## PREDICTION
        x = (F * x) + u
        P = F * P * np.transpose(F)

        ## update estimate for the meteorite
        #self.meteorite_estimates[idx] = {'state': x, 'uncertainty': P}
        self.meteorite_estimates[idx] = (x,P)

        pass


    def _sort_targets(self):
        """Sorts latest observations by height ascending"""

        sorted_height  = sorted([m[1][1], m[1][0], m[0]] for m in self.meteorite_observations.items())
        # the sorting above had y in the first position of each tuple, x in the second and idx in the third
        targets = [(e[2],e[1],e[0]) for e in sorted_height]

        return targets

    def _evaluate_predictions(self):
        x_dists = []
        y_dists = []

        for idx in self.meteorite_observations:
            x_dists.append(self.meteorite_estimates[idx][0].tolist()[0][0] - self.meteorite_observations[idx][0])
            y_dists.append(self.meteorite_estimates[idx][0].tolist()[1][0] - self.meteorite_observations[idx][1])

        print(
            'iter:',
            str(self.count_iter).zfill(2), np.round(np.mean([np.abs(d) for d in x_dists]), 4),
            np.round(np.mean([np.abs(d) for d in y_dists]), 4),
            len(self.meteorite_observations.keys()),
            len(self.meteorite_estimates.keys())
            )



def who_am_i():
    # Please specify your GT login ID in the whoami variable (ex: jsmith123).
    whoami = 'sshepherd35'
    return whoami
