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
from typing import Dict, Tuple
import numpy as np


def pressure_pd_solution(delta_t: float, current_pressure: float, target_pressure: float,
                         data: Dict) -> Tuple[float, Dict]:
    """
    Student solution to maintain LOX pressure to the turbopump at a level of 100.

    Args:
        delta_t: Time step length.
        current_pressure: Current pressure level of the turbopump.
        target_pressure: Target pressure level of the turbopump.
        data: Data passed through out run.  Additional data can be added and existing values modified.
            'ErrorP': Proportional error.  Initialized to 0.0
            'ErrorD': Derivative error.  Initialized to 0.0
    """
    # TODO: implement PD solution here

    tau_p  = 0.5
    tau_d  = 1
    
    diff   = current_pressure - target_pressure
    
    change = -1 * tau_p * diff
    if 'prior_diff' in data.keys():
        change -= tau_d * (diff - data['prior_diff'])

    #print(data)

    data['prior_diff'] = diff

    return change, data


def rocket_pid_solution(delta_t: float, current_velocity: float, optimal_velocity: float,
                        data: Dict) -> Tuple[float, Dict]:
    """
    Student solution for maintaining rocket throttle through the launch based on an optimal flight path

    Args:
        delta_t: Time step length.
        current_velocity: Current velocity of rocket.
        optimal_velocity: Optimal velocity of rocket.
        data: Data passed through out run.  Additional data can be added and existing values modified.
            'ErrorP': Proportional error.  Initialized to 0.0
            'ErrorI': Integral error.  Initialized to 0.0
            'ErrorD': Derivative error.  Initialized to 0.0

    Returns:
        Throttle to set, data dictionary to be passed through run.
    """
    # TODO: implement Rocket PID solution here
    tau_p = 37
    tau_d = .5
    tau_i = .0009

    #print(data['ErrorP'], data['ErrorD'], data['ErrorI'])

    if 'prev_diffs' not in data.keys():
        data['prev_diffs'] = []
        data['iter'] = 0

    diff = current_velocity - optimal_velocity

    change = -1 * tau_p * diff
    if len(data['prev_diffs']) > 0:
        change -= tau_d * ( diff - data['prev_diffs'][-1] ) - tau_i * sum( data['prev_diffs'] )

    data['prev_diffs'].append(diff)
    data['iter'] += 1

    #print(data['iter'], diff, change)

    return change, data


def bipropellant_pid_solution(delta_t: float, current_velocity: float, optimal_velocity: float,
                              data: Dict) -> Tuple[float, float, Dict]:
    """
    Student solution for maintaining fuel and oxidizer throttles through the launch based on an optimal flight path

    Args:
        delta_t: Time step length.
        current_velocity: Current velocity of rocket.
        optimal_velocity: Optimal velocity of rocket.
        data: Data passed through out run.  Additional data can be added and existing values modified.
            'ErrorP': Proportional error.  Initialized to 0.0
            'ErrorI': Integral error.  Initialized to 0.0
            'ErrorD': Derivative error.  Initialized to 0.0

    Returns:
        Fuel Throttle, Oxidizer Throttle to set, data dictionary to be passed through run.
    """
    # TODO: implement Bi-propellant PID solution here

    ## THROTTLE
    tau_p = 23
    tau_d = 0
    tau_i = .0009

    if 'prev_diffs' not in data.keys():
        data['prev_diffs'] = []
        data['iter'] = 0

    diff = current_velocity - optimal_velocity

    change = -1 * tau_p * diff
    if len(data['prev_diffs']) > 0:
        change -= tau_d * ( diff - data['prev_diffs'][-1] ) - tau_i * sum( data['prev_diffs'] )

    data['prev_diffs'].append(diff)
    data['iter'] += 1

    th = change

    ## OXIDIZER
    tau_p = 45
    tau_d = .5
    tau_i = .0009

    if 'prev_diffs' not in data.keys():
        data['prev_diffs'] = []
        data['iter'] = 0

    diff = current_velocity - optimal_velocity

    change = -1 * tau_p * diff
    if len(data['prev_diffs']) > 0:
        change -= tau_d * ( diff - data['prev_diffs'][-1] ) - tau_i * sum( data['prev_diffs'] )

    data['prev_diffs'].append(diff)
    data['iter'] += 1

    ox = change

    return th, ox, data

def who_am_i():
    # Please specify your GT login ID in the whoami variable (ex: jsmith123).
    whoami = 'sshepherd35'
    return whoami
