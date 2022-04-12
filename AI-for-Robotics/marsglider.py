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

#These import statements give you access to library functions which you may
# (or may not?) want to use.
from math import *
from glider import *

import numpy as np



#This is the function you will have to write for part A. 
#-The argument 'height' is a floating point number representing 
# the number of meters your glider is above the average surface based upon 
# atmospheric pressure. (You can think of this as hight above 'sea level'
# except that Mars does not have seas.) Note that this sensor may be
# slightly nosiy. 
# This number will go down over time as your glider slowly descends.
#
#-The argument 'radar' is a floating point number representing the
# number of meters your glider is above the specific point directly below
# your glider based off of a downward facing radar distance sensor. Note that
# this sensor has random Gaussian noise which is different for each read.

#-The argument 'mapFunc' is a function that takes two parameters (x,y)
# and returns the elevation above "sea level" for that location on the map
# of the area your glider is flying above.  Note that although this function
# accepts floating point numbers, the resolution of your map is 1 meter, so
# that passing in integer locations is reasonable.
#
#
#-The argument OTHER is initially None, but if you return an OTHER from
# this function call, it will be passed back to you the next time it is
# called, so that you can use it to keep track of important information
# over time.
#

## GLOBALS
num_particles = 10000

init_x_noise = 150
init_y_noise = init_x_noise
init_r_noise = np.pi / 4

measurement_noise = 5
altitude_noise    = 5
turning_noise     = np.pi / 16

#OTHER_global = None

def estimate_next_pos(height, radar, mapFunc, OTHER=None):
  """Estimate the next (x,y) position of the glider."""

  ## Initialize partciles and persistent values to pass between iterations in OTHER
  if not OTHER:
    #print('---------------------------------------------------------------')
    OTHER = {'particles': [], 'iter': 0}
    OTHER['particles'] = init_particles(num_particles, radar, height, mapFunc)
    OTHER['weights'] = []
    OTHER['zero_points'] = init_zero_points()
    OTHER['found_flag'] = False
    OTHER['steering_angle'] = 0.0

  # get variables from passed data
  ps = OTHER['particles']
  iter_n = OTHER['iter']
  
  # movement and measurement
  ws = []
  for i in range(len(ps)):
    ps[i].z = height ## update particle altitude with new barometric measurement
    ps[i] = ps[i].glide(rudder = OTHER['steering_angle'])

    ## calculate similarity of particle height (from ground) to radar's height measurement, given Gaussian measurement noise
    ## this is the 'weight' we give to each particle
    mp = Gaussian( radar, measurement_noise, ps[i].sense() )
    ws.append( mp )

    ## Add fuzz to paricle movement to allow a "fan-out" probe search like behavior
    ## lower if earlier in simulation, higher if later
    if iter_n <= 30:
      ps[i].x += np.random.normal(0, scale= 1 )
      ps[i].y += np.random.normal(0, scale= 1 )

    if iter_n > 30:
      ps[i].turning_noise = np.pi / 32 ## reduce turning noise as hopefully we're closer and don't want to deviate as much
      ps[i].x += np.random.normal(0, scale= mp * 20 )
      ps[i].y += np.random.normal(0, scale= mp * 20 )

    ## reduce noise even more if we think we've found the glider
    if OTHER['found_flag']:
      ps[i].turning_noise = np.pi / 64
      ## now particles with lower weights will have more noise and particles with higher weights will have less
      ps[i].x += np.random.normal(0, scale= ( (.1 - mp) * 10 ) )
      ps[i].y += np.random.normal(0, scale= ( (.1 - mp) * 10 ) )

  ## zip together the particles and their measurement weights
  ps_r = list(zip(ps,ws))

  ## average old and new weight for each particle to give a bonus to particles with consistently high weights
  if 0 < iter_n <= 20:
    ows = OTHER['weights']
    for i in range(len(ps)):
      if i < len(ows):
        ow = ows[i]
      else:
        ow = ws[i]
      nw = ws[i]
      ws[i] = (ow + nw) / 2

  ## sort the particle list by weight descending
  ps_r = list(reversed(sorted( ps_r , key=lambda pr: pr[1])))
  ps = [p[0] for p in ps_r]

  ## Gradually cut off lower-weight particles according to the iteratio of the simulation
  if iter_n < 10:
    ## keep 5,000 particles at iteration 0, 4,000 at iteration 1, etc. until you're down to 1,000 particles
    ps = [p for p in ps[:1000 * max(1, 5 - iter_n)]]
  if iter_n >= 10:
    ps = ps[:500]
    ps = ps + ps[:100]
  if iter_n >= 20:
    ps = ps[:250]
    ps = ps + ps[:50]
  if iter_n >= 30:
    ps = ps[:100]
    ps = ps + ps[:25]

  ## Summary to print
  #print(iter_n, len(ps), np.median([p[1] for p in ps_r]), np.round( [p[1] for p in ps_r[:10]], 2 ), np.round( [p[1] for p in ps_r[-10:]], 2 ) )
  #print(len(ps_r), np.mean([p[1] for p in ps_r]), np.max([p[1] for p in ps_r]), np.min([p[1] for p in ps_r]))

  ## average partcile positions for estimate
  x_est = np.mean([p.x for p in ps])
  y_est = np.mean([p.y for p in ps])

  ## points to plot in the visualization
  optionalPointsToPlot = [(p.x, p.y, p.heading) for p in ps]

  ## update the data we pass into the next iteration
  OTHER['iter'] += 1
  OTHER['particles'] = ps
  OTHER['weights'] = [p[1] for p in ps_r]

  if np.mean([p[1] for p in ps_r]) >= .051:
    OTHER['found_flag'] = True

  ## return
  return (x_est, y_est), OTHER, optionalPointsToPlot


# This is the function you will have to write for part B. The goal in part B
# is to navigate your glider towards (0,0) on the map steering # the glider 
# using its rudder. Note that the Z height is unimportant.

#
# The input parameters are exactly the same as for part A.

def next_angle(height, radar, mapFunc, OTHER=None):

  ## Get glider position measurement from particle filter
  (x_est, y_est), OTHER, a_points_plot = estimate_next_pos(height, radar, mapFunc, OTHER)

  iter_n = OTHER['iter']
  ps = OTHER['particles']
  zp = OTHER['zero_points']

  ## If the position estimation thinks it's found the glider, steer the glider towards (0,0)
  if OTHER['found_flag']:
    ## estimated heading from particles
    est_heading = np.mean( [p.heading for p in ps] )

    ## get reflection of current position across (0,0), to steer toward that heading
    aim_x = -1 * x_est
    aim_y = -1 * y_est

    aim_heading = np.arctan2(aim_y, aim_x) ## radian of heading we want to aim

    ## change steering angle by taking the difference between the aim heading and our current heading
    steering_angle = aim_heading - est_heading
    OTHER['steering_angle'] = steering_angle

  ## if we haven't found the glider, don't steer it yet
  else:
    steering_angle = 0.0

  # Returns
  optionalPointsToPlot = zp[:] + a_points_plot
  return steering_angle, OTHER, optionalPointsToPlot

def who_am_i():
    # Please specify your GT login ID in the whoami variable (ex: jsmith123).
    whoami = 'sshepherd35'
    return whoami

def Gaussian(mu, sigma, x):
  ''' From Sebastian lecture'''
  # calculates the probability of x for 1-dim Gaussian with mean mu and var. sigma
  return exp(- ((mu - x) ** 2) / (sigma ** 2) / 2.0) / sqrt(2.0 * pi * (sigma ** 2))

def init_particles(num_particles, radar, height, mapFunc):
  '''Initializes lots of particles then takes the top num_particles based on measurement similarity to the real glider'''
  counter = 0
  ps = []
  while counter < 50000:
    x = np.random.uniform(low = -260, high = 260)
    y = np.random.uniform(low = -260, high = 260)
    er = mapFunc(x,y)
    mp = Gaussian( radar, measurement_noise, height - er + np.random.normal(0, scale = measurement_noise) )
    #print('init meas:',mp)
    g = glider( 
      #x = np.random.normal(0, scale=init_x_noise),
      #y = np.random.normal(0, scale=init_y_noise),
      x = x,
      y = y,
      z = height,
      heading = np.random.normal(0, scale=init_r_noise),
      mapFunc = mapFunc
    )
    g.set_noise(measurement_noise, turning_noise, altitude_noise)
    #g = g.glide()

    ps.append( (g,mp) )
    counter += 1

  ps = list(reversed(sorted( ps , key=lambda p: p[1])))
  ps = [p[0] for p in ps[:num_particles]]

  return ps

def init_zero_points():
  '''Initializes points around (0,0) to draw on the map'''
  zp = []

  for i in range(100):
    zp.append( (np.random.normal(0, scale=4.5), np.random.normal(0, scale=4.5)) )

  return zp