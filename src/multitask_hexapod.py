#! /usr/bin/env python3
# Kinematic arm experiment from:
# Mouret JB and Maguire G. (2020) Quality Diversity for Multitask Optimization. Proc of ACM GECCO/

# (so that we do not need to install the module properly)
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'pymap_elites')))

import time
import math
import numpy as np
import pybullet

import pyhexapod.simulator as simulator
import pycontrollers.hexapod_controller as ctrl
import map_elites.multitask as mt_map_elites
import map_elites.common as cm_map_elites


def hexapod(x, features):
    #t0 = time.perf_counter()
    urdf_file = features[1]
    simu = simulator.HexapodSimulator(gui=False, urdf=urdf_file,video='')
    controller = ctrl.HexapodController(x)
    dead = False
    fit = -1e10
    steps = 3. / simu.dt
    i = 0
    while i < steps and not dead:
        simu.step(controller)
        p = simu.get_pos()[0] 
        a = pybullet.getEulerFromQuaternion(simu.get_pos()[1])
        out_of_corridor = abs(p[1]) > 0.5
        out_of_angles = abs(a[0]) > math.pi/8 or abs(a[1]) > math.pi/8 or abs(a[2]) > math.pi/8
        if out_of_angles or out_of_corridor:
            dead = True
        i += 1
    fit = p[0]
    #print(time.perf_counter() - t0, " ms", '=>', fit)
    return fit 


def load(directory, k):
    tasks = []
    centroids = []
    for i in range(0, k):
        centroid = np.loadtxt(directory + '/lengthes_' + str(i) + '.txt')
        urdf_file = directory + '/pexod_' + str(i) + '.urdf'
        centroids += [centroid]
        tasks += [(centroid, urdf_file)]
    return np.array(centroids), tasks

print('loading URDF files...', end='')
centroids, tasks = load(sys.argv[1], 2000) # 2000 tasks
print('data loaded')
dim_x = 36

px = cm_map_elites.default_params.copy()
px['min'] = np.zeros(dim_x)
px['max'] = np.ones(dim_x)
px['parallel'] = True


archive = mt_map_elites.compute(dim_map=2, dim_x=dim_x, f=hexapod, centroids=centroids, tasks=tasks, num_evals=1e6, params=px, log_file=open('cover_max_mean.dat', 'w'))
