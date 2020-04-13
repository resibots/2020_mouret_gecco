#! /usr/bin/env python3
# Kinematic arm experiment from:
# Mouret JB and Maguire G. (2020) Quality Diversity for Multitask Optimization. Proc of ACM GECCO/

# (so that we do not need to install the module properly)
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'pymap_elites')))

import math
import numpy as np
import map_elites.multitask as mt_map_elites
import map_elites.common as cm_map_elites


class Arm:
    def __init__(self, lengths):
        self.n_dofs = len(lengths)
        self.lengths = np.concatenate(([0], lengths))
        self.joint_xy = []

    def fw_kinematics(self, p):
        from math import cos, sin, pi, sqrt
        assert(len(p) == self.n_dofs)
        p = np.append(p, 0)
        self.joint_xy = []
        mat = np.matrix(np.identity(4))
        for i in range(0, self.n_dofs + 1):
            m = [[cos(p[i]), -sin(p[i]), 0, self.lengths[i]],
                 [sin(p[i]),  cos(p[i]), 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]]
            mat = mat * np.matrix(m)
            v = mat * np.matrix([0, 0, 0, 1]).transpose()
            self.joint_xy += [v[0:2].A.flatten()]
        return self.joint_xy[self.n_dofs], self.joint_xy



def fitness_arm(angles, task):
    angular_range = task[0] / len(angles)
    lengths = np.ones(len(angles)) * task[1] / len(angles)
    target = 0.5 * np.ones(2)
    a = Arm(lengths)
    command = (angles - 0.5) * angular_range * math.pi * 2
    ef, _ = a.fw_kinematics(command)
    f = -np.linalg.norm(ef - target)
    return f

    
if len(sys.argv) == 1 or ('help' in sys.argv):
    print("Usage: \"python3 ./examples/multitask_arm.py dimension [no_distance]\"")
    exit(0)


dim_x = int(sys.argv[1])

# dim_map, dim_x, function
px = cm_map_elites.default_params.copy()
px["dump_period"] = 2000
px["min"] = np.zeros(dim_x)
px["max"] = np.ones(dim_x)
px["parallel"] = False

n_tasks = 5000
dim_map = 2

# example : create centroids using a CVT (you can also create them randomly)
c = cm_map_elites.cvt(n_tasks, dim_map, 30000, True)

# CVT-based version
if len(sys.argv) == 2 or sys.argv[2] == 'distance':
    archive = mt_map_elites.compute(dim_x=dim_x, f=fitness_arm, centroids=c, num_evals=1e6, params=px, log_file=open('cover_max_mean.dat', 'w'))
else:
    # no distance:
    archive = mt_map_elites.compute(dim_x=dim_x, f=fitness_arm, tasks=c, num_evals=1e6, params=px, log_file=open('cover_max_mean.dat', 'w'))
