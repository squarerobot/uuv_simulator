#!/usr/bin/env python
# Copyright (c) 2016 The UUV Simulator Authors.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import rospy
import sys
import os
import numpy as np

import PyKDL
from uuv_manipulators_control import CartesianController


class SMC(CartesianController):
    """
    Joint space Sliding Mode controller
    """

    LABEL = 'Joint space Sliding Mode controller'
    def __init__(self):
        """
        Class constructor
        """
        CartesianController.__init__(self)
        # Retrieve the controller parameters from the parameter server
        Q_tag = '~Q'
        K_tag = '~K'
        if not rospy.has_param(Q_tag):
            rospy.ROSException('Q gain matrix not available for tag=%s' % Q_tag)
        if not rospy.has_param(K_tag):
            rospy.ROSException('K gain matrix not available for tag=%s' % K_tag)

        self._last_joint_goal = np.matrix([0, 0.5*np.pi, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).T

        # Initialization of Sliding Variables
        self._I = np.zeros((6,6))
        np.fill_diagonal(self._I, np.ones(6))
        self._I = np.asmatrix(self._I)

        self._lambda = self._I
        np.fill_diagonal(self._lambda, 1 * np.ones(6))

        self._T = self._I
        t = [.2, .2, .2, .2, .2, .2]
        np.fill_diagonal(self._T, np.divide(1.0, t) * np.ones(6))

        self._Q = self._I
        self._q = rospy.get_param(Q_tag)
        np.fill_diagonal(self._Q, self._q * np.ones(6))

        self._K = self._I
        self._k = rospy.get_param(K_tag)
        np.fill_diagonal(self._K, self._k * np.ones(6))

        self._sPub = rospy.Publisher('SlidingVariable', Float32MultiArray, queue_size=10)
        self._sMsg = Float32MultiArray()
        self._ePub = rospy.Publisher('errorVariable', Float32MultiArray, queue_size=10)
        self._eMsg = Float32MultiArray()

        # self._last_time = rospy.get_time()
        # self._last_qddot_cmd = np.asmatrix(np.zeros(6)).T

        # Initialization flag, to wait the end-effector get to the home
        # position
        self._is_init = False

        self._run()

    def _update(self):
        # Leave if ROS is not running or command is not valid
        if rospy.is_shutdown() or self._last_goal is None:
            return

        # Calculate the goal pose
        goal = self._get_goal()
        # # Calculate the joints goal
        # J_inv = self._arm_interface.jacobian_pseudo_inverse()
        # qdot_cmd = J_inv * np.asmatrix(self._command[6:]).T
        # joint_goal = np.matrix(np.append(self._arm_interface.inverse_kinematics(goal.p, goal.M.GetQuaternion()), qdot_cmd)).T
        # if joint_goal.item(0) is None:
        #     joint_goal = self._last_joint_goal
        # else:
        #     self._last_joint_goal = joint_goal

        # self._joint_state = np.matrix([self._arm_interface.joint_angles + self._arm_interface.joint_velocities]).T

        ########################################################################
        ########################################################################

        # End-effector's pose
        ee_pose = self._arm_interface.get_ee_pose_as_frame()
        # End-effector's velocity
        ee_vel = self._arm_interface.get_ee_vel_as_kdl_twist()
        # Calculate pose error
        error = PyKDL.diff(goal, ee_pose)
        # End-effector wrench to achieve target
        wrench = np.matrix(np.zeros(6)).T
        for i in range(len(wrench)):
            wrench[i] = -(10000 * error[i] + 0 * ee_vel[i])
            #wrench[i] = -(self._Kp[i] * error[i] + self._Kd[i] * ee_vel[i])

        # Compute jacobian transpose
        JT = self._arm_interface.jacobian_transpose()
        # Calculate the torques for the joints
        tau = JT * wrench
        # Store current pose target
        self._last_goal = goal

        self.publish_goal()
        self.publish_joint_efforts(tau)

if __name__ == '__main__':
    # Start the node
    node_name = os.path.splitext(os.path.basename(__file__))[0]
    rospy.init_node(node_name)
    rospy.loginfo('Starting [%s] node' % node_name)

    smc_controller = SMC()

    rospy.spin()
    rospy.loginfo('Shutting down [%s] node' % node_name)
