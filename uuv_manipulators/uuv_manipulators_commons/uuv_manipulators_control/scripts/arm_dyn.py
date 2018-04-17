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

import sys
import os
import numpy as np
import sympy as sym
import rospy
from uuv_manipulators_msgs.msg import ManDyn
from sensor_msgs.msg import JointState

class ArmDyn(object):
    """
    Manipulator arm dynamic matrices
    obs: currently computing only the gravitational matrix
    """

    LABEL = 'Manipulator arm dynamic matrices'

    g = 9.81

    def __init__(self):

        # Retrieve the manipulator parameters from the parameter server
        uuv_name_tag = '~uuv_name'
        arm_name_tag = '~arm_name'
        d_tag = '~d'
        a_tag = '~a'
        alpha_tag = '~alpha'
        com_i_tag = '~com_i'
        M_tag = '~M'
        js_offset_tag = '~js_offset'
        js_sign_tag = '~js_sign'
        if not rospy.has_param(uuv_name_tag):
            rospy.ROSException('uuv_name not available for tag=%s' % uuv_name_tag)
        if not rospy.has_param(arm_name_tag):
            rospy.ROSException('arm_name not available for tag=%s' % arm_name_tag)
        if not rospy.has_param(d_tag):
            rospy.ROSException('d array not available for tag=%s' % d_tag)
        if not rospy.has_param(a_tag):
            rospy.ROSException('a array not available for tag=%s' % a_tag)
        if not rospy.has_param(alpha_tag):
            rospy.ROSException('alpha array not available for tag=%s' % alpha_tag)
        if not rospy.has_param(com_i_tag):
            rospy.ROSException('com_i array not available for tag=%s' % com_i_tag)
        if not rospy.has_param(M_tag):
            rospy.ROSException('M array not available for tag=%s' % M_tag)
        if not rospy.has_param(js_offset_tag):
            rospy.ROSException('js_offset array not available for tag=%s' % js_offset_tag)
        if not rospy.has_param(js_sign_tag):
            rospy.ROSException('js_sign array not available for tag=%s' % js_sign_tag)

        self._uuv_name = rospy.get_param(uuv_name_tag)
        self._arm_name = rospy.get_param(arm_name_tag)

        # Denavit Hartenberg configuration parameters
        self._d = rospy.get_param(d_tag)
        self._a = rospy.get_param(a_tag)
        self._alpha = rospy.get_param(alpha_tag)
        self._js_offset = rospy.get_param(js_offset_tag)
        self._js_sign = rospy.get_param(js_sign_tag)
        self._d_size = len(self._d)
        com_i = rospy.get_param(com_i_tag)
        self._com = []
        for i in range(self._d_size):
            self._com.append(np.transpose(np.matrix([com_i[4*i], com_i[4*i+1], com_i[4*i+2], com_i[4*i+3]])))
        self._com = self._com
        # Inertia matrix
        self._M = rospy.get_param(M_tag)
        # Differential interval for the gradient calculation
        self._delta_diff = np.array([0.001])
        # Manipulators potential energy
        self._P = np.array(0)
        # Manipulators gravitational matrix
        self._Gq = np.zeros((self._d_size,1))
        # Last step manipulators gravitational matrix
        self._Gq_last = np.zeros((self._d_size,1))

        # State of the manipulator joints
        self._joint_state = np.matrix(np.zeros(self._d_size)).T
        # Subscriber for the state of the manipulator joints
        self._joint_sub = rospy.Subscriber("/"+self._uuv_name+"/joint_states", JointState, self._joint_callback)
        # Publisher for the arm dynamic matrices
        self._dyn_pub = rospy.Publisher("/"+self._uuv_name+"/"+self._arm_name+"/"+"man_dyn", ManDyn, queue_size=10)

        self._rate = rospy.Rate(100)

        self._dynMsg = ManDyn()

    def pub_dyn(self):
        while not rospy.is_shutdown():
            # Compute approximation of the gradient of P at joint_states
            for i in range(self._d_size):
                dist_joint_state = self._joint_state
                dist_joint_state[i] += self._delta_diff
                P_sup = self._get_potential_energy(dist_joint_state)
                dist_joint_state = self._joint_state
                dist_joint_state[i] -= self._delta_diff
                P_inf = self._get_potential_energy(dist_joint_state)
                self._Gq[i] = (P_sup - P_inf) / (2 * self._delta_diff)
                # Outliers detection and elimination
                if rospy.get_time() > 1:
                    if np.abs(self._Gq[i] - self._Gq_last[i]) > 5:
                        self._Gq[i] = self._Gq_last[i]
                    if abs(self._Gq[i]) < 0.1:
                        self._Gq[i] = 0
                self._Gq_last[i] = self._Gq[i]
            # Composing and publishing the dynamic matrices message
            self._dynMsg.gravitational = np.array(self._Gq[:])
            self._dyn_pub.publish(self._dynMsg)
            self._rate.sleep()

    def _get_potential_energy(self, joint_values):
        # Compute manipulators potential energy
        _P_i = []
        _hTi = []
        _aTn = np.eye(4)
        _comz_i = None
        for i in range(self._d_size):
            # Build homogeneous transformation matrix T from wrist to base frame with DH parameters
            _hTi.append(self._t_dh(joint_values[i], self._d[i], self._a[i], self._alpha[i]))
            _aTn = _aTn * _hTi[i]
            # Build potential energy term for the whole uuv_manipulator
            _comz_i = (_aTn[2,:] * self._com[i])
            _P_i.append(self._M[i] * ArmDyn.g * _comz_i)
        self._P = np.sum(_P_i)
        return self._P

    def _t_dh(self, theta, d, a, alpha):
        # Compute the Denavit Hartenberg matrix
        return np.matrix([[np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
                [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
                [0, np.sin(alpha), np.cos(alpha), d],
                [0, 0, 0, 1]])

    def _joint_callback(self, joint_state):
        # Get manipulator joint states
        self._joint_state = np.matrix(joint_state.position[1:self._d_size+1]).T
        for i in range(self._d_size):
            self._joint_state[i] = self._js_offset[i] + self._js_sign[i] * self._joint_state[i]

if __name__ == '__main__':
    # Start the node
    node_name = os.path.splitext(os.path.basename(__file__))[0]
    rospy.init_node(node_name)
    rospy.loginfo('Starting [%s] node' % node_name)

    arm_dyn = ArmDyn()

    arm_dyn.pub_dyn()

    rospy.spin()
    rospy.loginfo('Shutting down [%s] node' % node_name)
