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
    Arm dynamics Inertia and Gravitational matrices
    """

    LABEL = 'Arm dynamics Inertia and Gravitational matrices'

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

        self._d = tuple(rospy.get_param(d_tag))
        self._a = tuple(rospy.get_param(a_tag))
        self._alpha = tuple(rospy.get_param(alpha_tag))
        self._js_offset = rospy.get_param(js_offset_tag)
        self._js_sign = rospy.get_param(js_sign_tag)
        self._d_size = len(self._d)
        com_i = tuple(rospy.get_param(com_i_tag))
        self._com = []
        for i in range(self._d_size):
            self._com.append(sym.Matrix([com_i[4*i], com_i[4*i+1], com_i[4*i+2], com_i[4*i+3]]))
        self._com = tuple(self._com)

        # build extended mass matrix
        M = tuple(rospy.get_param(M_tag))
        self._M_ext = []
        for i in M:
            self._M_ext.append(i * sym.eye(self._d_size))
        # create vector of symbolic joints
        self._q = []
        for i in range(self._d_size):
            self._q.append(sym.symbols('q'+str(i+1)))
        self._q = tuple(self._q)

        self._hTi = []
        self._aTn = sym.eye(4)
        self._aTi = [sym.eye(4)]
        self._o_i = [sym.Matrix([0, 0, 0])]
        self._z_i = [sym.Matrix([0, 0, 1])]
        self._comz_i = None
        self._P_i = []
        self._P = sym.zeros(1)
        self._aJi = [sym.zeros(self._d_size), sym.zeros(self._d_size), sym.zeros(self._d_size), sym.zeros(self._d_size), sym.zeros(self._d_size), sym.zeros(self._d_size)]
        self._Gq = sym.zeros(self._d_size,1)
        self._Mq = sym.zeros(self._d_size)
        for i in range(len(self._q)):
            # Build homogeneous transformation matrix T from wrist to base frame with DH parameters
            self._hTi.append(self.T_DH(self._q[i], self._d[i], self._a[i], self._alpha[i]))
            self._aTn = self._aTn * self._hTi[i]
            self._aTi.append(self._aTn)
            # Get origins coordinates and rotation axis of each frame wrt base frame
            self._o_i.append(self._aTi[i+1][:3,3])
            self._z_i.append(self._aTi[i+1][:3,2])
            # Build potential energy term for the whole uuv_manipulator
            self._comz_i = (self._aTn[2,:] * self._com[i])
            self._P_i.append(self._M_ext[i][1,1] * ArmDyn.g * self._comz_i)
            self._P += self._P_i[i]

        for i in range(len(self._q)):
            # Build link Jacobian matrices Ji from link 1 to link n
            if i < 1:
                self._aJi[0][:3,i] = self._z_i[i].cross(self._o_i[1] - self._o_i[i])
                self._aJi[0][3:,i] = self._z_i[i]
            if i < 2:
                self._aJi[1][:3,i] = self._z_i[i].cross(self._o_i[2] - self._o_i[i])
                self._aJi[1][3:,i] = self._z_i[i]
            if i < 3:
                self._aJi[2][:3,i] = self._z_i[i].cross(self._o_i[3] - self._o_i[i])
                self._aJi[2][3:,i] = self._z_i[i]
            if i < 4:
                self._aJi[3][:3,i] = self._z_i[i].cross(self._o_i[4] - self._o_i[i])
                self._aJi[3][3:,i] = self._z_i[i]
            if i < 5:
                self._aJi[4][:3,i] = self._z_i[i].cross(self._o_i[5] - self._o_i[i])
                self._aJi[4][3:,i] = self._z_i[i]
            self._aJi[5][:3,i] = self._z_i[i].cross(self._o_i[6] - self._o_i[i])
            self._aJi[5][3:,i] = self._z_i[i]

            # Build gravitational vector Gq by differentiating potential energy of each link
            self._Gq[i] = sym.diff(self._P, self._q[i])

            # Build inertia Matrix Mq
            for i in range(self._d_size):
                self._Mq += self._aJi[i].T * self._M_ext[i] * self._aJi[i]

        self._joint_state = np.matrix(np.zeros(self._d_size)).T
        self._joint_sub = rospy.Subscriber("/"+self._uuv_name+"/joint_states", JointState, self._joint_callback)
        self._dyn_pub = rospy.Publisher("/"+self._uuv_name+"/"+self._arm_name+"/ManDyn", ManDyn, queue_size=10)
        self._rate = rospy.Rate(10)
        self._dynMsg = ManDyn()

    def T_DH(self, theta, d, a, alpha):
        return sym.Matrix([[sym.cos(theta), -sym.sin(theta)*sym.cos(alpha), sym.sin(theta)*sym.sin(alpha), a*sym.cos(theta)],
                [sym.sin(theta), sym.cos(theta)*sym.cos(alpha), -sym.cos(theta)*sym.sin(alpha), a*sym.sin(theta)],
                [0, sym.sin(alpha), sym.cos(alpha), d],
                [0, 0, 0, 1]])

    def get_Mq(self):
        return self._Mq

    def get_Gq(self):
        return self._Gq

    def Mq(self, q_actual, n=30):
        M_temp = self._Mq.evalf(n, subs={self._q[0]:q_actual[0], self._q[1]:q_actual[1], self._q[2]:q_actual[2],
                                         self._q[3]:q_actual[3], self._q[4]:q_actual[4], self._q[5]:q_actual[5]})
        return M_temp

    def Gq(self, q_actual, n=30):
        G_temp = self._Gq.evalf(n, subs={self._q[0]:q_actual[0], self._q[1]:q_actual[1], self._q[2]:q_actual[2],
                                         self._q[3]:q_actual[3], self._q[4]:q_actual[4], self._q[5]:q_actual[5]})
        return G_temp

    def _joint_callback(self, joint_state):
        self._joint_state = np.matrix(joint_state.position[1:self._d_size+1]).T
        for i in range(self._d_size):
            self._joint_state[i] = self._js_offset[i] + self._js_sign[i] * self._joint_state[i]

    def pub_dyn(self):
        while not rospy.is_shutdown():
            G_temp = self.Gq(self._joint_state)
            print 'G_temp: ', G_temp, '\n'
            self._dynMsg.Vector6 = np.array(G_temp[:])
            #self._dynMsg.Matrix6x6 = self.Mq(self._joint_state)
            self._dyn_pub.publish(self._dynMsg)
            self._rate.sleep()

if __name__ == '__main__':
    # Start the node
    node_name = os.path.splitext(os.path.basename(__file__))[0]
    rospy.init_node(node_name)
    rospy.loginfo('Starting [%s] node' % node_name)

    arm_dyn = ArmDyn()

    arm_dyn.pub_dyn()

    rospy.spin()
    rospy.loginfo('Shutting down [%s] node' % node_name)
