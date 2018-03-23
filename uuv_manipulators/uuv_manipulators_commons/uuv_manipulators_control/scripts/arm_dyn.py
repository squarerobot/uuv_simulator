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
        l_tag = '~l'
        d_tag = '~d'
        a_tag = '~a'
        alpha_tag = '~alpha'
        com_i_tag = '~com_i'
        m_tag = '~m'
        if not rospy.has_param(l_tag):
            rospy.ROSException('l array not available for tag=%s' % l_tag)
        if not rospy.has_param(d_tag):
            rospy.ROSException('d array not available for tag=%s' % d_tag)
        if not rospy.has_param(a_tag):
            rospy.ROSException('a array not available for tag=%s' % a_tag)
        if not rospy.has_param(alpha_tag):
            rospy.ROSException('alpha array not available for tag=%s' % alpha_tag)
        if not rospy.has_param(com_i_tag):
            rospy.ROSException('com_i array not available for tag=%s' % com_i_tag)
        if not rospy.has_param(m_tag):
            rospy.ROSException('m array not available for tag=%s' % m_tag)

    def T_DH(self, theta, d, a, alpha):
        print 'a'

    def get_Mq(self):
        print 'a'

    def get_Gq(self):
        print 'a'

    def Mq(self, q_actual):
        print 'a'

    def Gq(self, q_actual):
        print 'a'

    def _joint_callback(self, joint_state):
        print 'a'

    def pub_dyn(self):
        print 'a'

if __name__ == '__main__':
    # Start the node
    node_name = os.path.splitext(os.path.basename(__file__))[0]
    rospy.init_node(node_name)
    rospy.loginfo('Starting [%s] node' % node_name)

    arm_dyn = ArmDyn()

    arm_dyn.pub_dyn()

    rospy.spin()
    rospy.loginfo('Shutting down [%s] node' % node_name)
