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
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState
from uuv_gazebo_ros_plugins_msgs.srv import GetModelProperties
from uuv_manipulators_msgs.msg import ManDyn
from geometry_msgs.msg import PoseStamped

import PyKDL
from uuv_manipulators_control import CartesianController


class SMC(CartesianController):
    """
    Cartesian space Sliding Mode controller
    """

    LABEL = 'Cartesian space Sliding Mode controller'
    def __init__(self):
        """
        Class constructor
        """
        CartesianController.__init__(self)
        # Retrieve the controller parameters from the parameter server
        Q_tag = '~Q'
        K_tag = '~K'
        lambda_tag = '~Lambda'
        T_tag = '~T'
        uuv_name_tag = '~uuv_name'
        arm_name_tag = '~arm_name'
        semi_autonom = '~semi_autonom'
        if not rospy.has_param(Q_tag):
            rospy.ROSException('Q gain matrix not available for tag=%s' % Q_tag)
        if not rospy.has_param(K_tag):
            rospy.ROSException('K gain matrix not available for tag=%s' % K_tag)
        if not rospy.has_param(lambda_tag):
            rospy.ROSException('Lambda gain matrix not available for tag=%s' % lambda_tag)
        if not rospy.has_param(T_tag):
            rospy.ROSException('T gain matrix not available for tag=%s' % T_tag)
        if not rospy.has_param(uuv_name_tag):
            rospy.ROSException('uuv name not available for tag=%s' % uuv_name_tag)
        if not rospy.has_param(arm_name_tag):
            rospy.ROSException('arm name not available for tag=%s' % arm_name_tag)
        if not rospy.has_param(semi_autonom):
            rospy.ROSException('semi-autonomous selection not available for tag=%s' % semi_autonom)

        # Last velocity reference in cartesian coordinates
        self._last_goal_vel = np.asmatrix(np.zeros(6)).T

        # Initialization flag, to wait the end-effector get to the home position
        self._is_init = False

        # Initialization of Sliding Variables
        # Sliding surface slope
        self._lambda = np.diagflat([rospy.get_param(lambda_tag)])
        # Robustness term multiplier
        self._Q = np.diagflat([rospy.get_param(Q_tag)])
        # PD multiplier
        self._K = np.diagflat([rospy.get_param(K_tag)])
        # hyperbolic tangent slope
        self._T = np.diagflat([rospy.get_param(T_tag)])

        # Gravitational matrix
        self._Gq = np.asmatrix(np.zeros(6)).T

        self._uuv_name = rospy.get_param(uuv_name_tag)
        self._arm_name = rospy.get_param(arm_name_tag)

        # Operation mode (not autonomous / semi-autonomous)
        if rospy.get_param(semi_autonom):
            self._semi_autonom = 1
        else:
            self._semi_autonom = 0

        # Semi-autonomous manipulator goal
        self._goal_semi_autonom = PoseStamped()

        self._last_time = rospy.get_time()

        self._update_model_props()

        # Topic that receives the gravitational matrix of the manipulator
        self._mandyn_sub = rospy.Subscriber("/"+self._uuv_name+"/"+self._arm_name+"/"+"man_dyn", ManDyn, self._mandyn_callback)

        # Topic that receives the semi-autonomous manipulator goal
        self._semi_autonom_goal_sub = rospy.Subscriber("/"+self._uuv_name+"/"+self._arm_name+"/"+"goal_semi_autonom", PoseStamped, self._semi_autonom_goal_callback)

        self._run()

    def _update(self):
        # Leave if ROS is not running or command is not valid
        if rospy.is_shutdown() or self._last_goal is None:
            return

        # Calculate the goal pose
        if self._semi_autonom and self._goal_semi_autonom:
            # goal.p[0] = self._goal_semi_autonom.pose.position.x
            # goal.p[1] = self._goal_semi_autonom.pose.position.y
            # goal.p[2] = self._goal_semi_autonom.pose.position.z
            goal_p_x = self._goal_semi_autonom.pose.position.x
            goal_p_y = self._goal_semi_autonom.pose.position.y
            goal_p_z = self._goal_semi_autonom.pose.position.z
            goal_p = PyKDL.Vector(goal_p_x, goal_p_y, goal_p_z)
            # goal.M = PyKDL.Rotation.Quaternion(self._goal_semi_autonom.pose.orientation.x, self._goal_semi_autonom.pose.orientation.y, self._goal_semi_autonom.pose.orientation.z, self._goal_semi_autonom.pose.orientation.w)
            goal_M = PyKDL.Rotation.Quaternion(self._goal_semi_autonom.pose.orientation.x, self._goal_semi_autonom.pose.orientation.y, self._goal_semi_autonom.pose.orientation.z, self._goal_semi_autonom.pose.orientation.w)
            goal = PyKDL.Frame(goal_M, goal_p)
        else:
            goal = self._get_goal()

        ######################################
        ### Sliding mode cartesian control ###
        ######################################

        # Calculate reference velocity
        time_step = rospy.get_time() - self._last_time
        self._last_time = rospy.get_time()
        if time_step > 0:
            goal_p_dot = (goal.p - self._last_goal.p) / time_step
        else:
            goal_p_dot = (goal.p - self._last_goal.p) / 0.01
        goal_vel = np.array([goal_p_dot[0], goal_p_dot[1], goal_p_dot[2], 0, 0, 0])
        # End-effector's pose
        ee_pose = self._arm_interface.get_ee_pose_as_frame()
        # End-effector's velocity
        ee_vel = self._arm_interface.get_ee_vel_as_kdl_twist()
        # Calculate pose error
        error_pos = PyKDL.diff(ee_pose, goal)
        error_pose = np.array([error_pos[0], error_pos[1], error_pos[2], error_pos[3], error_pos[4], error_pos[5]]).reshape((6,1))
        # Calculate velocity error
        ee_velo = np.array([ee_vel[0], ee_vel[1], ee_vel[2], ee_vel[3], ee_vel[5], ee_vel[5]])
        error_velo = (goal_vel - ee_velo).reshape((6,1))
        # Calculate sliding Variable
        s = np.dot(self._lambda, error_pose) + error_velo
        # Calculate reference acceleration
        if time_step > 0:
            goal_acc = (goal_vel - self._last_goal_vel) / time_step
        else:
            goal_acc = (goal_vel - self._last_goal_vel) / 0.01
        self._last_goal_vel = goal_vel
        # Calculate inertia matrix
        Mq = 0
        for key in self._linkloads:
            Mq += self._arm_interface.jacobian_transpose(end_link=key) * self._linkinertias[key] * self._arm_interface.jacobian(end_link=key)

        # Use masses different from the ones of the real vehicle to test !!!!
        # Wrenches - Inertial term
        tau_inertia = np.dot(Mq, np.asmatrix(goal_acc).T + np.dot(self._lambda, error_velo) + np.dot(self._Q, np.tanh(np.dot(self._T, s))))
        # Wrenches - PD term
        tau_pd = np.dot(self._K, s)
        # Wrenches - Gravitational term
        tau_gq = self._Gq
        # Compute jacobian transpose
        JT = self._arm_interface.jacobian_transpose()
        # Total wrench for sliding mode controller
        tau = JT * (tau_inertia + tau_pd + tau_gq)

        self.publish_joint_efforts(tau)

        # Store current pose target
        # if not (self._semi_autonom and self._goal_semi_autonom):
        if not (self._semi_autonom and self._goal_semi_autonom):
            self._last_goal = goal
            self.publish_goal()

    # Update model properties
    def _update_model_props(self):
        rospy.wait_for_service("/"+self._uuv_name+"/get_model_properties")
        self._get_model_props = rospy.ServiceProxy("/"+self._uuv_name+"/get_model_properties", GetModelProperties)
        self._linkloads = dict()
        self._linkinertias = dict()
        hydromodel = self._get_model_props()
        rho = hydromodel.models[0].fluid_density
        g = 9.806
        for index, name in enumerate(hydromodel.link_names):
            if not 'base' in name:
                B = rho * g * hydromodel.models[index].volume
                I = hydromodel.models[index].inertia
                M = np.zeros((6,6))
                np.fill_diagonal(M, (I.m, I.m, I.m, I.ixx, I.iyy, I.izz))
                self._linkloads[name] = np.matrix([0, 0, -I.m + B, 0, 0, 0]).T
                self._linkinertias[name] = np.asmatrix(M)

    def _mandyn_callback(self, mandyn):
        self._Gq = np.asmatrix(mandyn.gravitational).T

    def _semi_autonom_goal_callback(self, semi_autonom_goal):
        self._goal_semi_autonom = semi_autonom_goal

if __name__ == '__main__':
    # Start the node
    node_name = os.path.splitext(os.path.basename(__file__))[0]
    rospy.init_node(node_name)
    rospy.loginfo('Starting [%s] node' % node_name)

    smc_controller = SMC()

    rospy.spin()
    rospy.loginfo('Shutting down [%s] node' % node_name)
