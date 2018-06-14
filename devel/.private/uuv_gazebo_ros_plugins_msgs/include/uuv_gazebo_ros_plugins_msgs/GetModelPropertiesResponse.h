// Generated by gencpp from file uuv_gazebo_ros_plugins_msgs/GetModelPropertiesResponse.msg
// DO NOT EDIT!


#ifndef UUV_GAZEBO_ROS_PLUGINS_MSGS_MESSAGE_GETMODELPROPERTIESRESPONSE_H
#define UUV_GAZEBO_ROS_PLUGINS_MSGS_MESSAGE_GETMODELPROPERTIESRESPONSE_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <uuv_gazebo_ros_plugins_msgs/UnderwaterObjectModel.h>

namespace uuv_gazebo_ros_plugins_msgs
{
template <class ContainerAllocator>
struct GetModelPropertiesResponse_
{
  typedef GetModelPropertiesResponse_<ContainerAllocator> Type;

  GetModelPropertiesResponse_()
    : link_names()
    , models()  {
    }
  GetModelPropertiesResponse_(const ContainerAllocator& _alloc)
    : link_names(_alloc)
    , models(_alloc)  {
  (void)_alloc;
    }



   typedef std::vector<std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other > , typename ContainerAllocator::template rebind<std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other > >::other >  _link_names_type;
  _link_names_type link_names;

   typedef std::vector< ::uuv_gazebo_ros_plugins_msgs::UnderwaterObjectModel_<ContainerAllocator> , typename ContainerAllocator::template rebind< ::uuv_gazebo_ros_plugins_msgs::UnderwaterObjectModel_<ContainerAllocator> >::other >  _models_type;
  _models_type models;




  typedef boost::shared_ptr< ::uuv_gazebo_ros_plugins_msgs::GetModelPropertiesResponse_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::uuv_gazebo_ros_plugins_msgs::GetModelPropertiesResponse_<ContainerAllocator> const> ConstPtr;

}; // struct GetModelPropertiesResponse_

typedef ::uuv_gazebo_ros_plugins_msgs::GetModelPropertiesResponse_<std::allocator<void> > GetModelPropertiesResponse;

typedef boost::shared_ptr< ::uuv_gazebo_ros_plugins_msgs::GetModelPropertiesResponse > GetModelPropertiesResponsePtr;
typedef boost::shared_ptr< ::uuv_gazebo_ros_plugins_msgs::GetModelPropertiesResponse const> GetModelPropertiesResponseConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::uuv_gazebo_ros_plugins_msgs::GetModelPropertiesResponse_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::uuv_gazebo_ros_plugins_msgs::GetModelPropertiesResponse_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace uuv_gazebo_ros_plugins_msgs

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': False, 'IsMessage': True, 'HasHeader': False}
// {'uuv_gazebo_ros_plugins_msgs': ['/home/amishsqrrob/uuv_simulator/src/uuv_gazebo_plugins/uuv_gazebo_ros_plugins_msgs/msg'], 'std_msgs': ['/opt/ros/kinetic/share/std_msgs/cmake/../msg'], 'geometry_msgs': ['/opt/ros/kinetic/share/geometry_msgs/cmake/../msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::uuv_gazebo_ros_plugins_msgs::GetModelPropertiesResponse_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::uuv_gazebo_ros_plugins_msgs::GetModelPropertiesResponse_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::uuv_gazebo_ros_plugins_msgs::GetModelPropertiesResponse_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::uuv_gazebo_ros_plugins_msgs::GetModelPropertiesResponse_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::uuv_gazebo_ros_plugins_msgs::GetModelPropertiesResponse_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::uuv_gazebo_ros_plugins_msgs::GetModelPropertiesResponse_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::uuv_gazebo_ros_plugins_msgs::GetModelPropertiesResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "222d64ab6fa46c24f1abd065170ebe7a";
  }

  static const char* value(const ::uuv_gazebo_ros_plugins_msgs::GetModelPropertiesResponse_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x222d64ab6fa46c24ULL;
  static const uint64_t static_value2 = 0xf1abd065170ebe7aULL;
};

template<class ContainerAllocator>
struct DataType< ::uuv_gazebo_ros_plugins_msgs::GetModelPropertiesResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "uuv_gazebo_ros_plugins_msgs/GetModelPropertiesResponse";
  }

  static const char* value(const ::uuv_gazebo_ros_plugins_msgs::GetModelPropertiesResponse_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::uuv_gazebo_ros_plugins_msgs::GetModelPropertiesResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "string[]  link_names\n\
uuv_gazebo_ros_plugins_msgs/UnderwaterObjectModel[] models\n\
\n\
\n\
================================================================================\n\
MSG: uuv_gazebo_ros_plugins_msgs/UnderwaterObjectModel\n\
# Copyright (c) 2016 The UUV Simulator Authors.\n\
# All rights reserved.\n\
#\n\
# Licensed under the Apache License, Version 2.0 (the \"License\");\n\
# you may not use this file except in compliance with the License.\n\
# You may obtain a copy of the License at\n\
#\n\
#     http://www.apache.org/licenses/LICENSE-2.0\n\
#\n\
# Unless required by applicable law or agreed to in writing, software\n\
# distributed under the License is distributed on an \"AS IS\" BASIS,\n\
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n\
# See the License for the specific language governing permissions and\n\
# limitations under the License.\n\
\n\
float64[] added_mass\n\
float64[] linear_damping\n\
float64[] linear_damping_forward_speed\n\
float64[] quadratic_damping\n\
float64 volume\n\
float64 bbox_height\n\
float64 bbox_length\n\
float64 bbox_width\n\
float64 fluid_density\n\
geometry_msgs/Vector3 cob\n\
bool neutrally_buoyant\n\
geometry_msgs/Inertia inertia\n\
================================================================================\n\
MSG: geometry_msgs/Vector3\n\
# This represents a vector in free space. \n\
# It is only meant to represent a direction. Therefore, it does not\n\
# make sense to apply a translation to it (e.g., when applying a \n\
# generic rigid transformation to a Vector3, tf2 will only apply the\n\
# rotation). If you want your data to be translatable too, use the\n\
# geometry_msgs/Point message instead.\n\
\n\
float64 x\n\
float64 y\n\
float64 z\n\
================================================================================\n\
MSG: geometry_msgs/Inertia\n\
# Mass [kg]\n\
float64 m\n\
\n\
# Center of mass [m]\n\
geometry_msgs/Vector3 com\n\
\n\
# Inertia Tensor [kg-m^2]\n\
#     | ixx ixy ixz |\n\
# I = | ixy iyy iyz |\n\
#     | ixz iyz izz |\n\
float64 ixx\n\
float64 ixy\n\
float64 ixz\n\
float64 iyy\n\
float64 iyz\n\
float64 izz\n\
";
  }

  static const char* value(const ::uuv_gazebo_ros_plugins_msgs::GetModelPropertiesResponse_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::uuv_gazebo_ros_plugins_msgs::GetModelPropertiesResponse_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.link_names);
      stream.next(m.models);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct GetModelPropertiesResponse_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::uuv_gazebo_ros_plugins_msgs::GetModelPropertiesResponse_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::uuv_gazebo_ros_plugins_msgs::GetModelPropertiesResponse_<ContainerAllocator>& v)
  {
    s << indent << "link_names[]" << std::endl;
    for (size_t i = 0; i < v.link_names.size(); ++i)
    {
      s << indent << "  link_names[" << i << "]: ";
      Printer<std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other > >::stream(s, indent + "  ", v.link_names[i]);
    }
    s << indent << "models[]" << std::endl;
    for (size_t i = 0; i < v.models.size(); ++i)
    {
      s << indent << "  models[" << i << "]: ";
      s << std::endl;
      s << indent;
      Printer< ::uuv_gazebo_ros_plugins_msgs::UnderwaterObjectModel_<ContainerAllocator> >::stream(s, indent + "    ", v.models[i]);
    }
  }
};

} // namespace message_operations
} // namespace ros

#endif // UUV_GAZEBO_ROS_PLUGINS_MSGS_MESSAGE_GETMODELPROPERTIESRESPONSE_H
