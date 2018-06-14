// Generated by gencpp from file uuv_world_ros_plugins_msgs/GetCurrentModelRequest.msg
// DO NOT EDIT!


#ifndef UUV_WORLD_ROS_PLUGINS_MSGS_MESSAGE_GETCURRENTMODELREQUEST_H
#define UUV_WORLD_ROS_PLUGINS_MSGS_MESSAGE_GETCURRENTMODELREQUEST_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace uuv_world_ros_plugins_msgs
{
template <class ContainerAllocator>
struct GetCurrentModelRequest_
{
  typedef GetCurrentModelRequest_<ContainerAllocator> Type;

  GetCurrentModelRequest_()
    {
    }
  GetCurrentModelRequest_(const ContainerAllocator& _alloc)
    {
  (void)_alloc;
    }






  typedef boost::shared_ptr< ::uuv_world_ros_plugins_msgs::GetCurrentModelRequest_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::uuv_world_ros_plugins_msgs::GetCurrentModelRequest_<ContainerAllocator> const> ConstPtr;

}; // struct GetCurrentModelRequest_

typedef ::uuv_world_ros_plugins_msgs::GetCurrentModelRequest_<std::allocator<void> > GetCurrentModelRequest;

typedef boost::shared_ptr< ::uuv_world_ros_plugins_msgs::GetCurrentModelRequest > GetCurrentModelRequestPtr;
typedef boost::shared_ptr< ::uuv_world_ros_plugins_msgs::GetCurrentModelRequest const> GetCurrentModelRequestConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::uuv_world_ros_plugins_msgs::GetCurrentModelRequest_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::uuv_world_ros_plugins_msgs::GetCurrentModelRequest_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace uuv_world_ros_plugins_msgs

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': True, 'IsMessage': True, 'HasHeader': False}
// {'geometry_msgs': ['/opt/ros/kinetic/share/geometry_msgs/cmake/../msg'], 'std_msgs': ['/opt/ros/kinetic/share/std_msgs/cmake/../msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::uuv_world_ros_plugins_msgs::GetCurrentModelRequest_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::uuv_world_ros_plugins_msgs::GetCurrentModelRequest_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::uuv_world_ros_plugins_msgs::GetCurrentModelRequest_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::uuv_world_ros_plugins_msgs::GetCurrentModelRequest_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::uuv_world_ros_plugins_msgs::GetCurrentModelRequest_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::uuv_world_ros_plugins_msgs::GetCurrentModelRequest_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::uuv_world_ros_plugins_msgs::GetCurrentModelRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "d41d8cd98f00b204e9800998ecf8427e";
  }

  static const char* value(const ::uuv_world_ros_plugins_msgs::GetCurrentModelRequest_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xd41d8cd98f00b204ULL;
  static const uint64_t static_value2 = 0xe9800998ecf8427eULL;
};

template<class ContainerAllocator>
struct DataType< ::uuv_world_ros_plugins_msgs::GetCurrentModelRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "uuv_world_ros_plugins_msgs/GetCurrentModelRequest";
  }

  static const char* value(const ::uuv_world_ros_plugins_msgs::GetCurrentModelRequest_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::uuv_world_ros_plugins_msgs::GetCurrentModelRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "\n\
\n\
\n\
\n\
\n\
\n\
\n\
\n\
\n\
\n\
\n\
\n\
\n\
\n\
\n\
";
  }

  static const char* value(const ::uuv_world_ros_plugins_msgs::GetCurrentModelRequest_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::uuv_world_ros_plugins_msgs::GetCurrentModelRequest_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream&, T)
    {}

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct GetCurrentModelRequest_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::uuv_world_ros_plugins_msgs::GetCurrentModelRequest_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream&, const std::string&, const ::uuv_world_ros_plugins_msgs::GetCurrentModelRequest_<ContainerAllocator>&)
  {}
};

} // namespace message_operations
} // namespace ros

#endif // UUV_WORLD_ROS_PLUGINS_MSGS_MESSAGE_GETCURRENTMODELREQUEST_H
