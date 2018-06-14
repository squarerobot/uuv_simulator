// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: SensorImu.proto

#define INTERNAL_SUPPRESS_PROTOBUF_FIELD_DEPRECATION
#include "SensorImu.pb.h"

#include <algorithm>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/stubs/once.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/wire_format_lite_inl.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)

namespace sensor_msgs {
namespace msgs {

namespace {

const ::google::protobuf::Descriptor* Imu_descriptor_ = NULL;
const ::google::protobuf::internal::GeneratedMessageReflection*
  Imu_reflection_ = NULL;

}  // namespace


void protobuf_AssignDesc_SensorImu_2eproto() {
  protobuf_AddDesc_SensorImu_2eproto();
  const ::google::protobuf::FileDescriptor* file =
    ::google::protobuf::DescriptorPool::generated_pool()->FindFileByName(
      "SensorImu.proto");
  GOOGLE_CHECK(file != NULL);
  Imu_descriptor_ = file->message_type(0);
  static const int Imu_offsets_[6] = {
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Imu, orientation_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Imu, orientation_covariance_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Imu, angular_velocity_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Imu, angular_velocity_covariance_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Imu, linear_acceleration_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Imu, linear_acceleration_covariance_),
  };
  Imu_reflection_ =
    new ::google::protobuf::internal::GeneratedMessageReflection(
      Imu_descriptor_,
      Imu::default_instance_,
      Imu_offsets_,
      GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Imu, _has_bits_[0]),
      GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Imu, _unknown_fields_),
      -1,
      ::google::protobuf::DescriptorPool::generated_pool(),
      ::google::protobuf::MessageFactory::generated_factory(),
      sizeof(Imu));
}

namespace {

GOOGLE_PROTOBUF_DECLARE_ONCE(protobuf_AssignDescriptors_once_);
inline void protobuf_AssignDescriptorsOnce() {
  ::google::protobuf::GoogleOnceInit(&protobuf_AssignDescriptors_once_,
                 &protobuf_AssignDesc_SensorImu_2eproto);
}

void protobuf_RegisterTypes(const ::std::string&) {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedMessage(
    Imu_descriptor_, &Imu::default_instance());
}

}  // namespace

void protobuf_ShutdownFile_SensorImu_2eproto() {
  delete Imu::default_instance_;
  delete Imu_reflection_;
}

void protobuf_AddDesc_SensorImu_2eproto() {
  static bool already_here = false;
  if (already_here) return;
  already_here = true;
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  ::gazebo::msgs::protobuf_AddDesc_quaternion_2eproto();
  ::gazebo::msgs::protobuf_AddDesc_vector3d_2eproto();
  ::google::protobuf::DescriptorPool::InternalAddGeneratedFile(
    "\n\017SensorImu.proto\022\020sensor_msgs.msgs\032\020qua"
    "ternion.proto\032\016vector3d.proto\"\221\002\n\003Imu\022,\n"
    "\013orientation\030\001 \002(\0132\027.gazebo.msgs.Quatern"
    "ion\022\"\n\026orientation_covariance\030\002 \003(\002B\002\020\001\022"
    "/\n\020angular_velocity\030\003 \002(\0132\025.gazebo.msgs."
    "Vector3d\022\'\n\033angular_velocity_covariance\030"
    "\004 \003(\002B\002\020\001\0222\n\023linear_acceleration\030\005 \002(\0132\025"
    ".gazebo.msgs.Vector3d\022*\n\036linear_accelera"
    "tion_covariance\030\006 \003(\002B\002\020\001", 345);
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedFile(
    "SensorImu.proto", &protobuf_RegisterTypes);
  Imu::default_instance_ = new Imu();
  Imu::default_instance_->InitAsDefaultInstance();
  ::google::protobuf::internal::OnShutdown(&protobuf_ShutdownFile_SensorImu_2eproto);
}

// Force AddDescriptors() to be called at static initialization time.
struct StaticDescriptorInitializer_SensorImu_2eproto {
  StaticDescriptorInitializer_SensorImu_2eproto() {
    protobuf_AddDesc_SensorImu_2eproto();
  }
} static_descriptor_initializer_SensorImu_2eproto_;

// ===================================================================

#ifndef _MSC_VER
const int Imu::kOrientationFieldNumber;
const int Imu::kOrientationCovarianceFieldNumber;
const int Imu::kAngularVelocityFieldNumber;
const int Imu::kAngularVelocityCovarianceFieldNumber;
const int Imu::kLinearAccelerationFieldNumber;
const int Imu::kLinearAccelerationCovarianceFieldNumber;
#endif  // !_MSC_VER

Imu::Imu()
  : ::google::protobuf::Message() {
  SharedCtor();
  // @@protoc_insertion_point(constructor:sensor_msgs.msgs.Imu)
}

void Imu::InitAsDefaultInstance() {
  orientation_ = const_cast< ::gazebo::msgs::Quaternion*>(&::gazebo::msgs::Quaternion::default_instance());
  angular_velocity_ = const_cast< ::gazebo::msgs::Vector3d*>(&::gazebo::msgs::Vector3d::default_instance());
  linear_acceleration_ = const_cast< ::gazebo::msgs::Vector3d*>(&::gazebo::msgs::Vector3d::default_instance());
}

Imu::Imu(const Imu& from)
  : ::google::protobuf::Message() {
  SharedCtor();
  MergeFrom(from);
  // @@protoc_insertion_point(copy_constructor:sensor_msgs.msgs.Imu)
}

void Imu::SharedCtor() {
  _cached_size_ = 0;
  orientation_ = NULL;
  angular_velocity_ = NULL;
  linear_acceleration_ = NULL;
  ::memset(_has_bits_, 0, sizeof(_has_bits_));
}

Imu::~Imu() {
  // @@protoc_insertion_point(destructor:sensor_msgs.msgs.Imu)
  SharedDtor();
}

void Imu::SharedDtor() {
  if (this != default_instance_) {
    delete orientation_;
    delete angular_velocity_;
    delete linear_acceleration_;
  }
}

void Imu::SetCachedSize(int size) const {
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
}
const ::google::protobuf::Descriptor* Imu::descriptor() {
  protobuf_AssignDescriptorsOnce();
  return Imu_descriptor_;
}

const Imu& Imu::default_instance() {
  if (default_instance_ == NULL) protobuf_AddDesc_SensorImu_2eproto();
  return *default_instance_;
}

Imu* Imu::default_instance_ = NULL;

Imu* Imu::New() const {
  return new Imu;
}

void Imu::Clear() {
  if (_has_bits_[0 / 32] & 21) {
    if (has_orientation()) {
      if (orientation_ != NULL) orientation_->::gazebo::msgs::Quaternion::Clear();
    }
    if (has_angular_velocity()) {
      if (angular_velocity_ != NULL) angular_velocity_->::gazebo::msgs::Vector3d::Clear();
    }
    if (has_linear_acceleration()) {
      if (linear_acceleration_ != NULL) linear_acceleration_->::gazebo::msgs::Vector3d::Clear();
    }
  }
  orientation_covariance_.Clear();
  angular_velocity_covariance_.Clear();
  linear_acceleration_covariance_.Clear();
  ::memset(_has_bits_, 0, sizeof(_has_bits_));
  mutable_unknown_fields()->Clear();
}

bool Imu::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!(EXPRESSION)) goto failure
  ::google::protobuf::uint32 tag;
  // @@protoc_insertion_point(parse_start:sensor_msgs.msgs.Imu)
  for (;;) {
    ::std::pair< ::google::protobuf::uint32, bool> p = input->ReadTagWithCutoff(127);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // required .gazebo.msgs.Quaternion orientation = 1;
      case 1: {
        if (tag == 10) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadMessageNoVirtual(
               input, mutable_orientation()));
        } else {
          goto handle_unusual;
        }
        if (input->ExpectTag(18)) goto parse_orientation_covariance;
        break;
      }

      // repeated float orientation_covariance = 2 [packed = true];
      case 2: {
        if (tag == 18) {
         parse_orientation_covariance:
          DO_((::google::protobuf::internal::WireFormatLite::ReadPackedPrimitive<
                   float, ::google::protobuf::internal::WireFormatLite::TYPE_FLOAT>(
                 input, this->mutable_orientation_covariance())));
        } else if (tag == 21) {
          DO_((::google::protobuf::internal::WireFormatLite::ReadRepeatedPrimitiveNoInline<
                   float, ::google::protobuf::internal::WireFormatLite::TYPE_FLOAT>(
                 1, 18, input, this->mutable_orientation_covariance())));
        } else {
          goto handle_unusual;
        }
        if (input->ExpectTag(26)) goto parse_angular_velocity;
        break;
      }

      // required .gazebo.msgs.Vector3d angular_velocity = 3;
      case 3: {
        if (tag == 26) {
         parse_angular_velocity:
          DO_(::google::protobuf::internal::WireFormatLite::ReadMessageNoVirtual(
               input, mutable_angular_velocity()));
        } else {
          goto handle_unusual;
        }
        if (input->ExpectTag(34)) goto parse_angular_velocity_covariance;
        break;
      }

      // repeated float angular_velocity_covariance = 4 [packed = true];
      case 4: {
        if (tag == 34) {
         parse_angular_velocity_covariance:
          DO_((::google::protobuf::internal::WireFormatLite::ReadPackedPrimitive<
                   float, ::google::protobuf::internal::WireFormatLite::TYPE_FLOAT>(
                 input, this->mutable_angular_velocity_covariance())));
        } else if (tag == 37) {
          DO_((::google::protobuf::internal::WireFormatLite::ReadRepeatedPrimitiveNoInline<
                   float, ::google::protobuf::internal::WireFormatLite::TYPE_FLOAT>(
                 1, 34, input, this->mutable_angular_velocity_covariance())));
        } else {
          goto handle_unusual;
        }
        if (input->ExpectTag(42)) goto parse_linear_acceleration;
        break;
      }

      // required .gazebo.msgs.Vector3d linear_acceleration = 5;
      case 5: {
        if (tag == 42) {
         parse_linear_acceleration:
          DO_(::google::protobuf::internal::WireFormatLite::ReadMessageNoVirtual(
               input, mutable_linear_acceleration()));
        } else {
          goto handle_unusual;
        }
        if (input->ExpectTag(50)) goto parse_linear_acceleration_covariance;
        break;
      }

      // repeated float linear_acceleration_covariance = 6 [packed = true];
      case 6: {
        if (tag == 50) {
         parse_linear_acceleration_covariance:
          DO_((::google::protobuf::internal::WireFormatLite::ReadPackedPrimitive<
                   float, ::google::protobuf::internal::WireFormatLite::TYPE_FLOAT>(
                 input, this->mutable_linear_acceleration_covariance())));
        } else if (tag == 53) {
          DO_((::google::protobuf::internal::WireFormatLite::ReadRepeatedPrimitiveNoInline<
                   float, ::google::protobuf::internal::WireFormatLite::TYPE_FLOAT>(
                 1, 50, input, this->mutable_linear_acceleration_covariance())));
        } else {
          goto handle_unusual;
        }
        if (input->ExpectAtEnd()) goto success;
        break;
      }

      default: {
      handle_unusual:
        if (tag == 0 ||
            ::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_END_GROUP) {
          goto success;
        }
        DO_(::google::protobuf::internal::WireFormat::SkipField(
              input, tag, mutable_unknown_fields()));
        break;
      }
    }
  }
success:
  // @@protoc_insertion_point(parse_success:sensor_msgs.msgs.Imu)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:sensor_msgs.msgs.Imu)
  return false;
#undef DO_
}

void Imu::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:sensor_msgs.msgs.Imu)
  // required .gazebo.msgs.Quaternion orientation = 1;
  if (has_orientation()) {
    ::google::protobuf::internal::WireFormatLite::WriteMessageMaybeToArray(
      1, this->orientation(), output);
  }

  // repeated float orientation_covariance = 2 [packed = true];
  if (this->orientation_covariance_size() > 0) {
    ::google::protobuf::internal::WireFormatLite::WriteTag(2, ::google::protobuf::internal::WireFormatLite::WIRETYPE_LENGTH_DELIMITED, output);
    output->WriteVarint32(_orientation_covariance_cached_byte_size_);
  }
  for (int i = 0; i < this->orientation_covariance_size(); i++) {
    ::google::protobuf::internal::WireFormatLite::WriteFloatNoTag(
      this->orientation_covariance(i), output);
  }

  // required .gazebo.msgs.Vector3d angular_velocity = 3;
  if (has_angular_velocity()) {
    ::google::protobuf::internal::WireFormatLite::WriteMessageMaybeToArray(
      3, this->angular_velocity(), output);
  }

  // repeated float angular_velocity_covariance = 4 [packed = true];
  if (this->angular_velocity_covariance_size() > 0) {
    ::google::protobuf::internal::WireFormatLite::WriteTag(4, ::google::protobuf::internal::WireFormatLite::WIRETYPE_LENGTH_DELIMITED, output);
    output->WriteVarint32(_angular_velocity_covariance_cached_byte_size_);
  }
  for (int i = 0; i < this->angular_velocity_covariance_size(); i++) {
    ::google::protobuf::internal::WireFormatLite::WriteFloatNoTag(
      this->angular_velocity_covariance(i), output);
  }

  // required .gazebo.msgs.Vector3d linear_acceleration = 5;
  if (has_linear_acceleration()) {
    ::google::protobuf::internal::WireFormatLite::WriteMessageMaybeToArray(
      5, this->linear_acceleration(), output);
  }

  // repeated float linear_acceleration_covariance = 6 [packed = true];
  if (this->linear_acceleration_covariance_size() > 0) {
    ::google::protobuf::internal::WireFormatLite::WriteTag(6, ::google::protobuf::internal::WireFormatLite::WIRETYPE_LENGTH_DELIMITED, output);
    output->WriteVarint32(_linear_acceleration_covariance_cached_byte_size_);
  }
  for (int i = 0; i < this->linear_acceleration_covariance_size(); i++) {
    ::google::protobuf::internal::WireFormatLite::WriteFloatNoTag(
      this->linear_acceleration_covariance(i), output);
  }

  if (!unknown_fields().empty()) {
    ::google::protobuf::internal::WireFormat::SerializeUnknownFields(
        unknown_fields(), output);
  }
  // @@protoc_insertion_point(serialize_end:sensor_msgs.msgs.Imu)
}

::google::protobuf::uint8* Imu::SerializeWithCachedSizesToArray(
    ::google::protobuf::uint8* target) const {
  // @@protoc_insertion_point(serialize_to_array_start:sensor_msgs.msgs.Imu)
  // required .gazebo.msgs.Quaternion orientation = 1;
  if (has_orientation()) {
    target = ::google::protobuf::internal::WireFormatLite::
      WriteMessageNoVirtualToArray(
        1, this->orientation(), target);
  }

  // repeated float orientation_covariance = 2 [packed = true];
  if (this->orientation_covariance_size() > 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteTagToArray(
      2,
      ::google::protobuf::internal::WireFormatLite::WIRETYPE_LENGTH_DELIMITED,
      target);
    target = ::google::protobuf::io::CodedOutputStream::WriteVarint32ToArray(
      _orientation_covariance_cached_byte_size_, target);
  }
  for (int i = 0; i < this->orientation_covariance_size(); i++) {
    target = ::google::protobuf::internal::WireFormatLite::
      WriteFloatNoTagToArray(this->orientation_covariance(i), target);
  }

  // required .gazebo.msgs.Vector3d angular_velocity = 3;
  if (has_angular_velocity()) {
    target = ::google::protobuf::internal::WireFormatLite::
      WriteMessageNoVirtualToArray(
        3, this->angular_velocity(), target);
  }

  // repeated float angular_velocity_covariance = 4 [packed = true];
  if (this->angular_velocity_covariance_size() > 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteTagToArray(
      4,
      ::google::protobuf::internal::WireFormatLite::WIRETYPE_LENGTH_DELIMITED,
      target);
    target = ::google::protobuf::io::CodedOutputStream::WriteVarint32ToArray(
      _angular_velocity_covariance_cached_byte_size_, target);
  }
  for (int i = 0; i < this->angular_velocity_covariance_size(); i++) {
    target = ::google::protobuf::internal::WireFormatLite::
      WriteFloatNoTagToArray(this->angular_velocity_covariance(i), target);
  }

  // required .gazebo.msgs.Vector3d linear_acceleration = 5;
  if (has_linear_acceleration()) {
    target = ::google::protobuf::internal::WireFormatLite::
      WriteMessageNoVirtualToArray(
        5, this->linear_acceleration(), target);
  }

  // repeated float linear_acceleration_covariance = 6 [packed = true];
  if (this->linear_acceleration_covariance_size() > 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteTagToArray(
      6,
      ::google::protobuf::internal::WireFormatLite::WIRETYPE_LENGTH_DELIMITED,
      target);
    target = ::google::protobuf::io::CodedOutputStream::WriteVarint32ToArray(
      _linear_acceleration_covariance_cached_byte_size_, target);
  }
  for (int i = 0; i < this->linear_acceleration_covariance_size(); i++) {
    target = ::google::protobuf::internal::WireFormatLite::
      WriteFloatNoTagToArray(this->linear_acceleration_covariance(i), target);
  }

  if (!unknown_fields().empty()) {
    target = ::google::protobuf::internal::WireFormat::SerializeUnknownFieldsToArray(
        unknown_fields(), target);
  }
  // @@protoc_insertion_point(serialize_to_array_end:sensor_msgs.msgs.Imu)
  return target;
}

int Imu::ByteSize() const {
  int total_size = 0;

  if (_has_bits_[0 / 32] & (0xffu << (0 % 32))) {
    // required .gazebo.msgs.Quaternion orientation = 1;
    if (has_orientation()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::MessageSizeNoVirtual(
          this->orientation());
    }

    // required .gazebo.msgs.Vector3d angular_velocity = 3;
    if (has_angular_velocity()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::MessageSizeNoVirtual(
          this->angular_velocity());
    }

    // required .gazebo.msgs.Vector3d linear_acceleration = 5;
    if (has_linear_acceleration()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::MessageSizeNoVirtual(
          this->linear_acceleration());
    }

  }
  // repeated float orientation_covariance = 2 [packed = true];
  {
    int data_size = 0;
    data_size = 4 * this->orientation_covariance_size();
    if (data_size > 0) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::Int32Size(data_size);
    }
    GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
    _orientation_covariance_cached_byte_size_ = data_size;
    GOOGLE_SAFE_CONCURRENT_WRITES_END();
    total_size += data_size;
  }

  // repeated float angular_velocity_covariance = 4 [packed = true];
  {
    int data_size = 0;
    data_size = 4 * this->angular_velocity_covariance_size();
    if (data_size > 0) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::Int32Size(data_size);
    }
    GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
    _angular_velocity_covariance_cached_byte_size_ = data_size;
    GOOGLE_SAFE_CONCURRENT_WRITES_END();
    total_size += data_size;
  }

  // repeated float linear_acceleration_covariance = 6 [packed = true];
  {
    int data_size = 0;
    data_size = 4 * this->linear_acceleration_covariance_size();
    if (data_size > 0) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::Int32Size(data_size);
    }
    GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
    _linear_acceleration_covariance_cached_byte_size_ = data_size;
    GOOGLE_SAFE_CONCURRENT_WRITES_END();
    total_size += data_size;
  }

  if (!unknown_fields().empty()) {
    total_size +=
      ::google::protobuf::internal::WireFormat::ComputeUnknownFieldsSize(
        unknown_fields());
  }
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = total_size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
  return total_size;
}

void Imu::MergeFrom(const ::google::protobuf::Message& from) {
  GOOGLE_CHECK_NE(&from, this);
  const Imu* source =
    ::google::protobuf::internal::dynamic_cast_if_available<const Imu*>(
      &from);
  if (source == NULL) {
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
    MergeFrom(*source);
  }
}

void Imu::MergeFrom(const Imu& from) {
  GOOGLE_CHECK_NE(&from, this);
  orientation_covariance_.MergeFrom(from.orientation_covariance_);
  angular_velocity_covariance_.MergeFrom(from.angular_velocity_covariance_);
  linear_acceleration_covariance_.MergeFrom(from.linear_acceleration_covariance_);
  if (from._has_bits_[0 / 32] & (0xffu << (0 % 32))) {
    if (from.has_orientation()) {
      mutable_orientation()->::gazebo::msgs::Quaternion::MergeFrom(from.orientation());
    }
    if (from.has_angular_velocity()) {
      mutable_angular_velocity()->::gazebo::msgs::Vector3d::MergeFrom(from.angular_velocity());
    }
    if (from.has_linear_acceleration()) {
      mutable_linear_acceleration()->::gazebo::msgs::Vector3d::MergeFrom(from.linear_acceleration());
    }
  }
  mutable_unknown_fields()->MergeFrom(from.unknown_fields());
}

void Imu::CopyFrom(const ::google::protobuf::Message& from) {
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void Imu::CopyFrom(const Imu& from) {
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool Imu::IsInitialized() const {
  if ((_has_bits_[0] & 0x00000015) != 0x00000015) return false;

  if (has_orientation()) {
    if (!this->orientation().IsInitialized()) return false;
  }
  if (has_angular_velocity()) {
    if (!this->angular_velocity().IsInitialized()) return false;
  }
  if (has_linear_acceleration()) {
    if (!this->linear_acceleration().IsInitialized()) return false;
  }
  return true;
}

void Imu::Swap(Imu* other) {
  if (other != this) {
    std::swap(orientation_, other->orientation_);
    orientation_covariance_.Swap(&other->orientation_covariance_);
    std::swap(angular_velocity_, other->angular_velocity_);
    angular_velocity_covariance_.Swap(&other->angular_velocity_covariance_);
    std::swap(linear_acceleration_, other->linear_acceleration_);
    linear_acceleration_covariance_.Swap(&other->linear_acceleration_covariance_);
    std::swap(_has_bits_[0], other->_has_bits_[0]);
    _unknown_fields_.Swap(&other->_unknown_fields_);
    std::swap(_cached_size_, other->_cached_size_);
  }
}

::google::protobuf::Metadata Imu::GetMetadata() const {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::Metadata metadata;
  metadata.descriptor = Imu_descriptor_;
  metadata.reflection = Imu_reflection_;
  return metadata;
}


// @@protoc_insertion_point(namespace_scope)

}  // namespace msgs
}  // namespace sensor_msgs

// @@protoc_insertion_point(global_scope)
