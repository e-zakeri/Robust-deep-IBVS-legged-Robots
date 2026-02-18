cmake_minimum_required(VERSION 3.8)
project(go2_pkg)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/InterestPoints.msg"
  DEPENDENCIES std_msgs
)

ament_package()
