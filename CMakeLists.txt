cmake_minimum_required(VERSION 3.8)
project(ROS2_YOLOv7)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

################## CUDA STUFF Begins here ##################
enable_language( CUDA )

find_package(CUDA)
set( CMAKE_C_STANDARD 99 )
set( CMAKE_CXX_STANDARD 11 )
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -fpic -fpie -fpermissive  -std=c++11 -pthread" )
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})
find_package(jsoncpp CONFIG REQUIRED)

################## CUDA STUFF ENDS here ##################

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
# find_package(rmw_qos_profiles REQUIRED)
find_package(cv_bridge REQUIRED)

################## CUDA STUFF Begins here ##################
# global include_directories
include_directories( /usr/local/cuda/include )
#add judgement about system:

MESSAGE(STATUS "CMAKE_HOST_SYSTEM_PROCESSOR is ${CMAKE_HOST_SYSTEM_PROCESSOR}")
if (${CMAKE_HOST_SYSTEM_PROCESSOR} EQUAL aarch64)
    include_directories( /usr/include/aarch64-linux-gnu/ ) # for jetson
elseif(${CMAKE_HOST_SYSTEM_PROCESSOR} EQUAL x86_64)
    include_directories( /usr/lib/x86_64-linux-gnu/ )
endif()

include_directories( "${CMAKE_SOURCE_DIR}/src/" )
include_directories( "/usr/include/jsoncpp/")
# global definitions
add_definitions( -w)

# global library path
if (${CMAKE_HOST_SYSTEM_PROCESSOR} EQUAL aarch64)
    link_directories( "/usr/lib/aarch64-linux-gnu/" )
elseif(${CMAKE_HOST_SYSTEM_PROCESSOR} EQUAL x86_64)
    link_directories( "/usr/lib/x86_64-linux-gnu/" )
endif()

link_directories( "/usr/lib/" )
link_directories( "/usr/local/lib/")
link_directories( "/usr/local/cuda/lib64/" )

FILE(GLOB_RECURSE YOLO_SRC src/Yolov7.cpp )
add_library( yolo SHARED ${YOLO_SRC} )
target_link_libraries(yolo PRIVATE nvinfer)
target_link_libraries(yolo PRIVATE nvinfer_plugin)
target_link_libraries(yolo PRIVATE nvparsers)
target_link_libraries(yolo PRIVATE nvonnxparser cudart ${OpenCV_LIBS})
################## CUDA STUFF ENDS here ##################

add_executable(camera_inference_node src/image_inference.cpp)
ament_target_dependencies(camera_inference_node rclcpp std_msgs)

install(TARGETS
  camera_inference_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()
