cmake_minimum_required(VERSION 3.8)
project(ROS2_YOLOv7)
set(PROJECT_NAME ROS2_YOLOv7)
EXECUTE_PROCESS( COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCH )
message( STATUS "Architecture: ${ARCH}" )

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++11
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 11)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

################## CONDA STUFF Begins here ##################
# Ignore CONDA
if(DEFINED ENV{CONDA_PREFIX})
  set(h5_ignore_path
    $ENV{CONDA_PREFIX}/bin $ENV{CONDA_PREFIX}/lib $ENV{CONDA_PREFIX}/include
    $ENV{CONDA_PREFIX}/Library/bin $ENV{CONDA_PREFIX}/Library/lib $ENV{CONDA_PREFIX}/Library/include
  )
  list(APPEND CMAKE_IGNORE_PATH ${h5_ignore_path})
endif()
################## CONDA STUFF ENDS here ##################

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(CUDA REQUIRED)

################## CUDA STUFF Begins here ##################
set(CUDA_VERSION 11.6)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-${CUDA_VERSION})

SET(CMAKE_BUILD_TYPE "Release")
add_compile_options(-W)
add_compile_options(-std=c++11)

set(SMS 50 52 53 60 61 62 70 72 75 80 86)
foreach(sm ${SMS})
    set(GENCODE ${GENCODE} -gencode arch=compute_${sm},code=sm_${sm})
endforeach()
list(GET SMS -1 LATEST_SM)
set(GENCODE "${GENCODE} -gencode arch=compute_${LATEST_SM},code=compute_${LATEST_SM}")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
    -ccbin ${CMAKE_CXX_COMPILER}
    -Xcompiler -DWIN_INTERFACE_CUSTOM
    -Xcompiler -I/usr/${ARCH}-linux-gnu/include/
    -Xlinker -lsocket
    -Xlinker -rpath=/usr/lib/${ARCH}-linux-gnu/
    -Xlinker -rpath=/usr/${ARCH}-linux-gnu/lib/
    -Xlinker -L/usr/lib/${ARCH}-linux-gnu/
    -Xlinker -L/usr/${ARCH}-linux-gnu/lib/
)

set(TENSORRT_INCLUDE_DIRS /usr/include/${ARCH}-linux-gnu/)
set(TENSORRT_LIBRARY_DIRS /usr/lib/${ARCH}-linux-gnu/)

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIRS}
    include
    ${rclcpp_INCLUDE_DIRS}
)

link_directories(
    ${TENSORRT_LIBRARY_DIRS}
    /usr/lib/${ARCH}-linux-gnu
    /usr/${ARCH}-linux-gnu/lib/
)

file(GLOB_RECURSE SOURCE_FILES
    src/tools.cpp
    src/Yolov7.cpp
)


cuda_add_executable(${PROJECT_NAME} src/image_inference.cpp ${SOURCE_FILES})
ament_target_dependencies(${PROJECT_NAME} rclcpp sensor_msgs cv_bridge CUDA)

target_link_libraries(${PROJECT_NAME}
    libnvinfer.so
    libnvonnxparser.so
    libnvinfer_plugin.so
)

install(TARGETS
	${PROJECT_NAME}
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()