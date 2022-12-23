################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################
cmake_minimum_required( VERSION 3.8 )

project(ros2_yolov7)

# Default to C++17.
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -fpic -fpie -fpermissive  -std=c++11 -pthread" )

# Ignore CONDA
if(DEFINED ENV{CONDA_PREFIX})
  set(h5_ignore_path
    $ENV{CONDA_PREFIX}/bin $ENV{CONDA_PREFIX}/lib $ENV{CONDA_PREFIX}/include
    $ENV{CONDA_PREFIX}/Library/bin $ENV{CONDA_PREFIX}/Library/lib $ENV{CONDA_PREFIX}/Library/include
  )
  list(APPEND CMAKE_IGNORE_PATH ${h5_ignore_path})
endif()

find_package(ament_cmake_auto REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
enable_language( CUDA )

include_directories( ${OpenCV_INCLUDE_DIRS})

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

ament_auto_find_build_dependencies(REQUIRED
  ${${PROJECT_NAME}_BUILD_DEPENDS}
  ${${PROJECT_NAME}_BUILDTOOL_DEPENDS}
)

SET(${PROJECT_NAME}_LIB_HEADER
  include/${PROJECT_NAME}/tools.h
  include/${PROJECT_NAME}/Yolov7.h
)

SET(${PROJECT_NAME}_LIB_SRC
  src/Yolov7.cpp
)


# Export the library using ament_auto_cmake
ament_auto_add_library(${PROJECT_NAME} SHARED 
  ${${PROJECT_NAME}_LIB_SRC}
  ${${PROJECT_NAME}_LIB_HEADER}
)

# Link the library to the required dependencies
target_link_libraries(${PROJECT_NAME} nvinfer)
target_link_libraries(${PROJECT_NAME} nvinfer_plugin)
target_link_libraries(${PROJECT_NAME} nvparsers)
target_link_libraries(${PROJECT_NAME} nvonnxparser cudart ${OpenCV_LIBS})

ament_auto_add_executable(${PROJECT_NAME}_node_exe
  src/image_inference_node.cpp
)

target_link_libraries(${PROJECT_NAME}_node_exe ${PROJECT_NAME} cudart ${OpenCV_LIBS})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package(INSTALL_TO_SHARE
#   engines
#   param
#   launch
)

