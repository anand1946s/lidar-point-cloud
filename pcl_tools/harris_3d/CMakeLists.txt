cmake_minimum_required(VERSION 3.16)

project(harris3d)

find_package(PCL REQUIRED)

add_executable(harris3d main.cpp)

target_link_libraries(harris3d ${PCL_LIBRARIES})

target_include_directories(harris3d PRIVATE ${PCL_INCLUDE_DIRS})

add_definitions(${PCL_DEFINITIONS})