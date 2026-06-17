cmake_minimum_required(VERSION 3.16)

project(shot)

find_package(PCL REQUIRED COMPONENTS common io filters features search)

add_executable(shot main.cpp)

target_link_libraries(shot ${PCL_LIBRARIES})

target_include_directories(shot PRIVATE ${PCL_INCLUDE_DIRS})

add_definitions(${PCL_DEFINITIONS})