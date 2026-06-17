cmake_minimum_required(VERSION 3.16)

project(sift3d)

find_package(PCL REQUIRED COMPONENTS common io features keypoints search)

add_executable(sift3d sift.cpp)

target_link_libraries(sift3d ${PCL_LIBRARIES})
target_include_directories(sift3d PRIVATE ${PCL_INCLUDE_DIRS})
target_compile_features(sift3d PRIVATE cxx_std_14)

add_definitions(${PCL_DEFINITIONS})
