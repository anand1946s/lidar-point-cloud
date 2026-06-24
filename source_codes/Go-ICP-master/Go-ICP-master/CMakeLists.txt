project(GoICP)

cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(CMAKE_BUILD_TYPE Release)
if(NOT MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -ffast-math")
endif()


add_executable(GoICP
	jly_main.cpp
	jly_goicp.cpp
	jly_3ddt.cpp
	matrix.cpp
	ConfigMap.cpp
	StringTokenizer.cpp
	)
