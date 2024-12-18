cmake_minimum_required(VERSION 2.8)
project(ndtRANSAC)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()


find_package(OpenCV 3.2 QUIET)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()
find_package(Eigen3 3.1.0 REQUIRED)
find_package(PCL 1.8 REQUIRED)
find_package(nlohmann_json 3.2.0 REQUIRED)
find_package(OpenNI REQUIRED)
find_package(OpenNI2 REQUIRED)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${PCL_INCLUDE_DIRS}
${EIGEN3_INCLUDE_DIR}
${OpenCV_INCLUDE_DIRS}
)

set(SETTING_READER "src/SettingReader.cpp" "include/SettingReader.h")
set(VISUALIZER "src/visualizer.cpp" "include/visualizer.h")


# Build examples

# TUM RGBD
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/TUM)
FILE(GLOB SC_FILES "*.cpp" "*.h")
add_executable(ndt_tum.bin Examples/TUM/tum_prior.cpp ${SC_FILES} ${SETTING_READER} ${VISUALIZER} include/cmdline.h)
target_link_libraries(ndt_tum.bin ${PCL_LIBRARIES} ${OpenCV_LIBS} ${EIGEN3_INCLUDE_DIR})

# NYU V2 labeled
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/NYUV2)
FILE(GLOB SC_FILES "*.cpp" "*.h")
add_executable(ndt_nyu.bin Examples/NYUV2/ndt_nyu.cpp ${SC_FILES} ${SETTING_READER} ${VISUALIZER} include/cmdline.h)
target_link_libraries(ndt_nyu.bin ${PCL_LIBRARIES} ${OpenCV_LIBS} ${EIGEN3_INCLUDE_DIR})

# Stanfort 2D-3D-S
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Stanfort)
FILE(GLOB SC_FILES "*.cpp" "*.h")
add_executable(ndt_stanfort.bin Examples/Stanfort/ndt_stanfort.cpp ${SC_FILES} ${SETTING_READER} ${VISUALIZER} include/cmdline.h)
target_link_libraries(ndt_stanfort.bin ${PCL_LIBRARIES} ${OpenCV_LIBS} ${EIGEN3_INCLUDE_DIR})

#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/test)
#FILE(GLOB SC_FILES "*.cpp" "*.h")
#add_executable(test.bin Examples/test/main.cpp ${SC_FILES} ${SETTING_READER} ${VISUALIZER} include/cmdline.h)
#target_link_libraries(test.bin ${PCL_LIBRARIES} ${OpenCV_LIBS} ${EIGEN3_INCLUDE_DIR})
