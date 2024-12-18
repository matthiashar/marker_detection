cmake_minimum_required(VERSION 3.9)

project(marker_detection LANGUAGES CXX)

# Options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV
find_package(OpenCV REQUIRED)

# Source files
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set (project_SOURCES
    ${SRC_DIR}/markerdetection.cpp
    ${SRC_DIR}/codes.cpp)

# Create library
add_library(marker_detection_lib ${project_SOURCES})
target_link_libraries(marker_detection_lib ${OpenCV_LIBS})
target_include_directories(marker_detection_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Create example executable 
add_executable(marker_detection ${SRC_DIR}/main.cpp)
target_link_libraries(marker_detection marker_detection_lib)

# Test
enable_testing()
set(TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests)
add_executable(test_marker_detection ${TEST_DIR}/test.cpp)
add_test(marker_detection test_marker_detection)
target_link_libraries(test_marker_detection marker_detection_lib)
