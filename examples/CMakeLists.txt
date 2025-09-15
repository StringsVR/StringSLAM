cmake_minimum_required(VERSION 3.16)
project(StringSLAM_Examples)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
elseif(MSVC)
  add_compile_options(/W4)
endif()

# -------- Dependencies --------
find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV REQUIRED opencv4)
pkg_check_modules(EIGEN3 REQUIRED eigen3)
find_package(OpenEXR QUIET)

include(GNUInstallDirs)

# -------- Sources --------
file(GLOB_RECURSE EXAMPLE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

# -------- Include dirs --------
include_directories(
    ${CMAKE_SOURCE_DIR}/../include   # assuming examples/ is sibling to main project
    ${OPENCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
)

# -------- Create executables --------
foreach(example_file ${EXAMPLE_SOURCES})
    get_filename_component(example_name ${example_file} NAME_WE)
    add_executable(${example_name} ${example_file})

    target_link_libraries(${example_name} 
        PRIVATE 
            "${CMAKE_SOURCE_DIR}/../build/StringSLAM.so"
            ${OPENCV_LIBRARIES}   # OpenCV
            ${EIGEN3_LIBRARIES}   # Eigen
            pthread
    )
endforeach()