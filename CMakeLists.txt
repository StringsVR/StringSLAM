# ------------ Variables ------------
set(PROJECT_NAME StringSLAM)
set(PROJECT_VERSION 0.01)
set(DESCRIPTION "A SLAM Library meant for low powered hardware, built with customization in mind!")


# ------------ Project Definition ------------
cmake_minimum_required(VERSION 3.16)
project(${PROJECT_NAME}
    VERSION ${PROJECT_VERSION}
    DESCRIPTION ${DESCRIPTION}
    LANGUAGES CXX
)

# ------------ Compiler Settings ------------
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt)
    if (NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
        message(STATUS "No previous build - default to Release build")
    endif()
endif()

add_compile_options(-Wall -Wextra -pedantic -Wno-unused-parameter -faligned-new -Werror -Wfatal-errors -Wno-c11-extensions)
add_definitions(-D_FILE_OFFSET_BITS=64)

if (CMAKE_COMPILER_IS_GNUCXX)
    add_compile_options(-Wno-psabi)
endif()

option(BUILD_SHARED_LIBS "Build using shared libraries" ON)

IF (NOT ENABLE_COMPILE_FLAGS_FOR_TARGET)
    # On a Pi this will give us armhf or arm64.
    execute_process(COMMAND dpkg-architecture -qDEB_HOST_ARCH
        OUTPUT_VARIABLE ENABLE_COMPILE_FLAGS_FOR_TARGET OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
message(STATUS "Platform: ${ENABLE_COMPILE_FLAGS_FOR_TARGET}")
if ("${ENABLE_COMPILE_FLAGS_FOR_TARGET}" STREQUAL "arm64")
    add_definitions(-ftree-vectorize)
elseif ("${ENABLE_COMPILE_FLAGS_FOR_TARGET}" STREQUAL "armv8-neon")
    add_definitions(-mfpu=neon-fp-armv8 -ftree-vectorize)
endif()

# Source package generation setup.
set(CPACK_GENERATOR "TXZ")
set(CPACK_PACKAGE_FILE_NAME "${PROJECT_NAME}-build")
set(CPACK_SOURCE_GENERATOR "TXZ")
set(CPACK_INSTALL_SCRIPTS ${CMAKE_SOURCE_DIR}/package.cmake)
set(CPACK_SOURCE_PACKAGE_FILE_NAME "${PROJECT_NAME}-src")
set(CPACK_SOURCE_IGNORE_FILES "/\.git*;/build;")
include(CPack)

# ------------ Sources / Headers ------------
file(GLOB_RECURSE SOURCES
    "${PROJECT_SOURCE_DIR}/src/*.cpp"
)

file(GLOB_RECURSE HEADERS
    "${PROJECT_SOURCE_DIR}/include/*.hpp"
)

# ------------ Dependencies ------------
find_package(PkgConfig REQUIRED)

# Look For OpenCV and Eigen
pkg_check_modules(OPENCV REQUIRED opencv4)
pkg_check_modules(EIGEN3 REQUIRED eigen3)
find_package(OpenEXR QUIET)
include(GNUInstallDirs)

# ------------ Build Library ------------
add_library(${PROJECT_NAME} ${SOURCES})
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" IMPORT_PREFIX "")

target_include_directories(${PROJECT_NAME}
    PUBLIC
        include
        ${OPENCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
    pthread
    ${OPENCV_LIBRARIES}
    ${EIGEN3_LIBRARIES}
)

# ------------ DOxygen ------------
find_package(Doxygen REQUIRED)

set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile)
set(DOXYGEN_OUT ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile)

configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

add_custom_target(docs
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating API documentation with Doxygen"
    VERBATIM
)