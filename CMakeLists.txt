# set min version and policy
cmake_minimum_required(VERSION 3.24)
cmake_policy(VERSION 3.24)

# set project name and languages
project(warpkit_cpp LANGUAGES CXX)

# set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set option/variable policy
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0148 NEW)

# Link-time optimization (-flto). Portable across GCC/Clang/MSVC via the
# CheckIPOSupported helper, which also adds the matching link flag.
include(CheckIPOSupported)
check_ipo_supported(RESULT _ipo_supported OUTPUT _ipo_error)
if(_ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    message(STATUS "IPO/LTO enabled")
else()
    message(STATUS "IPO/LTO not supported by this toolchain: ${_ipo_error}")
endif()

# include FetchContent module
include(FetchContent)

# download and include pybind11
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG v3.0.4
    OVERRIDE_FIND_PACKAGE
)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# download, configure, and include itk
set(ZLIBNG_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(ITKGroup_Core OFF CACHE BOOL "" FORCE)
set(Module_ITKCommon ON CACHE BOOL "" FORCE)
set(Module_ITKImageCompose ON CACHE BOOL "" FORCE)
set(Module_ITKDisplacementField ON CACHE BOOL "" FORCE)
set(Module_ITKDistanceMap ON CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(BUILD_STATIC_LIBS ON CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(ITK_BUILD_DEFAULT_MODULES OFF CACHE BOOL "" FORCE)
set(ITK_WRAP_PYTHON OFF CACHE BOOL "" FORCE)
set(WITH_GTEST OFF CACHE BOOL "" FORCE)
set(ITK_INSTALL_LIBRARY_DIR ${PYTHON_INSTALL_TMPDIR}/lib CACHE STRING "" FORCE)
set(ITK_INSTALL_ARCHIVE_DIR ${PYTHON_INSTALL_TMPDIR}/lib CACHE STRING "" FORCE)
set(ITK_INSTALL_INCLUDE_DIR ${PYTHON_INSTALL_TMPDIR}/include CACHE STRING "" FORCE)
set(ITK_INSTALL_PACKAGE_DIR ${PYTHON_INSTALL_TMPDIR}/package CACHE STRING "" FORCE)
FetchContent_Declare(
    ITK
    GIT_REPOSITORY https://github.com/InsightSoftwareConsortium/ITK.git
    GIT_TAG v5.4.5
)

# itk with fetchcontent is a bit weird, OVERRIDE_FIND_PACKAGE does not work
# but FetchContent_MakeAvailable with ITK_DIR does?
FetchContent_MakeAvailable(ITK)
set(ITK_DIR ${itk_BINARY_DIR})
find_package(ITK CONFIG REQUIRED)

# create an ITKLIB target to link against
add_library(ITKLIB INTERFACE IMPORTED)
set_target_properties(ITKLIB PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${ITK_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES "${ITK_LIBRARIES}"
)

# add compile options based on architecture.
# x86_64 targets the v2 microarch level (Nehalem 2008 — SSE3/SSE4.1/SSE4.2/
# popcnt). v2 features are universal in anything that can install a current
# Linux distro, and it matches the "above baseline" stance taken on aarch64.
# Consumers needing to override (older CPU, or a higher microarch level) can
# pass CXXFLAGS at build time.
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    add_compile_options(-O3 -mcpu=apple-m1)
    message(STATUS "Using -mcpu=apple-m1")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    add_compile_options(-O3 -march=armv8.2-a)
    message(STATUS "Using -march=armv8.2-a")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    add_compile_options(-O3 -march=x86-64-v2)
    message(STATUS "Using -march=x86-64-v2")
endif()

# compile pybind11 interface
pybind11_add_module(warpkit_cpp MODULE ${CMAKE_SOURCE_DIR}/src/warpkit.cpp)
target_include_directories(warpkit_cpp
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PRIVATE ${PROJECT_SOURCE_DIR}/include/itk
)
target_compile_options(warpkit_cpp PRIVATE -Wall)
target_link_libraries(warpkit_cpp PRIVATE ITKLIB)

# install target
install(
    TARGETS warpkit_cpp
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}
)
