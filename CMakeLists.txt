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

# Provide Eigen3 ourselves so ITK skips its bundled ITKInternalEigen3 build.
# ITK's itkExternal_Eigen3.cmake otherwise runs a nested
# `execute_process(COMMAND ${CMAKE_COMMAND} ...)` to configure Eigen as an
# inner CMake project — that inner CMakeCache.txt caches the parent's ninja
# path, and when uv tears down its build env between syncs the cached path
# dangles and breaks incremental rebuilds. Setting ITK_USE_SYSTEM_EIGEN=ON
# routes ITK through `find_package(Eigen3)`, which (via OVERRIDE_FIND_PACKAGE)
# resolves to our FetchContent target with no inner configure.
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
# Pin to 3.4.0 — bumping to 3.4.1 requires `EIGEN_BUILD_CMAKE_PACKAGE=ON`
# (3.4.1 added an unconditional `install(TARGETS eigen EXPORT Eigen3Targets)`
# while leaving the matching `install(EXPORT)` behind that flag, which trips
# CMake's `install(EXPORT ITKTargets)` validator since ITKCommon's link
# interface now references the bare `eigen` target). With that flag flipped,
# CMake 4.2.x crashes with a glibc heap corruption (`corrupted size vs.
# prev_size`) immediately after Eigen's CMakeLists finishes on manylinux —
# reproducible across Python 3.11–3.14. macOS doesn't trip it, but the wheel
# matrix is unrunnable. Until CMake or Eigen ships a fix, 3.4.0 is the
# stable choice; we get nothing functional from 3.4.1 anyway.
FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    OVERRIDE_FIND_PACKAGE
)
set(ITK_USE_SYSTEM_EIGEN ON CACHE BOOL "" FORCE)
# OVERRIDE_FIND_PACKAGE bypasses Config-file processing, so Eigen3_VERSION
# isn't populated by find_package(Eigen3). ITK's ThirdParty/Eigen3 module
# enforces Eigen3_VERSION >= 3.3 — set it explicitly to match GIT_TAG above.
set(Eigen3_VERSION 3.4.0 CACHE STRING "" FORCE)

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
# ITK is statically linked into warpkit_cpp and we never invoke its install
# rules (see COMPONENT-restricted install at the bottom of this file plus
# `install.components` in pyproject.toml). The values below only exist to keep
# ITK's zlib-ng target's INSTALL_INTERFACE include path out of the project
# source tree — CMake 4.x rejects an INSTALL_INTERFACE include that resolves
# under ${CMAKE_SOURCE_DIR}, which would happen if we used ${CMAKE_BINARY_DIR}
# (scikit-build-core nests build/ under the project root).
string(MD5 _warpkit_build_hash "${CMAKE_BINARY_DIR}")
if(WIN32)
    set(_warpkit_itk_scratch "$ENV{TEMP}/warpkit-itk-${_warpkit_build_hash}")
else()
    set(_warpkit_itk_scratch "/tmp/warpkit-itk-${_warpkit_build_hash}")
endif()
set(ITK_INSTALL_LIBRARY_DIR ${_warpkit_itk_scratch}/lib CACHE STRING "" FORCE)
set(ITK_INSTALL_ARCHIVE_DIR ${_warpkit_itk_scratch}/lib CACHE STRING "" FORCE)
set(ITK_INSTALL_INCLUDE_DIR ${_warpkit_itk_scratch}/include CACHE STRING "" FORCE)
set(ITK_INSTALL_PACKAGE_DIR ${_warpkit_itk_scratch}/package CACHE STRING "" FORCE)
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

# Install the extension into the warpkit/ Python package within the wheel.
# scikit-build-core sets CMAKE_INSTALL_PREFIX to the wheel root, so this lands
# at <wheel>/warpkit/warpkit_cpp.so. The COMPONENT tag pairs with
# `install.components = ["extension"]` in pyproject.toml so cmake --install
# only copies our .so — ITK's install rules are skipped entirely.
install(
    TARGETS warpkit_cpp
    LIBRARY DESTINATION warpkit
    COMPONENT extension
)
