# 3.28 floor: FetchContent_Declare(EXCLUDE_FROM_ALL) — keeps the
# subprojects' targets and install rules out of the build unless linked.
cmake_minimum_required(VERSION 3.28)

project(warpkit_cpp LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG v3.0.4
    GIT_SHALLOW TRUE
    EXCLUDE_FROM_ALL
)

# Supply Eigen3 to ITK ourselves (ITK_USE_SYSTEM_EIGEN=ON) so ITK does not build
# its bundled ITKInternalEigen3. ITK's internal-Eigen path runs a nested CMake
# configure whose cache pins the ninja binary from uv's per-build environment;
# uv deletes that environment after each sync, so the pinned path goes stale and
# the next `uv sync --reinstall-package warpkit` fails to configure.
#
# ITK's Eigen3 module calls find_package(Eigen3 CONFIG) and then reads the
# Eigen3::Eigen target. OVERRIDE_FIND_PACKAGE makes find_package succeed via a
# generated stub but that stub defines no target, so we fetch Eigen's source and,
# since Eigen is header-only, define Eigen3::Eigen ourselves as an INTERFACE
# library over the headers. The tag is the Eigen snapshot ITK 6.0b02 bundles
# (3.4.90, from ITK's fork); ITK requires Eigen >= 3.3.
FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://github.com/InsightSoftwareConsortium/eigen.git
    GIT_TAG for/itk-20250707-master-6854da2e
    GIT_SHALLOW TRUE
    EXCLUDE_FROM_ALL
    OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(Eigen3)
if(NOT TARGET Eigen3::Eigen)
    add_library(warpkit_eigen INTERFACE)
    add_library(Eigen3::Eigen ALIAS warpkit_eigen)
    target_include_directories(warpkit_eigen INTERFACE
        $<BUILD_INTERFACE:${eigen3_SOURCE_DIR}>)
endif()
set(ITK_USE_SYSTEM_EIGEN ON CACHE BOOL "" FORCE)
# The stub config does not set Eigen3_VERSION; set it so ITK's >= 3.3 check passes.
set(Eigen3_VERSION 3.4.90 CACHE STRING "" FORCE)

# ITK 6 configuration.
set(ITK_BUILD_DEFAULT_MODULES OFF)
# Module_*/ITKGroup_* must stay CACHE entries (not plain vars): ITK's
# ITKModuleEnablement.cmake does set_property(CACHE Module_X ...), which
# errors if CMP0077 turned ITK's option() into a no-op and no cache
# variable was ever created.
set(ITKGroup_Core OFF CACHE BOOL "" FORCE)
set(Module_ITKCommon ON CACHE BOOL "" FORCE)
set(Module_ITKImageCompose ON CACHE BOOL "" FORCE)
set(Module_ITKDisplacementField ON CACHE BOOL "" FORCE)
set(Module_ITKDistanceMap ON CACHE BOOL "" FORCE)
set(ITK_WRAP_PYTHON OFF)
set(BUILD_SHARED_LIBS OFF)
set(BUILD_TESTING OFF)
set(BUILD_EXAMPLES OFF)
set(ZLIBNG_ENABLE_TESTS OFF)
FetchContent_Declare(
    ITK
    GIT_REPOSITORY https://github.com/InsightSoftwareConsortium/ITK.git
    # GIT_SHALLOW skips ITK's ~1.5 GB history on every cold build (local and
    # each CI wheel job). Requires GIT_TAG to be a tag or branch, not a SHA.
    GIT_TAG v6.0b02
    GIT_SHALLOW TRUE
    EXCLUDE_FROM_ALL
)

FetchContent_MakeAvailable(pybind11 ITK)

# ITKEigen3Module links Eigen3::Eigen and belongs to ITK's ITKTargets export
# set, so CMake requires our Eigen3::Eigen target to be in an export set too.
# Register it with ITKTargets to satisfy that; nothing is actually installed
# from it (only the "extension" component ships — see pyproject install.components).
if(TARGET warpkit_eigen)
    install(TARGETS warpkit_eigen EXPORT ITKTargets)
endif()

# Arch baseline for our extension only (ITK targets keep their own flags).
# x86_64 targets the v2 microarch level (Nehalem 2008 — SSE3/SSE4.1/SSE4.2/
# popcnt). v2 features are universal in anything that can install a current
# Linux distro, and it matches the "above baseline" stance taken on aarch64.
# Consumers needing to override (older CPU, or a higher microarch level) can
# pass CXXFLAGS at build time.
if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    set(_warpkit_arch_flags -mcpu=apple-m1)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(_warpkit_arch_flags -march=armv8.2-a)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(_warpkit_arch_flags -march=x86-64-v2)
else()
    set(_warpkit_arch_flags "")
endif()
message(STATUS "warpkit arch flags for ${CMAKE_SYSTEM_PROCESSOR}: ${_warpkit_arch_flags}")

pybind11_add_module(warpkit_cpp MODULE src/warpkit.cpp)
target_compile_features(warpkit_cpp PRIVATE cxx_std_20)
target_include_directories(warpkit_cpp PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/itk
)
target_compile_options(warpkit_cpp PRIVATE -Wall -O3 ${_warpkit_arch_flags})
target_link_libraries(warpkit_cpp PRIVATE
    ITK::ITKCommonModule
    ITK::ITKImageComposeModule
    ITK::ITKDisplacementFieldModule
    ITK::ITKDistanceMapModule
)

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
