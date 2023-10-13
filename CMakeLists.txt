# set min version and policy
cmake_minimum_required(VERSION 3.24)
cmake_policy(VERSION 3.24)

# set project name and languages
project(warpkit_cpp LANGUAGES CXX)

# set C++ standard
set(CMAKE_CXX_STANDARD 17)

# set option/variable policy
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0148 NEW)

# load auxillary cmake modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake_aux")

# find julia headers
find_package(Julia REQUIRED)

# ensure ROMEO dependency installed during compile time
# there is an additional check in setup.py for pre-built python packages.
execute_process(COMMAND ${Julia_EXECUTABLE} "-e" "using Pkg; !in(\"ROMEO\",\
 [dep.name for (uuid, dep) in Pkg.dependencies()]) ? \
 Pkg.add(Pkg.PackageSpec(;name=\"ROMEO\", version=\"1.0.0\")) : nothing")

# include FetchContent module
include(FetchContent)

# download and include pybind11
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG v2.11.1
    OVERRIDE_FIND_PACKAGE
)
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
    GIT_TAG v5.4rc01
)
# itk with fetchcontent is a bit weird, OVERRIDE_FIND_PACKAGE does not work
# so we have to explicitly build itk with make available
# and then set the itk dir to the build location so find_package can find it
FetchContent_MakeAvailable(ITK)
FetchContent_GetProperties(ITK)
set(ITK_DIR ${itk_BINARY_DIR})
find_package(ITK CONFIG REQUIRED COMPONENTS ITKCommon ITKImageCompose ITKDisplacementField ITKDistanceMap)

# display all warnings
add_compile_options(-Wall -O3 -march=x86-64-v3)

# compile pybind11 interface
pybind11_add_module(warpkit_cpp MODULE ${CMAKE_SOURCE_DIR}/src/warpkit.cpp)
target_include_directories(warpkit_cpp
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PRIVATE ${ITK_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include/itk
)
target_link_libraries(warpkit_cpp PRIVATE Julia ITKCommon)

# install target
install(
    TARGETS warpkit_cpp
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}
)
