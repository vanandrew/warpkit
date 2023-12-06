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

# find julia headers and library
if(DEFINED CIBUILDWHEEL)
    # manually define Julia for CIBUILDWHEEL since using julia will cause it to crash
    # get the version string
    execute_process(
        COMMAND /opt/julia/bin/julia --startup-file=no --version
        OUTPUT_VARIABLE Julia_VERSION_STRING
    )
    string(
        REGEX REPLACE ".*([0-9]+\\.[0-9]+\\.[0-9]+).*" "\\1"
        Julia_VERSION_STRING "${Julia_VERSION_STRING}"
    )

    # get the include directory
    set(Julia_INCLUDE_DIRS /opt/julia/include/julia)

    # get the library
    set(Julia_LIBRARY /opt/julia/lib/libjulia.so.1)

    # print out status
    message(STATUS "Using Julia ${Julia_VERSION_STRING}")
    message(STATUS "Julia_INCLUDE_DIRS: ${Julia_INCLUDE_DIRS}")
    message(STATUS "Julia_LIBRARY: ${Julia_LIBRARY}")

    # make importable target
    add_library(Julia INTERFACE IMPORTED)
    set_target_properties(Julia PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${Julia_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${Julia_LIBRARY}"
    )
else()
    find_package(Julia REQUIRED)
endif()

# ensure ROMEO dependency installed during compile time
# there is an additional check in setup.py for pre-built python packages.
# only do if not CIBUILDWHEEL since it will crash
if (NOT DEFINED CIBUILDWHEEL)
    execute_process(COMMAND ${Julia_EXECUTABLE} "-e" "using Pkg; !in(\"ROMEO\",\
    [dep.name for (uuid, dep) in Pkg.dependencies()]) ? \
    Pkg.add(Pkg.PackageSpec(;name=\"ROMEO\", version=\"1.0.0\")) : nothing")
endif()

# include FetchContent module
include(FetchContent)

# download and include pybind11
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG v2.11.1
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
    GIT_TAG v5.4rc01
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

# add compile options based on architecture
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    # we need minimum of gcc 11 for -march=x86-64-v3
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11)
            add_compile_options(-Wall -O3 -march=x86-64)
            message(STATUS "Using -march=x86-64")
        else()
            add_compile_options(-Wall -O3 -march=x86-64-v3)
            message(STATUS "Using -march=x86-64-v3")
        endif()

    # else if clang
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 12)
            add_compile_options(-Wall -O3 -march=x86-64)
            message(STATUS "Using -march=x86-64")
        else()
            add_compile_options(-Wall -O3 -march=x86-64-v3)
            message(STATUS "Using -march=x86-64-v3")
        endif()
    endif()

# else if arm
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    # just try native, it usually works
    add_compile_options(-Wall -O3 -march=armv8.4-a)
    message(STATUS "Using -march=armv8.4-a")
endif()

# compile pybind11 interface
pybind11_add_module(warpkit_cpp MODULE ${CMAKE_SOURCE_DIR}/src/warpkit.cpp)
target_include_directories(warpkit_cpp
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PRIVATE ${PROJECT_SOURCE_DIR}/include/itk
)
target_link_libraries(warpkit_cpp PRIVATE Julia ITKLIB)

# install target
install(
    TARGETS warpkit_cpp
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}
)
