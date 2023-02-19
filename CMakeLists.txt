# set min version and policy
cmake_minimum_required(VERSION 3.24)
cmake_policy(VERSION 3.24)

# set project name and languages
project(mosaic LANGUAGES CXX)

# set C++ standard
set(CMAKE_CXX_STANDARD 17)

# set option/variable policy
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

# load auxillary cmake modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake_aux")

# set library output directory
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# find julia headers
find_package(Julia REQUIRED)

# ensure ROMEO dependency installed during compile time
# there is an additional check in setup.py for pre-built python packages.
execute_process(COMMAND ${Julia_EXECUTABLE} "-e" "using Pkg; !in(\"ROMEO\",\
 [dep.name for (uuid, dep) in Pkg.dependencies()]) ? Pkg.add(\"ROMEO\") : nothing")

# include FetchContent module
include(FetchContent)

# download and include pybind11
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG v2.10.1
)
FetchContent_MakeAvailable(pybind11)

# download and include itk
set(ITKGroup_Core ON CACHE BOOL "" FORCE)
set(ITKGroup_Filtering ON CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(ITK_BUILD_DEFAULT_MODULES OFF CACHE BOOL "" FORCE)
set(ITK_WRAP_PYTHON OFF CACHE BOOL "" FORCE)
set(ZLIBNG_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    ITK
    GIT_REPOSITORY https://github.com/InsightSoftwareConsortium/ITK.git
    GIT_TAG v5.3.0
)
FetchContent_MakeAvailable(ITK)
FetchContent_GetProperties(ITK)
set(ITK_DIR ${itk_BINARY_DIR})
find_package(ITK REQUIRED COMPONENTS ITKCommon ITKImageCompose ITKDisplacementField)

# display all warnings
add_compile_options(-Wall)

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
    LIBRARY DESTINATION lib
)
