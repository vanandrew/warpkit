# set min version and policy
cmake_minimum_required(VERSION 3.13)
cmake_policy(VERSION 3.13)

# set project name and languages
project(mosaic LANGUAGES CXX)

# set C++ standard
set(CMAKE_CXX_STANDARD 17)

# set option/variable policy
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

# load auxillary cmake modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake_aux")

# find julia headers
find_package(Julia REQUIRED)

# include FetchContent module
include(FetchContent)

# download and include pybind11
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v2.10.1
)
FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
    set(PYBIND11_CPP_STANDARD -std=c++17)
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()

# display all warnings
add_compile_options(-Wall)

# compile pybind11 interface
pybind11_add_module(mosaic_cpp MODULE ${CMAKE_SOURCE_DIR}/src/mosaic.cpp)
target_include_directories(mosaic_cpp PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(mosaic_cpp PRIVATE Julia)

# install target
install(
    TARGETS mosaic_cpp
    LIBRARY DESTINATION lib
)
