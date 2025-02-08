cmake_minimum_required(VERSION 3.12)
project(simulation)

# Specify C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find Python and pybind11
find_package(Python3 COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

# Find OpenMP (Optional, but necessary if you want to enable OpenMP)
find_package(OpenMP REQUIRED)

# Add the simulation library (Pybind11 extension)
add_library(simulation MODULE simulation_binding.cpp simulation10.cpp)

# Link pybind11 module
target_link_libraries(simulation PRIVATE pybind11::module)

# Add Python include directories
target_include_directories(simulation PRIVATE ${Python3_INCLUDE_DIRS})

# Set the properties for creating a Python module
set_target_properties(simulation PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}" SUFFIX "${PYTHON_MODULE_EXTENSION}")

# Link OpenMP if found
if(OpenMP_CXX_FOUND)
    target_link_libraries(simulation PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP activated")
endif()
