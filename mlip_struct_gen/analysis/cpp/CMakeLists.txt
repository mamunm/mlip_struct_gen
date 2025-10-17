cmake_minimum_required(VERSION 3.15)
project(mlip_analysis_core LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files
set(SOURCE_FILES
    src/trajectory_reader.cpp
    src/rdf_calculator.cpp
    src/bindings.cpp
)

# Create pybind11 module
pybind11_add_module(_analysis_core MODULE ${SOURCE_FILES})

# Compiler optimizations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(_analysis_core PRIVATE
        -O3
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-march=native>
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-ffast-math>
    )
endif()

# Optional: Enable OpenMP for parallelization
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(_analysis_core PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP found and enabled")
else()
    message(STATUS "OpenMP not found, building without parallel support")
endif()

# Platform-specific settings
if(APPLE)
    set_target_properties(_analysis_core PROPERTIES
        INSTALL_RPATH "@loader_path"
    )
elseif(UNIX)
    set_target_properties(_analysis_core PROPERTIES
        INSTALL_RPATH "$ORIGIN"
    )
endif()

# Installation - scikit-build-core handles this
install(TARGETS _analysis_core
    LIBRARY DESTINATION ${SKBUILD_PLATLIB_DIR}/mlip_struct_gen/analysis
    RUNTIME DESTINATION ${SKBUILD_PLATLIB_DIR}/mlip_struct_gen/analysis
)
