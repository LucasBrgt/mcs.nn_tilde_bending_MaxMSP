cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(nn_tilde)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (APPLE)
    set(CMAKE_CXX_FLAGS "-faligned-allocation")
    if (CMAKE_OSX_ARCHITECTURES STREQUAL "")
        set(CMAKE_OSX_ARCHITECTURES ${CMAKE_HOST_SYSTEM_PROCESSOR})
    endif()
    message("CMAKE_OSX_ARCHITECTURES: ${CMAKE_OSX_ARCHITECTURES}")
    set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0")
endif()

add_subdirectory(backend) # DEEP LEARNING BACKEND

if(APPLE OR MSVC)
    add_subdirectory(frontend/maxmsp/mcs.nn_tilde) # MAX MSP EXTERNAL
endif()
