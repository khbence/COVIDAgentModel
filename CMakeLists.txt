cmake_minimum_required (VERSION 3.0)

project(covid CXX)
include(cmake/StandardProjectSettings.cmake)

#set(THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_CUDA)
set(THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_OMP)
#set(THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_TBB)
set(CMAKE_CXX_FLAGS "-DTHRUST_DEVICE_SYSTEM=${THRUST_DEVICE_SYSTEM}")
include_directories(include 
                    include/agentData
                    include/tools
                    include/policies
                    rapidjson/include
                    thrust/
                    include/IO
                    include/IO/inputFormats
                    parameterParser/include)
file (GLOB SOURCES
      "src/*.cpp"
      "src/agentData/*.cpp"
      "src/tools/*.cpp")

# Link this 'library' to set the c++ standard / compile-time options requested
add_library(project_options INTERFACE)
target_compile_features(project_options INTERFACE cxx_std_17)

# Link this 'library' to use the warnings specified in CompilerWarnings.cmake
add_library(project_warnings INTERFACE)

# standard compiler warnings
include(cmake/CompilerWarnings.cmake)
set_project_warnings(project_warnings)

# sanitizer options if supported by compiler
include(cmake/Sanitizers.cmake)
enable_sanitizers(project_options)

# enable doxygen
include(cmake/Doxygen.cmake)
enable_doxygen()

# allow for static analysis options
include(cmake/StaticAnalyzers.cmake)

# Very basic PCH example
#option(ENABLE_PCH "Enable Precompiled Headers" ON)
#if (ENABLE_PCH)
  # This sets a global PCH parameter, each project will build its own PCH, which
  # is a good idea if any #define's change
  # 
  # 
#  target_precompile_headers(project_options INTERFACE <vector> <string> <iostream>)
#endif()

if(ENABLE_TESTING)
  enable_testing()
  message(
    "Building Tests"
  )
  add_subdirectory(test)
endif()

set (CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")

find_package(OpenMP)
if(OPENMP_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3")

set (CMAKE_EXPORT_COMPILE_COMMANDS ON)

IF( EXISTS "${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json"  )
    EXECUTE_PROCESS( COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json
        ${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json
    )
ENDIF()

add_executable (${PROJECT_NAME} ${SOURCES})
