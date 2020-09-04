cmake_minimum_required(VERSION 3.9)

project(covid)
include(cmake/StandardProjectSettings.cmake)

# set(THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_TBB)
include_directories(
  include
  include/agentData
  include/tools
  include/policies
  rapidjson/include
  thrust/
  cxxopts/include
  include/IO
  include/IO/inputFormats
  include/IO/dataProviders
  include/IO/outputs
  parameterParser/include
  include/tools/progressionMatrices
  cmake/config/out)

file(
  GLOB
  SOURCES
  "src/*.cpp"
  "src/agentData/*.cpp"
  "src/tools/*.cpp"
  "src/tools/progressionMatrices/*.cpp"
  "src/IO/dataProviders/*.cpp"
  "src/IO/outputs/*.cpp")

add_executable(${PROJECT_NAME} ${SOURCES})

find_package(OpenMP)
if(OPENMP_FOUND)
  target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
  set(THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_OMP)
endif()

option(
  USE_GPU
  "If Cuda found but set as OFF it CPU code will be generated, if set as ON but no CUDA found then also CPU code will be generated."
  ON)

# TODO move it to a function later
find_package(CUDA)
if(CUDA_FOUND AND USE_GPU)
  message("CUDA exists")
  enable_language(CUDA)
  set(THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_CUDA)

  # architecture
  set(ARCHITECTURE
      "AUTO"
      CACHE STRING "What is your target CUDA architecture?")
  if(${ARCHITECTURE} STREQUAL "AUTO")
    set(CUDA_ARCH_LIST Auto)
    cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS ${CUDA_ARCH_LIST})
    list(APPEND CUDA_NVCC_FLAGS ${CUDA_ARCH_FLAGS})
    string(
      REPLACE ";"
              " "
              ARCH_FLAGS
              "${CUDA_NVCC_FLAGS}")
  else()
    string(
      REPLACE "NUM"
              ${ARCHITECTURE}
              ARCH_FLAGS
              "-gencode arch=compute_NUM,code=sm_NUM")
  endif()

  set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CUDA)
  set(CMAKE_CUDA_FLAGS
      "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr ${ARCH_FLAGS} --expt-extended-lambda -DTHRUST_DEVICE_SYSTEM=${THRUST_DEVICE_SYSTEM} ${OpenMP_CUDA_FLAGS}"
  )

else()
  if(NOT CUDA_FOUND)
    message("CUDA does not exists")
  endif(NOT CUDA_FOUND)
  enable_language(CXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTHRUST_DEVICE_SYSTEM=${THRUST_DEVICE_SYSTEM}")
endif()

find_package(Threads)

# Git version information (build time)
set(GIT_VERSION "unknown")
find_package(Git QUIET)
if(GIT_FOUND)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} describe
    OUTPUT_VARIABLE GIT_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  message(STATUS "Current program version is ${GIT_VERSION}")
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/config/in/version.h.in
               ${CMAKE_CURRENT_SOURCE_DIR}/cmake/config/out/version.h @ONLY)

# Preset simulation template types
set(MovementPolicy
    "RealMovement"
    CACHE STRING "Type of the movement policy.")
set_property(
  CACHE MovementPolicy
  PROPERTY STRINGS
           RealMovement
           DummyMovement
           NoMovement)
set(TestingPolicy
    "DetailedTesting"
    CACHE STRING "Type of the testing policy.")
set_property(
  CACHE TestingPolicy
  PROPERTY STRINGS
           NoTesting
           DetailedTesting)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/config/in/configTypes.h.in
               ${CMAKE_CURRENT_SOURCE_DIR}/cmake/config/out/configTypes.h @ONLY)

# Link this 'library' to set the c++ standard / compile-time options requested
add_library(project_options INTERFACE)
target_compile_features(project_options INTERFACE cxx_std_14)

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

# Very basic PCH example option(ENABLE_PCH "Enable Precompiled Headers" ON) if (ENABLE_PCH) This sets a global PCH
# parameter, each project will build its own PCH, which is a good idea if any #define's change
#
# target_precompile_headers(project_options INTERFACE <vector> <string> <iostream>) endif()

if(ENABLE_TESTING)
  enable_testing()
  message("Building Tests")
  add_subdirectory(test)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json")
  execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json
                          ${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json)
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

if(CUDA_FOUND)
  target_link_libraries(${PROJECT_NAME} PUBLIC ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
else(CUDA_FOUND)
  target_link_libraries(${PROJECT_NAME} PUBLIC ${CMAKE_THREAD_LIBS_INIT})
endif(CUDA_FOUND)
