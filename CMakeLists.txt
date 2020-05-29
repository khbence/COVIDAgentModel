cmake_minimum_required (VERSION 3.0)

project(covid)
include(cmake/StandardProjectSettings.cmake)

#set(THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_TBB)
include_directories(include 
                    include/agentData
                    include/tools
                    include/policies
                    rapidjson/include
                    thrust/
                    cxxopts/include
                    include/IO
                    include/IO/inputFormats)
file (GLOB SOURCES
      "src/*.cpp"
      "src/agentData/*.cpp"
      "src/tools/*.cpp")

find_package(OpenMP)
if(OPENMP_FOUND)
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# TODO move it to a function later
#SET(CUDA_SEPARABLE_COMPILATION ON)
find_package(CUDA)
if(CUDA_FOUND)
	message("CUDA exists!!!!")
	enable_language(CUDA)
	#find_library(CUDART_LIBRARY cudart)
	set(THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_CUDA)
	set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CUDA)
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo --expt-relaxed-constexpr -rdc=true --expt-extended-lambda -arch=sm_60 -DTHRUST_DEVICE_SYSTEM=${THRUST_DEVICE_SYSTEM} ${OpenMP_CUDA_FLAGS}")
	#set(LINKER_LANGUAGE CUDA)
	#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -rdc=true -arch=sm_60 -Xcompiler=-fopenmp")
else(CUDA_FOUND)
	message("CUDA not exists!!!!")
	enable_language(CXX)
	# TODO check if openmp exists
	set(THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_OMP)
	set(CMAKE_CXX_FLAGS "-DTHRUST_DEVICE_SYSTEM=${THRUST_DEVICE_SYSTEM}")
    	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif(CUDA_FOUND)


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
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

if(CUDA_FOUND)
	target_link_libraries(${PROJECT_NAME} PUBLIC ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX)
endif(CUDA_FOUND)

