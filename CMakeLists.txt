cmake_minimum_required(VERSION 3.10) 
 
project(deploy) 
set(CMAKE_CXX_STANDARD 11) 
set(CMAKE_CXX_STANDARD_REQUIRED ON) 
set(CMAKE_CXX_EXTENSIONS OFF) 
set(CMAKE_CXX_FLAGS -g) 
 
aux_source_directory(src/ SOURCE_DIR)

# build ncnn library
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ncnn/CMakeLists.txt")
    message(FATAL_ERROR "The submodules were not downloaded! Please update submodules with \"git submodule update --init --recursive\" and try again.")
endif()

option(NCNN_INSTALL_SDK "" OFF)
option(NCNN_PIXEL_ROTATE "" OFF)
option(NCNN_VULKAN "" ON)
option(NCNN_VULKAN_ONLINE_SPIRV "" ON)
option(NCNN_BUILD_BENCHMARK "" OFF)
option(NCNN_BUILD_TESTS "" OFF)
option(NCNN_BUILD_TOOLS "" OFF)
option(NCNN_BUILD_EXAMPLES "" OFF)
option(NCNN_DISABLE_RTTI "" ON)
option(NCNN_DISABLE_EXCEPTION "" ON)

add_subdirectory(ncnn)
set_target_properties(
        ncnn PROPERTIES
        INTERFACE_COMPILE_OPTIONS "-frtti;-fexceptions"
)

find_package(Vulkan REQUIRED)
find_package(OpenCV REQUIRED)

list(APPEND EXTRA_INCLUDES ${PROJECT_SOURCE_DIR}/include)

list(APPEND EXTRA_LIBS ${OpenCV_LIBS})

#list(APPEND EXTRA_LIBS ${PROJECT_SOURCE_DIR}/ncnn)
 
add_executable(${CMAKE_PROJECT_NAME} ${SOURCE_DIR}) 
 
target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC ${PROJECT_BINARY_DIR} ${EXTRA_INCLUDES})
target_link_libraries(${CMAKE_PROJECT_NAME} PUBLIC ${EXTRA_LIBS} ncnn)
