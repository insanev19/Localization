cmake_minimum_required(VERSION 3.10)

project(netvlad_onnx VERSION 1.0.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)

#onnxruntime
set(ONNXRUNTIME_ROOT_PATH /hy-tmp/onnxruntime-linux-x64-gpu-1.8.0)
set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_ROOT_PATH}/include)
set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT_PATH}/lib/libonnxruntime.so)
include_directories(${ONNXRUNTIME_INCLUDE_DIRS})

set(DEPS ${DEPS} "/usr/local/cuda/lib64/libcudart.so")
set(DEPS ${DEPS} "/usr/lib/x86_64-linux-gnu/libcudnn.so" )

find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
    message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
else ()
    message(FATAL_ERROR "opencv Not Found!")
endif (OpenCV_FOUND)

add_executable(netvlad_onnx src/netvlad_onnx.cc)

target_link_libraries(netvlad_onnx PRIVATE 
        ${ONNXRUNTIME_LIB} 
        ${DEPS}
        ${OpenCV_LIBS}
        )

