cmake_minimum_required(VERSION 2.8)

project(deepart)

#set(CMAKE_VERBOSE_MAKEFILE ON)

set (EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set (LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set (ARCHIVE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

if (NOT INSTALL_PREFIX)
	set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/distribute")
else()
	set(CMAKE_INSTALL_PREFIX "${INSTALL_PREFIX}")
endif()

IF (CMAKE_COMPILER_IS_GNUCXX AND "${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE Release)
	#set(CMAKE_BUILD_TYPE Debug)
ENDIF()

IF(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-write-strings")
ENDIF(CMAKE_COMPILER_IS_GNUCXX)

IF(DCMTK_WITH_DEBUG_POSTFIX)
    SET(CMAKE_DEBUG_POSTFIX  "d")
ENDIF(DCMTK_WITH_DEBUG_POSTFIX)

find_package(CUDA)

if (NOT CAFFE_ROOT)
	set(CAFFE_ROOT ${PROJECT_SOURCE_DIR}/caffe)
endif()

if (NOT CudaLBFGS_ROOT)
	set(CudaLBFGS_ROOT ${PROJECT_SOURCE_DIR}/CudaLBFGS/distribute)
endif()

find_package(Boost REQUIRED)

include_directories(
	${CAFFE_ROOT}/distribute/include
	${CudaLBFGS_ROOT}/include
	${PROJECT_SOURCE_DIR}/liblbfgs/install/include
	${CUDA_INCLUDE_DIRS}
	${PROJECT_SOURCE_DIR}
	)

link_directories(
	${CAFFE_ROOT}/distribute/lib
	${PROJECT_SOURCE_DIR}/liblbfgs/install/lib
	${CudaLBFGS_ROOT}/lib
	)

cuda_compile(MATH_KERNELS_O deepart/math_kernels.h deepart/math_kernels.cu)

IF(CMAKE_COMPILER_IS_GNUCXX)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -DUSE_OPENCV")
ENDIF(CMAKE_COMPILER_IS_GNUCXX)


#compile protobuf
set (CMAKE_MODULE_PATH  ${CAFFE_ROOT}/cmake)
message(STATUS ${CMAKE_MODULE_PATH})
include(Utils)
include(ProtoBuf)

# generate protobuf sources
file(GLOB proto_files deepart/proto/*.proto)
set(proto_gen_folder "${PROJECT_BINARY_DIR}/include/deepart/proto")
caffe_protobuf_generate_cpp_py(${proto_gen_folder} proto_srcs proto_hdrs proto_python ${proto_files})

# include python files either to force generation
add_library(proto ${proto_hdrs} ${proto_srcs} ${proto_python})
set_target_properties(proto PROPERTIES
	DEBUG_POSTFIX "d"
    ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin"
	)
# make sure we build all external depepdencies first
if (DEFINED external_project_dependencies)
	add_dependencies(proto ${external_project_dependencies})
endif()

set(src_list ${MATH_KERNELS_O}
	deepart/art_loss_grad.h
	deepart/art_loss_grad.cpp
	deepart/deepart.h
	deepart/deepart.cpp
	)
set(lib_list pthread proto caffe cudalbfgs
	gflags glog protobuf
	opencv_core opencv_highgui opencv_imgproc boost_program_options
	${CUDA_LIBRARIES} ${PROTOBUF_LIBRARIES}
	)


add_executable(deepart 	deepart/main.cpp ${src_list})
target_link_libraries(deepart ${lib_list})

add_library(deep_art ${src_list})
target_link_libraries(deep_art ${lib_list})


# test
add_executable(test_grad deepart/test/test_grad.cpp ${src_list})
target_link_libraries(test_grad ${lib_list})

# ========================== build platform ==========================
message(STATUS "")
message(STATUS "  Platform:")
message(STATUS "    Host:"             ${CMAKE_HOST_SYSTEM_NAME} ${CMAKE_HOST_SYSTEM_VERSION} ${CMAKE_HOST_SYSTEM_PROCESSOR})
if(CMAKE_CROSSCOMPILING)
	message(STATUS "    Target:"         ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_VERSION} ${CMAKE_SYSTEM_PROCESSOR})
endif()
message(STATUS "    CMake:"            ${CMAKE_VERSION})
message(STATUS "    CMake generator:"  ${CMAKE_GENERATOR})
message(STATUS "    CMake build tool:" ${CMAKE_BUILD_TOOL})
if(MSVC)
	message(STATUS "    MSVC:"           ${MSVC_VERSION})
endif()
if(CMAKE_GENERATOR MATCHES Xcode)
	message(STATUS "    Xcode:"          ${XCODE_VERSION})
endif()
if(NOT CMAKE_GENERATOR MATCHES "Xcode")
	message(STATUS "    Configuration:"  ${CMAKE_BUILD_TYPE})
endif()

message(STATUS "")
message(STATUS "  C/C++:")
message(STATUS "    C++ Compiler:"           ${CMAKE_CXX_COMPILER_STR})
message(STATUS "    C++ flags: "    ${CMAKE_CXX_FLAGS})
message(STATUS "    C++ flags (Release):"    ${CMAKE_CXX_FLAGS_RELEASE})
message(STATUS "    C++ flags (Debug):"       ${CMAKE_CXX_FLAGS_DEBUG})
message(STATUS "    C flags (Release):"       ${CMAKE_C_FLAGS_RELEASE})
message(STATUS "    C flags (Debug):"         ${CMAKE_C_FLAGS_DEBUG})
if(NOT WIN32)
	message(STATUS "    Linker flags (Release):"  ${CMAKE_SHARED_LINKER_FLAGS_RELEASE})
	message(STATUS "    Linker flags (Debug):"    ${CMAKE_SHARED_LINKER_FLAGS_DEBUG})
endif()

message(STATUS "Caffe: "  ${CAFFE_ROOT})
message(STATUS "Install to: " ${CMAKE_INSTALL_PREFIX})

install(TARGETS deepart  deep_art proto
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION bin
	ARCHIVE DESTINATION lib
	)
install(FILES deepart/deepart.h
	deepart/art_loss_grad.h
	deepart/cpu_lbfgs_helper.h
	deepart/error_code.h
	DESTINATION include/deepart
	)
install(FILES ${PROJECT_BINARY_DIR}/include/deepart/proto/deepart.pb.h
	DESTINATION include/deepart/proto
	)
