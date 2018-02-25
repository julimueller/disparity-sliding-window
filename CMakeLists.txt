cmake_minimum_required(VERSION 2.8.3)
project(disparity_sliding_window)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)

find_package(OpenCV 2.4)
IF(OpenCV_FOUND)
  message("ocv found")
  add_definitions("-DOpenCV_FOUND")
ENDIF()
SET(CMAKE_CXX_FLAGS "-std=c++0x")
add_definitions ("-Wall -lboost_system")

find_package( PythonLibs REQUIRED )
include_directories( ${PYTHON_INCLUDE_DIRS} )

find_package( Boost REQUIRED COMPONENTS iostreams python)
include_directories( ${Boost_INCLUDE_DIR} )


INCLUDE_DIRECTORIES(include)

# lib
SET(SOURCES   src/disparity_sliding_window.cpp src/disparity_sliding_window_test.cpp src/disparity_sliding_window_python.cpp src/kitti_evaluate_object.cpp)
SET(HEADERS   include/disparity_sliding_window.h)
message(${Boost_LIBRARIES})
ADD_EXECUTABLE(disparity_sliding_window_test ${HEADERS} ${SOURCES})
add_library(disparity_sliding_window SHARED ${HEADERS} ${SOURCES})

target_link_libraries(disparity_sliding_window ${OpenCV_LIBS} ${PYTHON_LIBRARIES} ${Boost_LIBRARIES} )
target_link_libraries(disparity_sliding_window_test ${OpenCV_LIBS} ${PYTHON_LIBRARIES} ${Boost_LIBRARIES})

# Define the wrapper library that wraps our library
add_library( dsw_python SHARED src/disparity_sliding_window_python.cpp )
target_link_libraries( dsw_python ${PYTHON_LIBRARIES} ${Boost_LIBRARIES}   disparity_sliding_window)
# don't prepend wrapper library name with lib
set_target_properties( dsw_python PROPERTIES PREFIX "" )

install(TARGETS disparity_sliding_window ARCHIVE DESTINATION lib LIBRARY DESTINATION lib)
install(FILES ${HEADERS} DESTINATION "include/disparity_sliding_window")
INSTALL(TARGETS disparity_sliding_window_test RUNTIME DESTINATION bin)
