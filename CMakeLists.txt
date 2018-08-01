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
SET(SOURCES   src/disparity_sliding_window.cpp src/disparity_sliding_window_test.cpp src/disparity_sliding_window_python.cpp src/pyboost_cv2_converter.cpp src/python_conversions.cpp)
SET(HEADERS   include/disparity_sliding_window.h include/pyboost_cvconverter.hpp include/python_conversions.h)
ADD_EXECUTABLE(disparity_sliding_window_test ${HEADERS} ${SOURCES})

add_library(pyboost_cvconverter SHARED src/pyboost_cv2_converter.cpp include/pyboost_cvconverter.hpp)
target_link_libraries(pyboost_cvconverter  ${Boost_LIBRARIES} ${OpenCV_LIBS} ${PYTHON_LIBRARIES})

add_library(python_converter SHARED src/python_conversions.cpp include/python_conversions.h)
target_link_libraries(python_converter ${Boost_LIBRARIES} ${OpenCV_LIBS} ${PYTHON_LIBRARIES})

install(TARGETS pyboost_cvconverter ARCHIVE DESTINATION lib LIBRARY DESTINATION lib)
install(FILES ${HEADERS} DESTINATION "include/disparity_sliding_window")

add_library(disparity_sliding_window SHARED ${HEADERS} ${SOURCES})

target_link_libraries(disparity_sliding_window ${OpenCV_LIBS} ${PYTHON_LIBRARIES} ${Boost_LIBRARIES} )
target_link_libraries(disparity_sliding_window_test ${OpenCV_LIBS} ${PYTHON_LIBRARIES} ${Boost_LIBRARIES})


# Define the wrapper library that wraps our library
add_library( dsw_python SHARED src/disparity_sliding_window_python.cpp )
target_link_libraries( dsw_python ${PYTHON_LIBRARIES} ${Boost_LIBRARIES} pyboost_cvconverter disparity_sliding_window python_converter )

# don't prepend wrapper library name with lib
set_target_properties( dsw_python PROPERTIES PREFIX "" )

# install python lib in python user site
execute_process(COMMAND python -m site --user-site OUTPUT_VARIABLE PYTHON_SITE)

# we have to remove a whitespace here :(
string(REGEX REPLACE "\n$" "" PYTHON_SITE "${PYTHON_SITE}")

install(TARGETS disparity_sliding_window ARCHIVE DESTINATION lib LIBRARY DESTINATION lib)
install(TARGETS dsw_python ARCHIVE DESTINATION ${PYTHON_SITE} LIBRARY DESTINATION ${PYTHON_SITE})
install(FILES ${HEADERS} DESTINATION "include/disparity_sliding_window")

INSTALL(TARGETS disparity_sliding_window_test RUNTIME DESTINATION bin)
