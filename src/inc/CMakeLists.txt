# CMakeLists.txt for src/in directory
# Author: Jason Turner

# Copy over the header files
configure_file(sample.h.in ${CMAKE_BINARY_DIR}/inc/sample.h COPYONLY)
