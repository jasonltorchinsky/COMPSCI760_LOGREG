# CMakeLists.txt for src/in directory
# Author: Jason Turner

# Copy over the namelist file
configure_file(titanic_data.csv.in ${CMAKE_BINARY_DIR}/titanic_data.csv COPYONLY)
