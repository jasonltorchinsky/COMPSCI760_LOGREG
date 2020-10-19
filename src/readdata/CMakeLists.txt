# CMakeLists.txt for src/readdata directory.
# Author : Jason Turner

# Defines readdata as a library.
add_library(read_data
            readdata.cpp
	   )

target_include_directories(read_data
                           PUBLIC ${PROJECT_BINARY_DIR}/inc
			  )
