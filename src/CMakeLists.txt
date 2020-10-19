# CMakeLists.txt for src directory
# Author: Jason Turner

# Set the version information file
configure_file(ver.h.in ${CMAKE_BINARY_DIR}/inc/ver.h)

# Add subdirectories containing libraries
add_subdirectory(in)
add_subdirectory(inc)
add_subdirectory(readdata)
add_subdirectory(trainmodel)

# Define the project's executable
add_executable(log_reg)

# Add source files to the executable
target_sources(log_reg
               PUBLIC main.cpp
	      )

# Add libraries to link to the executable
target_link_libraries(log_reg
                      PUBLIC read_data
		      PUBLIC train_model
		     )

# Add directories to include with the executable
target_include_directories(log_reg
                           PUBLIC ${CMAKE_BINARY_DIR}/inc
			   PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/readdata
			   PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/trainmodel
			  )
