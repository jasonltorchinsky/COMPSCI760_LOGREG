# CMakeLists.txt for src/trainmodel directory.
# Author : Jason Turner

# Defines trainmodel as a library.
add_library(train_model
            trainmodel.cpp
	   )

target_include_directories(train_model
                           PUBLIC ${PROJECT_BINARY_DIR}/inc
			  )
