# Code used to create a logistic regression model for Homework 3 of COMP SCI 760.
# Author: Jason Turner

cmake_minimum_required(VERSION 3.16)

# Set the project name and version
enable_language(CXX)
project(log_reg
        VERSION 1.0
        DESCRIPTION "Generates a linear regression model for information about the Titanic survivors."
       )
       
set(CMAKE_BUILD_TYPE Debug)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Specify the executable directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Add the subdirectories containing source code
add_subdirectory(src)

# Adds the higher-level libraries, etc. that are not hosted in the src directory
target_include_directories(log_reg
			   PUBLIC ${CMAKE_BINARY_DIR}
			  )
