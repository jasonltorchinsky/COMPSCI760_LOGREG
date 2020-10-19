# Information

This code was written for the third assignment in the COMP SCI 760 Machine Learning course at the University of Wisconsin-Madison. It trains a logistic regression model using a gradient ascent algorithm on some sample data about survivors of the Titanic to predict the chance of survival of an individual given their passenger class, sex, age, the number of friends/partners they had on the ship, the number of family members they had on the ship, and the fare they paid for their ticket.

This code utilizes the CMake build system, and requires CMake version 3.16. To build this program, create a subdirectory `build` and run the following commands inside of it
```
cmake ..
cmake --build .
```

This generates an executable log\_reg which does not require any command-line arguments to run. 

The raw data may be found in the `titanic_data.csv` file.
