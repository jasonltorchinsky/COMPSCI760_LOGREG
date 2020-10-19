/*
   Logistic regression model about survivors of the Titantic
*/

/* User-Defined Headers (1) [Independent/Overwrite Other Headers] */
#include "sample.h"
#include "ver.h"

/* 3rd-Party Headers */

/* Standard Library Headers */
#include <iostream> // cout.

/* Get common commands into global namespace */

/* User-Defined Headers (2) [Dependent] */
#include "readdata.h"
#include "trainmodel.h"

/* Preprocessor Directives */

int main(int argc, char *argv[]) {
  /* Read data from the input file. */
  sample *samplesHead = NULL; // List of samples.
  sample **const samplesHeadPtr = &samplesHead;

  int sampleCount = 0; // Number of samples
  int *const sampleCountPtr = &sampleCount;

  read_data(samplesHeadPtr,
            sampleCountPtr); // Read the titantic data in from the file.

  std::cout << "Read data about " << sampleCount << " passengers." << std::endl;

  /* Train the logistic model with the given data */
  float *model = new float[7]; // Model parameters - 0 = const, 1 = pClass, 2 =
                               // sex, 3 = age, 4 = gsts, 5 = fam, 6 = fare
  float **const modelPtr = &model;

  for (int i = 0; i < 7; i++) {
    model[i] = 0.0e0f; // Initialize model to all zeros.
  }

  train_model(samplesHead, sampleCount, modelPtr);

  std::cout << "Successfully trained the model! Hooray! Our new model is:"
            << std::endl;
  for (int i = 0; i < 7; i++) {
    std::cout << model[i] << std::endl;
  }

  return 0;
}
