/*
  Trains the logistic regression model.
*/

// Author: Jason Turner

/* User-Defined Headers (1) [Independent/Overwrite Other Headers] */
#include "sample.h"

/* 3rd-Party Headers */

/* Standard Library Headers */
#include <cmath> // Header that declares a set of functions to compute common mathematical operations and transformations, e.g., exp, log.
#include <iostream> // Header that defines the standard input/output stream objects, e.g., cout.
#include <limits> // Header defines elements with the characteristics of arithmetic types. More specifically, it defines a numeric_limits class template and a specialization of it for each of the fundamental types, e.g., infity.

/* Get common commands into global namespace */

/* User-Defined Headers (2) [Dependent] */

/* Preprocessor Directives */

/* Local function declarations */

/* Calculates the chance a passenger survived (theta . sample) */
// pssngr - Data about a given passenger
// model - Model vector
// survivedPtr - Likelihood of survival
void calc_survived(const sample *pssngr, const float *model,
                   float *const survivedPtr);

/* Calculates the log-likelihood for a given model */
// samplesHead - Head of the singly-linked samples list
// sampleCount - Total number of samples
// model - Model vector
// logLikelihoodPtr - Pointer to the log-likelihood value
void calc_log_likelihood(const sample *samplesHead, const int sampleCount,
                         const float *model, float *const logLikelihoodPtr);

/* Calculates the gradient of the log-likelihood for a given model */
// samplesHead - Head of the singly-linked samples list
// sampleCount - Total number of samples
// model - Model vector
// gradLogLikelihoodPtr - Pointer to the gradient of log-likelihood
void calc_grad_log_likelihood(const sample *samplesHead, const int sampleCount,
                              const float *model,
                              float **const gradLogLikelihoodPtr);

/* Function definition */

void train_model(const sample *samplesHead, const int sampleCount,
                 float **const modelPtr) {

  /* Set the ascent parameters */
  float minStepSize = 0.0f;   // Minimum step size
  float maxStepSize = 1.0e3f; // Maximum step size
  int maxSteps = 10000;       // Maximum number of steps

  /* Set the initial step-size, get the intiial log-likelihood */
  float stepSize = maxStepSize; // Initial step-size
  float logLikelihood = 0.0f;
  float *logLikelihoodPtr = &logLikelihood;
  calc_log_likelihood(samplesHead, sampleCount, *modelPtr, logLikelihoodPtr);

  /* Declare variables for ascent loop */
  int stepNum = 0;                 // Current step number.
  float *nextModel = new float[7]; // The model after a step, discard if worse.
  for (int i = 0; i < 7; i++) {
    nextModel[i] = 0.0f;
  }
  float **const nextModelPtr = &nextModel;
  float *gradLogLikelihood =
      new float[7]; // The gradient of the log-likelihood for the current model.
  for (int i = 0; i < 7; i++) {
    gradLogLikelihood[i] = 0.0f;
  }
  float **const gradLogLikelihoodPtr = &gradLogLikelihood;

  /* Main ascent loop */
  while (maxSteps > stepNum) {

    // Calculate the next model
    calc_grad_log_likelihood(samplesHead, sampleCount, *modelPtr,
                             gradLogLikelihoodPtr);
    for (int i = 0; i < 7; i++) {
      nextModel[i] = (*modelPtr)[i] + stepSize * gradLogLikelihood[i];
    }

    // Calculate the log-likelihood for the next model to see if we need to
    // adjust the step size
    float nextLogLikelihood = 0.0f; // Log-likelihood for the next model
    float *const nextLogLikelihoodPtr = &nextLogLikelihood;
    calc_log_likelihood(samplesHead, sampleCount, *nextModelPtr,
                        nextLogLikelihoodPtr);
    // I see the log-likelihood going nan a lot, so we need to handle that.
    if (std::isnan(nextLogLikelihood)) {
      nextLogLikelihood = -1.0f * std::numeric_limits<float>::infinity();
    }
    // If we don't increase the log-likelihood, decrease the step-size until we
    // do.
    while (logLikelihood > nextLogLikelihood && stepSize > minStepSize) {
      stepSize = 9.0e-1f * stepSize;
      for (int i = 0; i < 7; i++) {
        nextModel[i] = (*modelPtr)[i] + stepSize * gradLogLikelihood[i];
      }
      calc_log_likelihood(samplesHead, sampleCount, *nextModelPtr,
                          nextLogLikelihoodPtr);
    }

    // If we actually improve the model, replace our model with the next one
    if (nextLogLikelihood > logLikelihood) {
      if (stepNum % 10 == 0) {
        std::cout << "Step Number: " << stepNum;
        std::cout << " Step Size: " << stepSize;
        std::cout << " Log-Likelihood: " << nextLogLikelihood << std::endl;
      }

      for (int i = 0; i < 7; i++) {
        (*modelPtr)[i] = nextModel[i];
      }
      logLikelihood = nextLogLikelihood;

    } else { // If we didn't improve our model, we are done.
      std::cout << "Final number of steps: " << stepNum << std::endl;
      stepNum = maxSteps;
    }

    // Reset step size
    stepSize = maxStepSize;
    stepNum = stepNum + 1;
  }

  std::cout << "Final log-likelihood: " << logLikelihood << std::endl;
}

/* Local function defintion */

void calc_survived(const sample *pssngr, const float *model,
                   float *const survivedPtr) {

  *survivedPtr = 0.0f;             // Set survived to zero.
  *survivedPtr += model[0] * 1.0f; // Contribution from constant term.
  *survivedPtr +=
      model[1] *
      static_cast<float>(pssngr->pClass); // Contribution from Passenger Class.
  *survivedPtr +=
      model[2] * static_cast<float>(pssngr->sex); // Contribution from Sex.
  *survivedPtr +=
      model[3] * static_cast<float>(pssngr->age); // Contribution from Age.
  *survivedPtr +=
      model[4] * static_cast<float>(pssngr->gsts); // Contribution from Guests.
  *survivedPtr +=
      model[5] * static_cast<float>(pssngr->fam); // Contribution from Family.
  *survivedPtr += model[6] * pssngr->fare;        // Contirbution from Fare.
}

void calc_log_likelihood(const sample *samplesHead, const int sampleCount,
                         const float *model, float *const logLikelihoodPtr) {

  *logLikelihoodPtr = 0.0f; // Set log-likelihood to zero
  const sample *currSample = samplesHead;
  for (int sampleNum = 0; sampleNum < sampleCount; sampleNum++) {
    float dotProd = 0.0f; // Dot product of the model parameters with the sample
    float *const dotProdPtr = &dotProd;
    calc_survived(currSample, model, dotProdPtr);

    // We encounter numerical issues for large dot product, so we will use rough
    // approximation log(1 + e^z) approx z for large z.
    if (dotProd > 1.0e1f) {
      *logLikelihoodPtr +=
          -1.0f * static_cast<float>(currSample->survived) * dotProd;
    } else if (dotProd < -1.0e1f) {
      *logLikelihoodPtr +=
          (1.0f - static_cast<float>(currSample->survived)) * dotProd;
    } else {
      *logLikelihoodPtr += -1.0f * static_cast<float>(currSample->survived) *
                               std::log(1.0f + std::exp(-1.0f * dotProd)) +
                           (static_cast<float>(currSample->survived) - 1.0f) *
                               std::log(1.0f + std::exp(dotProd));
    }

    // Proceed to the next sample in the singly-linked list.
    currSample = currSample->next;
  }
}

void calc_grad_log_likelihood(const sample *samplesHead, const int sampleCount,
                              const float *model,
                              float **const gradLogLikelihoodPtr) {
  for (int i = 0; i < 7; i++) {
    (*gradLogLikelihoodPtr)[i] = 0.0f; // Set log-likelihood gradient to zero
  }
  const sample *currSample = samplesHead;
  for (int sampleNum = 0; sampleNum < sampleCount; sampleNum++) {
    float dotProd = 0.0f; // Dot product of the model parameters with the sample
    float *const dotProdPtr = &dotProd;
    calc_survived(currSample, model, dotProdPtr);

    float gradCoeff =
        static_cast<float>(currSample->survived) -
        (1.0f /
         (1.0f +
          std::exp(-1.0f * dotProd))); // Coefficient common to all entries of
                                       // the vector being added in the sum
    // Add contribution from sample to the gradient of the log-likelihood.
    (*gradLogLikelihoodPtr)[0] +=
        gradCoeff * 1.0f; // Contribution from constant term.
    (*gradLogLikelihoodPtr)[1] +=
        gradCoeff *
        static_cast<float>(
            currSample->pClass); // Contribution from Passenger Class.
    (*gradLogLikelihoodPtr)[2] +=
        gradCoeff *
        static_cast<float>(currSample->sex); // Contribution from Sex.
    (*gradLogLikelihoodPtr)[3] +=
        gradCoeff *
        static_cast<float>(currSample->age); // Contribution from Age.
    (*gradLogLikelihoodPtr)[4] +=
        gradCoeff *
        static_cast<float>(currSample->gsts); // Contribution from Guests.
    (*gradLogLikelihoodPtr)[5] +=
        gradCoeff *
        static_cast<float>(currSample->fam); // Contribution from Family.
    (*gradLogLikelihoodPtr)[6] +=
        gradCoeff * currSample->fare; // Contribution from Fare.

    // Proceed to the next sample in the singly-linked list.
    currSample = currSample->next;
  }
}
