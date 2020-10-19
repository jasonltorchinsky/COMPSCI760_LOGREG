/* Declares function for training the logistic regression model. */

// Author: Jason Turner

#ifndef TRAIN_MODEL
#define TRAIN_MODEL

/* Trains the logistic regression model, only changes the model parameters */

void train_model(const sample *samplesHead, const int sampleCount,
                 float **const modelPtr);

#endif
