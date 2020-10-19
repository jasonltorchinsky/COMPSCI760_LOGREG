/* Declares function for reading in training data for the logistic regression
 * model. */

// Author: Jason Turner

#ifndef READ_DATA
#define READ_DATA

/* Reads data for training the logistic regression model. */
// samplesHeadPtr - Pointer to the head of the singly-linked list of samples
// sampleCount - Count of total number of samples

void read_data(sample **const samplesHeadPtr, int *const sampleCount);

#endif
