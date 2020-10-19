/*
  Reads data for training the logistic regression model.
*/

// Author: Jason Turner

/* User-Defined Headers (1) [Independent/Overwrite Other Headers] */
#include "sample.h"

/* 3rd-Party Headers */

/* Standard Library Headers */
#include <fstream> // Header providing file stream classes, e.g., ifstream, ofstream.
#include <iostream> // Header that defines the standard input/output stream objects, e.g., cout.
#include <sstream> // Header providing string stream classes, e.g., istringstream.
#include <string> // Header introduces string types, character traits and a set of converting functions.

/* Get common commands into global namespace */

/* User-Defined Headers (2) [Dependent] */

/* Preprocessor Directives */

void read_data(sample **const samplesHeadPtr, int *const sampleCount) {

  // Open the file containing the data for the Titanic survivors
  std::ifstream titanicData; // Data for the Titanic Survivors.
  titanicData.open("titanic_data.csv");

  // Set sampleCount to 0
  *sampleCount = 0;

  if (titanicData.is_open()) {
    for (std::string line; std::getline(
             titanicData, line);) { // Read the Titanic data line-by-line.

      std::istringstream in(line); // Make a stream for the line itself.

      std::string entry; // Store the first entry of the line.
      std::getline(in, entry, ',');

      std::string title("Survived"); // First entry of the data title row

      if (entry.compare(title) !=
          0) { // The first entry on the first line is "Survived" and indicates
               // the data title row.
        struct sample *newSample = new sample[1];
        newSample->survived =
            std::stoi(entry);         // First entry of the data is Survived.
        std::getline(in, entry, ','); // Go to next entry on the line.
        newSample->pClass =
            std::stoi(entry); // Second entry of data is Passenger Class.
        std::getline(in, entry, ',');
        newSample->sex = std::stoi(entry); // Third entry of data is Sex.
        std::getline(in, entry, ',');
        newSample->age = std::stoi(entry); // Fourth entry of data is Age.
        std::getline(in, entry, ',');
        newSample->gsts = std::stoi(entry); // Fifth entry of data is Guests
                                            // (number of spouses/siblings).
        std::getline(in, entry, ',');
        newSample->fam = std::stoi(entry); // Sixth entry of data is Family
                                           // (number of parents/children).
        std::getline(in, entry, ',');
        newSample->fare =
            std::stof(entry); // Seventh entry of data is Passenger Fare.

        // Create pointer to next sample in list.
        newSample->next = *samplesHeadPtr;
        *samplesHeadPtr = newSample;

        // Increment the sample count
        *sampleCount += 1;
      }
    }
  }

  titanicData.close();
}
