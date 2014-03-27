#ifndef DESCRIPTORS_H
#define DESCRIPTORS_H

#include <vector>
#include <string>

#define DICTIONARY_SIZE 16

void meanDescriptor(const std::vector<std::string> &files, std::vector<std::vector<float>> &descriptors, int &featureCount);

void gaborDescriptor(const std::vector<std::string> &files, std::vector<std::vector<float>> &descriptors, int &featureCount);

void siftDescriptor(const std::vector<std::string> &files, std::vector<std::vector<float>> &descriptors, int &featureCount);

void surfDescriptor(const std::vector<std::string> &files, std::vector<std::vector<float>> &descriptors, int &featureCount);

void huMomentsDescriptor(const std::vector<std::string> &files, std::vector<std::vector<float>> &descriptors, int &featureCount);

#endif