#ifndef UTILS_H
#define UTILS_H

#include <opencv2/core/core.hpp>

#include <utility>
#include <vector>
#include <string>
#include <map>


void loadDB(const std::string &dbfilename, std::vector<std::string> &files, std::vector<int> &expetectedLabels);

std::pair<cv::Point2f, cv::Point2f> makeBoundingBox(std::vector<cv::Point2f> pointList);
std::pair<cv::Point2f, cv::Point2f> makeBoundingBox(std::vector<std::vector<cv::Point2f>> pointGroupList);
void showPoints(std::vector<cv::Point2f> pointList, cv::Scalar color, cv::Mat drawingImage, std::pair<cv::Point2f, cv::Point2f> boundingBox);
std::vector<cv::Point2f> pca2D(std::vector<std::vector<float>> descriptorList);
std::vector<std::vector<cv::Point2f>> pca2DList(std::vector<std::vector<std::vector<float>>> descriptorGroupList);
float randIndex(std::vector<std::string> names, std::vector<int> resultLabels, std::vector<int> expectedLabels);


#endif


