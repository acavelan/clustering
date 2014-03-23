#ifndef UTILS_H
#define UTILS_H


#include <opencv2/highgui/highgui.hpp>

#include <utility>
#include <vector>
#include <string>
#include <array>
#include <map>


std::map<std::string, int> loadDB(const std::string &dbfilename);
bool checkClass(const std::string &img, int cls);
std::pair<cv::Point2d, cv::Point2d> makeBoundingBox(std::vector<std::vector<cv::Point2d>> pointGroupList);
void showPoints(std::vector<cv::Point2d> pointList, cv::Scalar color, cv::Mat drawingImage, std::pair<cv::Point2d, cv::Point2d> boundingBox);


template<long descriptorSize>
std::vector<cv::Point2d> pca2D(std::vector<std::array<double, descriptorSize>> descriptorList)
{
    cv::Mat inputData(descriptorList.size(), descriptorSize, CV_64F);
    cv::Mat tmpRow(1, 2, CV_64F);
    std::vector<cv::Point2d> outputData(descriptorList.size());

    for(int i=0 ; i<inputData.rows ; i++)
        for(int j=0 ; j<inputData.cols ; j++)
            inputData.at<double>(i, j) = descriptorList[i][j];

    cv::PCA pca(inputData, cv::Mat(), CV_PCA_DATA_AS_ROW, 2);

    for(int i=0 ; i<inputData.rows ; i++)
    {
        pca.project(inputData.row(i), tmpRow.row(0));
        outputData[i].x = tmpRow.at<double>(0, 0);
        outputData[i].y = tmpRow.at<double>(0, 1);
    }

    return outputData;
}


template<long descriptorSize>
std::vector<std::vector<cv::Point2d>> pca2DList(std::vector<std::vector<std::array<double, descriptorSize>>> descriptorGroupList)
{
    int globalSize = 0;

    for(int i=0 ; i<descriptorGroupList.size() ; i++)
        globalSize += descriptorGroupList[i].size();

    std::vector<std::array<double, descriptorSize>> globalDescriptorList;
    globalDescriptorList.reserve(globalSize);

    for(int i=0 ; i<descriptorGroupList.size() ; i++)
        globalDescriptorList.insert(globalDescriptorList.end(), descriptorGroupList[i].begin(), descriptorGroupList[i].end());

    std::vector<cv::Point2d> globalPoints = pca2D<descriptorSize>(globalDescriptorList);

    std::vector<std::vector<cv::Point2d>> result;
    result.reserve(descriptorGroupList.size());

    int offset = 0;
    for(int i=0 ; i<descriptorGroupList.size() ; i++)
    {
        result.push_back(std::vector<cv::Point2d>(globalPoints.begin()+offset, globalPoints.begin()+offset+descriptorGroupList[i].size()));
        offset += descriptorGroupList[i].size();
    }

    return result;
}


#endif


