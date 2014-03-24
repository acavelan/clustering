#include "utils.hpp"


#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

#include <iostream>
#include <fstream>
#include <stdexcept>


using namespace std;
using namespace cv;


// Load CSV file
map<string, int> loadDB(const string &dbfilename)
{
    map<string, int> DB;

    ifstream dbfile(dbfilename);

    if(dbfile.is_open())
    {
        string line;
        dbfile >> line;

        while(!dbfile.eof())
        {
            dbfile >> line;
            istringstream ss(line);

            string name;
            getline(ss, name, ',');

            int cls;
            ss >> cls;

            DB[name] = cls;
        }
        dbfile.close();
    }

    return DB;
}


bool checkClass(const string &img, int cls)
{
    bool in = false;
    try
    {
        static map<string, int> DB = loadDB("data/classes.csv");

        in = (DB.at(img) == cls);
    }
    catch(out_of_range &e)
    {
        cout << "Out of range excpetion with " << img << endl;
    }
    catch(exception &e)
    {
        cout << e.what() << endl;
    }

    return in;
}


pair<Point2f, Point2f> makeBoundingBox(vector<Point2f> pointList)
{
    float minX = +numeric_limits<float>::infinity();
    float minY = +numeric_limits<float>::infinity();
    float maxX = -numeric_limits<float>::infinity();
    float maxY = -numeric_limits<float>::infinity();

    for(auto& point : pointList)
    {
        minX = (point.x < minX) ? point.x : minX;
        minY = (point.y < minY) ? point.y : minY;
        maxX = (point.x > maxX) ? point.x : maxX;
        maxY = (point.y > maxY) ? point.y : maxY;
    }

    return make_pair(Point2f(minX, minY), Point2f(maxX, maxY));
}


pair<Point2f, Point2f> makeBoundingBox(vector<vector<Point2f>> pointGroupList)
{
    float minX = +numeric_limits<float>::infinity();
    float minY = +numeric_limits<float>::infinity();
    float maxX = -numeric_limits<float>::infinity();
    float maxY = -numeric_limits<float>::infinity();

    for(auto& pointList : pointGroupList)
    {
        for(auto& point : pointList)
        {
            minX = (point.x < minX) ? point.x : minX;
            minY = (point.y < minY) ? point.y : minY;
            maxX = (point.x > maxX) ? point.x : maxX;
            maxY = (point.y > maxY) ? point.y : maxY;
        }
    }

    return make_pair(Point2f(minX, minY), Point2f(maxX, maxY));
}


void showPoints(vector<Point2f> pointList, Scalar color, Mat drawingImage, pair<Point2f, Point2f> boundingBox)
{
    for(auto& point : pointList)
    {
        Point2i pointPos;
        pointPos.x = 5 + int((point.x-boundingBox.first.x) / (boundingBox.second.x-boundingBox.first.x) * (drawingImage.cols-10) + 0.5);
        pointPos.y = drawingImage.rows - (5 + int((point.y-boundingBox.first.y) / (boundingBox.second.y-boundingBox.first.y) * (drawingImage.rows-10) + 0.5));
        circle(drawingImage, pointPos, 1, color, -1, 8);
        //cout << pointPos.x << " " << pointPos.y << endl;
    }
}


vector<Point2f> pca2D(vector<vector<float>> descriptorList)
{
    int descriptorSize = (descriptorList.size() > 0) ? descriptorList[0].size() : 0;

    Mat inputData(descriptorList.size(), descriptorSize, CV_32F);
    Mat tmpRow(1, 2, CV_32F);
    vector<Point2f> outputData(descriptorList.size());

    for(int i=0 ; i<inputData.rows ; i++)
        for(int j=0 ; j<inputData.cols ; j++)
            inputData.at<float>(i, j) = descriptorList[i][j];

    PCA pca(inputData, Mat(), CV_PCA_DATA_AS_ROW, 2);

    for(int i=0 ; i<inputData.rows ; i++)
    {
        pca.project(inputData.row(i), tmpRow.row(0));
        outputData[i].x = tmpRow.at<float>(0, 0);
        outputData[i].y = tmpRow.at<float>(0, 1);
    }

    return outputData;
}


vector<vector<Point2f>> pca2DList(vector<vector<vector<float>>> descriptorGroupList)
{
    int globalSize = 0;

    for(unsigned int i=0 ; i<descriptorGroupList.size() ; i++)
        globalSize += descriptorGroupList[i].size();

    vector<vector<float>> globalDescriptorList;
    globalDescriptorList.reserve(globalSize);

    for(unsigned int i=0 ; i<descriptorGroupList.size() ; i++)
        globalDescriptorList.insert(globalDescriptorList.end(), descriptorGroupList[i].begin(), descriptorGroupList[i].end());

    vector<Point2f> globalPoints = pca2D(globalDescriptorList);

    vector<vector<Point2f>> result;
    result.reserve(descriptorGroupList.size());

    int offset = 0;
    for(unsigned int i=0 ; i<descriptorGroupList.size() ; i++)
    {
        result.push_back(vector<Point2f>(globalPoints.begin()+offset, globalPoints.begin()+offset+descriptorGroupList[i].size()));
        offset += descriptorGroupList[i].size();
    }

    return result;
}


