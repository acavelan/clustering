#include "utils.hpp"


#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

#include <iostream>
#include <fstream>
#include <set>


using namespace std;
using namespace cv;


// Load CSV file
void loadDB(const string &dbfilename, vector<string> &files, vector<int> &expetectedLabels)
{
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

            files.push_back(name);
            expetectedLabels.push_back(cls);
        }
        dbfile.close();
    }
    else
        cout << dbfilename << " not found." << endl;
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
        circle(drawingImage, pointPos, 2, color, -1, 8);
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


set<int> toSet(vector<int> resultLabels)
{
    set<int> classList;

    for(int label : resultLabels)
        classList.insert(label);

    return classList;
}


float randIndex(vector<string> names, vector<int> resultLabels, vector<int> expectedLabels)
{
    auto setResults = toSet(resultLabels);
    auto setExpected = toSet(expectedLabels);

    if(setResults != setExpected)
    {
        cerr << "Assertion error: the number of classes differs between resultLabels and expectedLabels" << endl;
        
        cout << "resultLabels classes:" << endl;
        for(int e : setResults)
            cout << e << " ";
        cout << endl;

        cout << "expectedLabels classes:" << endl;
        for(int e : setExpected)
            cout << e << " ";
        cout << endl;

        exit(1);
    }

    if(resultLabels.size() != expectedLabels.size())
    {
        cerr << "Assertion error: size between resultLabels and expectedLabels differs" << endl;
        exit(1);
    }

    int a = 0, b = 0, c = 0, d = 0;

    for(unsigned int i=0 ; i<resultLabels.size() ; i++)
    {
        for(unsigned int j=0 ; j<resultLabels.size() ; j++)
        {
            if(i != j)
            {
                if(resultLabels[i] == resultLabels[j] && expectedLabels[i] == expectedLabels[j])
                    a++;

                if(resultLabels[i] != resultLabels[j] && expectedLabels[i] != expectedLabels[j])
                    b++;

                if(resultLabels[i] == resultLabels[j] && expectedLabels[i] != expectedLabels[j])
                    c++;

                if(resultLabels[i] != resultLabels[j] && expectedLabels[i] == expectedLabels[j])
                    d++;
            }
        }
    }

    return float(a+b) / float(a+b+c+d);
}


