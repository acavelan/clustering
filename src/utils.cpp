#include "utils.hpp"

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


pair<Point2d, Point2d> makeBoundingBox(vector<vector<Point2d>> pointGroupList)
{
    double minX = +numeric_limits<double>::infinity();
    double minY = +numeric_limits<double>::infinity();
    double maxX = -numeric_limits<double>::infinity();
    double maxY = -numeric_limits<double>::infinity();

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

    cout << "BoundingBox:" << endl;
    cout << "\tmin:(" << minX << " " << minY << ") max:(" << minX << " " << minY << ")" << endl;

    return make_pair(Point2d(minX, minY), Point2d(maxX, maxY));
}


void showPoints(vector<Point2d> pointList, Scalar color, Mat drawingImage, pair<Point2d, Point2d> boundingBox)
{
    Mat drawing = Mat::zeros(768, 1024, CV_8UC3);

    for(auto& point : pointList)
    {
        Point2i pointPos;
        pointPos.x = 5 + int((point.x-boundingBox.first.x) / (boundingBox.second.x-boundingBox.first.x) * (drawingImage.cols-10) + 0.5);
        pointPos.y = 5 + int((point.y-boundingBox.first.y) / (boundingBox.second.y-boundingBox.first.y) * (drawingImage.rows-10) + 0.5);
        circle(drawingImage, pointPos, 1, color, -1, 8);
        //cout << pointPos.x << " " << pointPos.y << endl;
    }
}


