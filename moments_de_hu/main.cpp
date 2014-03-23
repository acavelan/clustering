#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <array>

using namespace cv;
using namespace std;


template<long descriptorSize>
vector<Point2d> pca2D(vector<array<double, descriptorSize>> descriptorList)
{
    Mat inputData(descriptorList.size(), descriptorSize, CV_64F);
    Mat tmpRow(1, 2, CV_64F);
    vector<Point2d> outputData(descriptorList.size());

    for(int i=0 ; i<inputData.rows ; i++)
        for(int j=0 ; j<inputData.cols ; j++)
            inputData.at<double>(i, j) = descriptorList[i][j];

    PCA pca(inputData, Mat(), CV_PCA_DATA_AS_ROW, 2);

    for(int i=0 ; i<inputData.rows ; i++)
    {
        pca.project(inputData.row(i), tmpRow.row(0));
        outputData[i].x = tmpRow.at<double>(0, 0);
        outputData[i].y = tmpRow.at<double>(0, 1);
    }

    return outputData;
}


template<long descriptorSize>
vector<vector<Point2d>> pca2DList(vector<vector<array<double, descriptorSize>>> descriptorGroupList)
{
    int globalSize = 0;

    for(int i=0 ; i<descriptorGroupList.size() ; i++)
        globalSize += descriptorGroupList[i].size();

    vector<array<double, descriptorSize>> globalDescriptorList;
    globalDescriptorList.reserve(globalSize);

    for(int i=0 ; i<descriptorGroupList.size() ; i++)
        globalDescriptorList.insert(globalDescriptorList.end(), descriptorGroupList[i].begin(), descriptorGroupList[i].end());

    vector<Point2d> globalPoints = pca2D<descriptorSize>(globalDescriptorList);

    vector<vector<Point2d>> result;
    result.reserve(descriptorGroupList.size());

    int offset = 0;
    for(int i=0 ; i<descriptorGroupList.size() ; i++)
    {
        result.push_back(vector<Point2d>(globalPoints.begin()+offset, globalPoints.begin()+offset+descriptorGroupList[i].size()));
        offset += descriptorGroupList[i].size();
    }

    return result;
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


void showHuMoments(vector<array<double, 7>> hu)
{
    cout.precision(2);
    cout.setf(ios::fixed, ios::floatfield);
    cout << "Moments de Hu: (" << hu.size() << " descripteurs)" << endl;
    for(int i=0 ; i<hu.size() ; i++)
    {
        cout << "\t";
        for(int j=0 ; j<7 ; j++)
            cout << hu[i][j] << " ";
        cout << endl;
    }
}


Scalar randomColor(RNG& randomGen)
{
    static int i = 0;
    //int icolor = (unsigned int)randomGen;
    //return Scalar(icolor&255, (icolor>>8)&255, (icolor>>16)&255);
    vector<Scalar> colors;
    /*colors.push_back(Scalar(256, 0, 0));
    colors.push_back(Scalar(0, 256, 0));
    colors.push_back(Scalar(0, 0, 256));
    colors.push_back(Scalar(256, 256, 0));
    colors.push_back(Scalar(0, 256, 256));
    colors.push_back(Scalar(256, 0, 256));
    colors.push_back(Scalar(256, 256, 256));*/
    for(int j=0 ; j<10 ; j++)
        colors.push_back(Scalar(255, 255, 0));
    for(int j=0 ; j<10 ; j++)
        colors.push_back(Scalar(0, 255, 255));
    for(int j=0 ; j<10 ; j++)
        colors.push_back(Scalar(255, 0, 255));
    for(int j=0 ; j<10 ; j++)
        colors.push_back(Scalar(0, 255, 0));
    for(int j=0 ; j<10 ; j++)
        colors.push_back(Scalar(255, 0, 0));
    return colors[i++];
}


int main(int argc, char** argv)
{
    // Initialisations pour l'affichage
    Mat drawing = Mat::zeros(768, 1024, CV_8UC3);
    RNG randomGen(time(NULL));
    vector<vector<array<double, 7>>> huList;

    for(int h=1 ; h<argc ; h++)
    {
        Mat src = imread(argv[h]);

        /// Convertie l'image en gris et fait un flou gaussien
        Mat srcGray;
        cvtColor(src, srcGray, CV_BGR2GRAY);
        blur(srcGray, srcGray, Size(3, 3));

        // Détecte les contours avec Canny
        Mat cannyOutput;
        const int thresh = 128;
        Canny(srcGray, cannyOutput, thresh, thresh*2, 3);

        // Trouve les différents contours
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(cannyOutput, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

        // Récupère les moments de chaque contour
        vector<Moments> mu(contours.size());
        for(int i=0 ; i<contours.size() ; i++)
            mu[i] = moments(contours[i]);

        // Calcul 7 invariants de Hu pour chaque contour (descripteurs d'un contour)
        vector<array<double, 7>> hu(mu.size());
        for(int i=0 ; i<mu.size() ; i++)
            HuMoments(mu[i], hu[i].data());

        // Affichage des descripteurs de l'image trouvé
        //showHuMoments(hu);

        huList.push_back(hu);
    }

    // Suppression des points lointains
    for(auto& hu : huList)
    {
        for(auto& point : hu)
        {
            double sum = 0;

            for(int i=0 ; i<7 ; i++)
                sum += point[i]*point[i];

            if(sqrt(sum) > 0.18)
                for(int i=0 ; i<7 ; i++)
                    point[i] = 0;
        }
    }
/*
    // Moyennage des descripteurs d'une image
    for(auto& hu : huList)
    {
        array<double, 7> sum = {0, 0, 0, 0, 0, 0, 0};

        for(auto& point : hu)
        {
            for(int i=0 ; i<7 ; i++)
                sum[i] += point[i] / hu.size();
        }

        hu.clear();
        hu.push_back(sum);
        cout << sum[0] << " " << sum[1] << " " << sum[2] << " " << sum[3] << " " << sum[4] << " " << sum[5] << " " << sum[6] << endl;
    }
*/
    auto pointGroupList = pca2DList<7>(huList);
    auto boundingBox = makeBoundingBox(pointGroupList);

    cout << "Colors:" << endl;
    for(auto& pointList : pointGroupList)
    {
        Scalar color = randomColor(randomGen);
        showPoints(pointList, color, drawing, boundingBox);
        cout << "\t" << color << endl;
    }

    // Affichage
    namedWindow("2DPointView", CV_WINDOW_AUTOSIZE);
    imshow("2DPointView", drawing);
    waitKey(0);

    return 0;
}


