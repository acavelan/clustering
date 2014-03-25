#include "utils.hpp"
#include "descriptors.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/flann/flann.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <array>
#include <ctime>

#define USE_KMEANS 1

using namespace cv;
using namespace std;

void showDescriptors(const vector<string> &names, const vector<vector<float>> &descriptors)
{
    printf("Descriptors:\n");
    for(unsigned int i=0; i<names.size(); i++)
    {
	    cout << names[i] << ":\t";
    	for(unsigned int j=0; j<descriptors[i].size(); j++)
    		cout << descriptors[i][j] << " ";
    	cout << endl;
    }
}

float distance(const vector<float> &v1, const vector<float> &v2)
{
    float sum = 0.0f;

    for(unsigned int i=0; i<v1.size(); i++)
        sum += (v1[i]-v2[i]) * (v1[i]-v2[i]);
    
    return sqrt(sum);
}

int matchClass(vector<float> desc, Mat centers)
{
	int c = 0;
	float min = 100000.0f;

    for(int i=0; i<centers.rows; i++)
    {
    	vector<float> v2;

    	for(int j=0; j<centers.cols; j++)
    		v2.push_back(centers.at<float>(i, j));
    	
    	float d = distance(desc, v2);

    	if(d < min)
    	{
    		min = d;
    		c = i;
    	}
    }
    return c;
}

void computeResult(const vector<string> &names, const vector<vector<float>> &descriptors, const Mat &centers, vector<int> &allLabels)
{
	int total = names.size();

    for(int i=0; i<total; i++)
    {
    	string name = names[i];
        int cls = matchClass(descriptors[i], centers);

        allLabels.push_back(cls);
    }
}

int main(int argc, char** argv)
{
	if(argc < 3)
	{
		printf("usage: %s classes.csv clusters\n", argv[0]);
		return 0;
	}

	srand(time(0));

	// CHARGEMENT DES DONNEES
	// =======================

	int clusters = atoi(argv[2]);

	cout << "Loading data ..." << endl;

	vector<string> files;
	vector<string> names;
	vector<int> expetectedLabels;

	string arg(argv[1]);
	size_t token = arg.find_last_of("/");

	string directory = arg.substr(0, token);
	string csv = arg.substr(token+1);

	loadDB(directory + "/" + csv, files, expetectedLabels);

	if(files.size() == 0)
		return 0;

	names = files;
	for(unsigned int i=0; i<files.size(); i++)
		files[i] = directory + "/" + files[i];

	int featureCount = -1;
	vector<vector<float>> descriptors;

    // DESCRIPTORS
    //=============

    cout << "Creating descriptors ..." << endl;
    CREATE_DESCRIPTORS(files, descriptors, featureCount);

    // Affichage des descripteurs
    //showDescriptors(names, descriptors);

    // CREATION DE LA BASE DE DONNEE PAR APPRENTISSAGE
    //=================================================

    cout << "Training Kmeans ..." << endl;

    int total = files.size();

    // Nombre de clusters
    int K = clusters;

    float best = 0.0f;
    int maxIter = 500;

    Mat resultCenters;
    vector<int> resultLabels;
    vector<vector<vector<float>>> bestClusters;

    int baseSize = total;
    if(total < baseSize)
    {
        cerr << "Not enouth input pictures" << endl;
        exit(1);
    }

    for(int it=0; it<maxIter; it++)
    {
    	// Take images
    	vector<int> ids;
    	for(int i=0; i<total; i+=total/baseSize)
    		ids.push_back(i);

	    // Création de la matrice d'entré pour K-Means (un descripteur par ligne)
	    Mat samples(baseSize, featureCount, DataType<float>::type);
	    for(int i=0 ; i<baseSize ; i++)
	    	for(int j=0; j<featureCount; j++)
	        	samples.at<float>(i, j) = descriptors[ids[i]][j];

	    // KMEANS
	    //========
	    Mat labels, centers;

        #if USE_KMEANS == 1
            // Nombre d'essais
            int attempts = 10;
    	    // Critère d'arrêt fixé à 100 itération max avec une précision de 1.0
    	    TermCriteria termCriteria(CV_TERMCRIT_EPS|CV_TERMCRIT_ITER, 250, 0.001);

    	    kmeans(samples, K, labels, termCriteria, attempts, KMEANS_RANDOM_CENTERS, centers);
        #else
            cvflann::KMeansIndexParams kmean_params(2, 100, cvflann::FLANN_CENTERS_KMEANSPP);

            centers = Mat(K, featureCount, DataType<float>::type);

            int count = flann::hierarchicalClustering<flann::L2<float>>(samples, centers, kmean_params);

            // since you get less clusters than you specified we can also truncate our matrix. 
            centers = centers.rowRange(cv::Range(0, count));
        #endif

	    // VERIFICATION DES RESULTATS
	    //============================

	    vector<int> allLabels;
	    computeResult(names, descriptors, centers, allLabels);

        vector<vector<vector<float>>> clusters(K);
        for(unsigned int i=0; i<allLabels.size(); i++)
        {
            vector<float> features;
            int label = allLabels[i];
            for(int j=0; j<featureCount; j++)
            {
                float feature = samples.at<float>(label, j);
                features.push_back(feature);
            }
            clusters[label].push_back(features);
        }

	     // Renomage des classe (0..(n-1) => 1..n)
    	for(int& label : allLabels)
        	label++;

	    // Compute validation
	    float result = randIndex(names, allLabels, expetectedLabels);

	    if(result >= best)
	    {
	    	best = result;
	    	resultLabels = allLabels;
	    	resultCenters = Mat(centers);
	    	bestClusters = clusters;
	    }
	}

	// Affiche la base de données
	cout << "BASE " << baseSize << ", " << K << " centers" << endl;

	for(unsigned int i=0; i<bestClusters.size(); i++)
	    printf("Cluster[%d].size = %zu\n", i+1, bestClusters[i].size());

	for(unsigned int i=0; i<resultLabels.size(); i++)
		cout << names[i] << " = " << resultLabels[i] << endl;

    // Calcul de l'indice de Rand
    float indice = randIndex(names, resultLabels, expetectedLabels);

    cout << "Rand index: " << indice << endl;

    // ACP et affichage

    Mat drawing = Mat::zeros(768, 1024, CV_8UC3);

//*
    auto pointList = pca2D(descriptors);
//*
    //vector<Point2f> pointList;
    for(auto& desc : descriptors)
        pointList.push_back(Point2f(desc[0], desc[1]));
//*
    auto boundingBox = makeBoundingBox(pointList);
//*
    // Affichage des couleurs en fonction des groupes réels
    Scalar color;
    for(unsigned int i=0 ; i<pointList.size() ; i++)
    {
        if(i%10 == 0)
            color = Scalar(rand()%255, rand()%255, rand()%255);
        showPoints({pointList[i]}, color, drawing, boundingBox);
    }
//*
    // Affichage des couleurs en fonction des groupes trouvé par apprentissage
    vector<Scalar> colorList(5);
    for(unsigned int i=0 ; i<colorList.size() ; i++)
        colorList[i] = Scalar(rand()%255, rand()%255, rand()%255);
    for(unsigned int i=0 ; i<pointList.size() ; i++)
        showPoints({pointList[i]}, colorList[resultLabels[i]], drawing, boundingBox);
//*
    cout << "BoundingBox:" << endl;
    cout << "\tmin = (" << boundingBox.first.x << ", " << boundingBox.first.y << ")" << endl;
    cout << "\tmax = (" << boundingBox.second.x << ", " << boundingBox.second.y << ")" << endl;

    namedWindow("2DPointView", CV_WINDOW_AUTOSIZE);
    imshow("2DPointView", drawing);
    waitKey(0);

    cout << "Done." << endl;

    return 0;
}


