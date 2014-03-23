#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <array>
#include <ctime>

#include "../utils/utils.hpp"

using namespace cv;
using namespace std;

int matchClass(float desc, Mat centers)
{
	int c = 0;
	float min = 1000.0f;
    for(int i=0; i<centers.rows; i++)
    {
    	float d = abs(desc-centers.at<float>(i, 0));
    	if(d < min)
    	{
    		min = d;
    		c = i;
    	}
    }
    return c;
}

void createDescriptors(const vector<string> &files, vector<float> &descriptors)
{
	for(auto& file : files)
    {
        Mat src = imread(file);

        Mat srcGray;
        cvtColor(src, srcGray, CV_BGR2GRAY);
        blur(srcGray, srcGray, Size(3, 3));

        float m = mean(srcGray)[0];

        descriptors.push_back(m);
    }
}

void showDescriptors(const vector<string> &names, const vector<float> &descriptors)
{
    printf("Descriptors:\n");
    for(unsigned int i=0; i<descriptors.size(); i++)
    	cout << names[i] << ":\t" << descriptors[i] << endl;
}

float computeResult(const vector<string> &names, const vector<float> &descriptors, const Mat &centers)
{
	int good = 0;
	int total = names.size();
    for(int i=0; i<total; i++)
    {
    	string name = names[i];
        float desc = descriptors[i];

        int cls = matchClass(desc, centers);

    	if(checkClass(name, cls))
    		good++;
    }

    return ((float)good/total) * 100;
}

int main(int argc, char** argv)
{
	if(argc < 2)
	{
		printf("usage: %s img1 img2 ...\n", argv[0]);
		return 0;
	}

	srand(time(0));

	// CHARGEMENT DES DONNEE
	// ======================
	int total = argc-1;
	vector<string> files;
	vector<string> names;
	vector<float> descriptors;

	// Initialisation
	cout << "Initialization ..." << endl;
	for(int h=1 ; h<argc ; h++)
	{
		files.push_back(argv[h]);
    	names.push_back("img" + std::to_string(h) + ".jpg");
    }

    // Création des descripteurs
    cout << "Creating descriptors ..." << endl;
    createDescriptors(files, descriptors);

    // Affichage des descripteurs
    //showDescriptors(names, descriptors);

    // CREATION DE LA BASE DE DONNEE PAR APPRENTISSAGE
    //=================================================

    float best = 0;
    int baseSize = 8;
    int maxIter = 1000;
    Mat bestLabels, bestCenters;
    vector<vector<float>> bestClusters;

    for(int it=0; it<maxIter; it++)
    {
    	// Take 10 random images
    	vector<int> ids;
    	for(int i=0; i<baseSize; i++)
    		ids.push_back(rand()%total);

	    // Création de la matrice d'entré pour K-Means (un descripteur par ligne)
	    Mat samples(baseSize, 1, DataType<float>::type);
	    for(int i=0 ; i<baseSize ; i++)
	        samples.at<float>(i, 0) = descriptors[ids[i]];

	    int K = 5;				// Nombre de clusters
	    int attempts = 100;		// Nombre d'essais
	    Mat labels, centers;	// Sorties

	    // Critère d'arrêt fixé à 100 itération max avec une précision de 1.0
	    TermCriteria termCriteria(CV_TERMCRIT_EPS|CV_TERMCRIT_ITER, 100, 1.0);

	    //printf("Running kmeans ...\n");
	    kmeans(samples, K, labels, termCriteria, attempts, KMEANS_RANDOM_CENTERS, centers);

	    vector<vector<float>> clusters(K);
	    for(int i=0; i<labels.rows; i++)
	    {
	        float m = samples.at<float>(labels.at<int>(i, 0), 0);
	        clusters[labels.at<int>(i, 0)].push_back(m);
	    }

	    // VERIFICATION DES RESULTATS
	    //============================

	    float result = computeResult(names, descriptors, centers);

	    if(result > best)
	    {
	    	best = result;
	    	bestLabels = Mat(labels);
	    	bestCenters = Mat(centers);
	    	bestClusters = clusters;
	    }
	}

	cout << "BEST RESULT: " << best << "%" << endl;

	// Show base
	cout << "BASE " << baseSize << ", " << 5 << " centers" << endl;

	for(unsigned int i=0; i<bestClusters.size(); i++)
	    printf("Cluster[%d].size = %zu\n", i, bestClusters[i].size());

	for(int i=0; i<bestCenters.rows; i++)
	    printf("Center[%d] = %f\n", i, bestCenters.at<float>(i, 0));

    cout << "Done." << endl;

    return 0;
}


