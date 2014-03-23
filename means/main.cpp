#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <array>

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

int main(int argc, char** argv)
{
	if(argc < 2)
	{
		printf("usage: %s img1 img2 ...\n", argv[0]);
		return 0;
	}

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
    showDescriptors(names, descriptors);

    // CREATION DE LA BASE DE DONNEE PAR APPRENTISSAGE
    //=================================================

    // Création de la matrice d'entré pour K-Means (un descripteur par ligne)
    Mat samples(descriptors.size(), 1, DataType<float>::type);
    for(unsigned int i=0 ; i<descriptors.size() ; i++)
        samples.at<float>(i, 0) = descriptors[i];

    int K = 5;				// Nombre de clusters
    int attempts = 100;		// Nombre d'essais
    Mat labels, centers;	// Sorties

    // Critère d'arrêt fixé à 100 itération max avec une précision de 1.0
    TermCriteria termCriteria(CV_TERMCRIT_EPS|CV_TERMCRIT_ITER, 100, 1.0);

    printf("Running kmeans ...\n");
    kmeans(samples, K, labels, termCriteria, attempts, KMEANS_RANDOM_CENTERS, centers);

    vector<vector<float>> clusters(K);
    for(int i=0; i<labels.rows; i++)
    {
        float m = samples.at<float>(labels.at<int>(i, 0), 0);
        clusters[labels.at<int>(i, 0)].push_back(m);
    }

    for(unsigned int i=0; i<clusters.size(); i++)
        printf("Cluster[%d].size = %zu\n", i, clusters[i].size());

    for(int i=0; i<centers.rows; i++)
    	printf("Center[%d] = %f\n", i, centers.at<float>(i, 0));

    // VERIFICATION DES RESULTATS
    //============================

    int good = 0;
    for(int i=0; i<total; i++)
    {
    	string name = names[i];
        float desc = descriptors[i];

        int cls = matchClass(desc, centers);

    	cout << name << " is in " << cls;

    	if(checkClass(name, cls))
    	{
    		cout << " [TRUE]" << endl;
    		good++;
    	}
    	else
    		cout << " [FALSE]" << endl;
    }

    cout << "RESULT: " << ((float)good/total) * 100 << "%" << endl;

    cout << "Done." << endl;

    return 0;
}


