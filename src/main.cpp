#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <array>
#include <ctime>

using namespace cv;
using namespace std;

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
		static map<string, int> DB = loadDB("../data/classes.csv");

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

void createDescriptors(const vector<string> &files, vector<vector<float>> &descriptors, int &featureCount)
{
	for(auto& file : files)
    {
        Mat src = imread(file);

        Mat srcGray;
        cvtColor(src, srcGray, CV_BGR2GRAY);
        blur(srcGray, srcGray, Size(3, 3));

        float m = mean(srcGray)[0];

        vector<float> features;
        features.push_back(m);

        descriptors.push_back(features);
    }

    featureCount = 1;
}

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
    float sum = 0;

    for(unsigned int i=0; i<v1.size(); i++)
        sum += (v1[i]-v2[i]) * (v1[i]-v2[i]);
    
    return sqrt(sum);
}

int matchClass(vector<float> desc, Mat centers)
{
	int c = 0;
	float min = 1000.0f;
    for(int i=0; i<centers.rows; i++)
    {
    	vector<float> center;

    	for(int j=0; j<centers.cols; j++)
    		center.push_back(centers.at<float>(i, j));

    	float d = distance(desc, center);

    	if(d < min)
    	{
    		min = d;
    		c = i;
    	}
    }
    return c;
}

float computeResult(const vector<string> &names, const vector<vector<float>> &descriptors, const Mat &centers)
{
	int good = 0;
	int total = names.size();
    for(int i=0; i<total; i++)
    {
    	string name = names[i];
        int cls = matchClass(descriptors[i], centers);

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

	// CHARGEMENT DES DONNEES
	// =======================

	int total = argc-1;

	vector<string> files;
	vector<string> names;

	int featureCount = -1;
	vector<vector<float>> descriptors;

	// Initialisation
	cout << "Initialization ..." << endl;
	for(int h=1 ; h<argc ; h++)
	{
		files.push_back(argv[h]);
    	names.push_back("img" + to_string(h) + ".jpg");
    }

    // DESCRIPTORS
    //=============

    cout << "Creating descriptors ..." << endl;
    createDescriptors(files, descriptors, featureCount);

    // Affichage des descripteurs
    //showDescriptors(names, descriptors);

    // CREATION DE LA BASE DE DONNEE PAR APPRENTISSAGE
    //=================================================

    float best = 0;
    int baseSize = 8;
    int maxIter = 1000;
    Mat bestLabels, bestCenters;
    vector<vector<vector<float>>> bestClusters;

    for(int it=0; it<maxIter; it++)
    {
    	// Take 10 random images
    	vector<int> ids;
    	for(int i=0; i<baseSize; i++)
    		ids.push_back(rand()%total);

	    // Création de la matrice d'entré pour K-Means (un descripteur par ligne)
	    Mat samples(baseSize, featureCount, DataType<float>::type);
	    for(int i=0 ; i<baseSize ; i++)
	    	for(int j=0; j<featureCount; j++)
	        	samples.at<float>(i, j) = descriptors[ids[i]][j];

	    // KMEANS
	    //========

	    int K = 5;				// Nombre de clusters
	    int attempts = 100;		// Nombre d'essais
	    Mat labels, centers;	// Sorties

	    // Critère d'arrêt fixé à 100 itération max avec une précision de 1.0
	    TermCriteria termCriteria(CV_TERMCRIT_EPS|CV_TERMCRIT_ITER, 100, 1.0);

	    //printf("Running kmeans ...\n");
	    kmeans(samples, K, labels, termCriteria, attempts, KMEANS_RANDOM_CENTERS, centers);

	    vector<vector<vector<float>>> clusters(K);
	    for(int i=0; i<labels.rows; i++)
	    {
	    	int label = labels.at<int>(i, 0);
	    	vector<float> features;
	    	for(int j=0; j<featureCount; j++)
	    	{
	        	float feature = samples.at<float>(label, j);
	        	features.push_back(feature);
	        }
	        clusters[label].push_back(features);
	    }

	    // VERIFICATION DES RESULTATS
	    //============================

	    float result = computeResult(names, descriptors, centers);

	    if(result >= best)
	    {
	    	best = result;
	    	bestLabels = Mat(labels);
	    	bestCenters = Mat(centers);
	    	bestClusters = clusters;
	    }
	}

	cout << "BEST RESULT: " << best << "%" << endl;

	// Affiche la base de données
	cout << "BASE " << baseSize << ", " << 5 << " centers" << endl;

	for(unsigned int i=0; i<bestClusters.size(); i++)
	    printf("Cluster[%d].size = %zu\n", i, bestClusters[i].size());

	for(int i=0; i<bestCenters.rows; i++)
	    printf("Center[%d] = %f\n", i, bestCenters.at<float>(i, 0));

    cout << "Done." << endl;

    return 0;
}


