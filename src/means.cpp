#include "utils.hpp"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <array>
#include <ctime>

using namespace cv;
using namespace std;


/*void createDescriptors(const vector<string> &files, vector<vector<float>> &descriptors, int &featureCount)
{
	for(auto& file : files)
    {
        Mat srcGray = imread(file, CV_LOAD_IMAGE_GRAYSCALE);

        //blur(srcGray, srcGray, Size(3, 3));

        float m = mean(srcGray)[0];

        vector<float> features;
        features.push_back(m);

        descriptors.push_back(features);
    }

    featureCount = 1;
}*/

void createDescriptors(const vector<string> &files, vector<vector<float>> &descriptors, int &featureCount)
{
	Mat dictionary; 
	FileStorage fs("dictionary.yml", FileStorage::READ);

	if(fs.isOpened() == false)
	{
		cout << "Creating Bag of Words dictionary ..." << endl;
		Mat input;    
		//To store the keypoints that will be extracted by SIFT
		vector<KeyPoint> keypoints;
		//To store the SIFT descriptor of current image
		Mat descriptor;
		//To store all the descriptors that are extracted from all the images.
		Mat featuresUnclustered;
		//The SIFT feature extractor and descriptor
		SiftDescriptorExtractor detector;

		//I select 20 (1000/50) images from 1000 images to extract
		//feature descriptors and build the vocabulary
		for(auto& file : files)
		{
		    //open the file
		    input = imread(file, CV_LOAD_IMAGE_GRAYSCALE); //Load as grayscale                

		    if(input.rows == 0)
		    {
		    	cout << "Image: " << file << " not found" << endl;
		    	break;
		    }

		    //detect feature points
		    detector.detect(input, keypoints);
		    //compute the descriptors for each keypoint
		    detector.compute(input, keypoints,descriptor);        
		    //put the all feature descriptors in a single Mat object 
		    featuresUnclustered.push_back(descriptor);    
		}    

		//Construct BOWKMeansTrainer
		//the number of bags
		int dictionarySize = 128;
		//define Term Criteria
		TermCriteria tc(CV_TERMCRIT_ITER,100,0.001);
		//retries number
		int retries = 1;
		//necessary flags
		int flags = KMEANS_PP_CENTERS;

		//Create the BoW (or BoF) trainer
		BOWKMeansTrainer bowTrainer(dictionarySize,tc,retries,flags);
		//cluster the feature vectors

		dictionary = bowTrainer.cluster(featuresUnclustered);

		//store the vocabulary
		FileStorage fs("dictionary.yml", FileStorage::WRITE);
		fs << "vocabulary" << dictionary;
		fs.release();
	}
	else
	{
		fs["vocabulary"] >> dictionary;
		fs.release(); 
	}

	cout << "Creating SIFT descriptors ..." << endl;

	//create a nearest neighbor matcher
    Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
    //create Sift feature point extracter
    Ptr<FeatureDetector> detector(new SiftFeatureDetector());
    //create Sift descriptor extractor
    Ptr<DescriptorExtractor> extractor(new SiftDescriptorExtractor);    
    //create BoF (or BoW) descriptor extractor
    BOWImgDescriptorExtractor bowDE(extractor,matcher);
    //Set the dictionary with the vocabulary we created in the first step
    bowDE.setVocabulary(dictionary);

	for(auto& file : files)
    {
    	//To store the BoW (or BoF) representation of the image
    	Mat bowDescriptor;
    	string fileDescriptor = string(file + ".yml");
    	FileStorage descf(fileDescriptor, FileStorage::READ);

    	if(descf.isOpened() == false)
    	{
	        Mat src = imread(file, CV_LOAD_IMAGE_GRAYSCALE);

	        if(src.rows == 0)
		    {
		    	cout << "Image: " << file << " not found" << endl;
		    	break;
		    }

	        vector<KeyPoint> keypoints;        
		    //Detect SIFT keypoints (or feature points)
		    detector->detect(src, keypoints);
 
		    //extract BoW (or BoF) descriptor from given image
		    bowDE.compute(src, keypoints, bowDescriptor);

		    FileStorage descfw(fileDescriptor, FileStorage::WRITE);
		    descfw << "image" << bowDescriptor;
		    descfw.release();
		}
		else
		{
			descf["image"] >> bowDescriptor;
			descf.release(); 
		}
	    
	    vector<float> features;
		for(int i=0; i<bowDescriptor.cols; i++)
			features.push_back(bowDescriptor.at<float>(0, i));

		if(features.size() == 0)
		{
			cout << "Warning: empty SIFT descriptor for " << file << endl;
			for(int i=0; i<dictionary.rows; i++)
				features.push_back(1.0/dictionary.rows);
		}

        descriptors.push_back(features);
    }

    featureCount = dictionary.rows;
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
	if(argc < 2)
	{
		printf("usage: %s classes.csv\n", argv[0]);
		return 0;
	}

	srand(time(0));

	// CHARGEMENT DES DONNEES
	// =======================

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
    createDescriptors(files, descriptors, featureCount);

    // Affichage des descripteurs
    //showDescriptors(names, descriptors);

    // CREATION DE LA BASE DE DONNEE PAR APPRENTISSAGE
    //=================================================

    cout << "Training Kmeans ..." << endl;

    int total = files.size();

    int K = 5;				// Nombre de clusters
    int attempts = 20;		// Nombre d'essais

    float best = 0;
    int maxIter = 1;

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
    	for(int i=0; i<baseSize; i++)
    		ids.push_back(i);
cout << featureCount << endl;
	    // Création de la matrice d'entré pour K-Means (un descripteur par ligne)
	    Mat samples(baseSize, featureCount, DataType<float>::type);
	    for(int i=0 ; i<baseSize ; i++)
	    {
	    	for(int j=0; j<featureCount; j++)
	    	{
	    		cout << i << endl;
	        	samples.at<float>(i, j) = descriptors[ids[i]][j];
	    	}
	    }

	    // KMEANS
	    //========
	    Mat labels, centers;	// Sorties

	    // Critère d'arrêt fixé à 100 itération max avec une précision de 1.0
	    TermCriteria termCriteria(CV_TERMCRIT_EPS|CV_TERMCRIT_ITER, 250, 0.001);

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

	    vector<int> allLabels;
	    computeResult(names, descriptors, centers, allLabels);

	    // Compute validation
	    float result = 1.0f;

	    if(result >= best)
	    {
	    	best = result;
	    	resultLabels = allLabels;
	    	resultCenters = Mat(centers);
	    	bestClusters = clusters;
	    }
	}

	cout << "BEST RESULT: " << best << "%" << endl;

	// Affiche la base de données
	cout << "BASE " << baseSize << ", " << K << " centers" << endl;

	for(unsigned int i=0; i<bestClusters.size(); i++)
	    printf("Cluster[%d].size = %zu\n", i+1, bestClusters[i].size());

	for(unsigned int i=0; i<resultLabels.size(); i++)
		cout << names[i] << " = " << resultLabels[i]+1 << endl;

	/*for(int i=0; i<resultCenters.rows; i++)
	{
	    cout << "Center[" << i << "] = ";
	    for(int j=0; j<resultCenters.cols; j++)
	    	cout << resultCenters.at<float>(i, j) << " ";
	    cout << endl;
	}*/

    cout << "Done." << endl;


    // ACP et affichage

    cout << "DEBUG: " << descriptors.size() << "x" << descriptors[0].size() << endl;

    Mat drawing = Mat::zeros(768, 1024, CV_8UC3);

//*
    auto pointList = pca2D(descriptors);
/*/
    vector<Point2f> pointList;
    for(auto& desc : descriptors)
        pointList.push_back(Point2f(desc[0], desc[1]));
//*/
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
/*/
    // Affichage des couleurs en fonction des groupes trouvé par apprentissage
    vector<Scalar> colorList(5);
    for(unsigned int i=0 ; i<colorList.size() ; i++)
        colorList[i] = Scalar(rand()%255, rand()%255, rand()%255);
    for(unsigned int i=0 ; i<pointList.size() ; i++)
        showPoints({pointList[i]}, colorList[resultLabels[i]], drawing, boundingBox);
//*/
    cout << "BoundingBox:" << endl;
    cout << "\tmin = (" << boundingBox.first.x << ", " << boundingBox.first.y << ")" << endl;
    cout << "\tmax = (" << boundingBox.second.x << ", " << boundingBox.second.y << ")" << endl;

    namedWindow("2DPointView", CV_WINDOW_AUTOSIZE);
    imshow("2DPointView", drawing);
    waitKey(0);

    return 0;
}


