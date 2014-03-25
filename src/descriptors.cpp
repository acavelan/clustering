#include "descriptors.hpp"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include <iostream>

using namespace std;
using namespace cv;

void meanDescriptor(const vector<string> &files, vector<vector<float>> &descriptors, int &featureCount)
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
}

void siftDescriptor(const vector<string> &files, vector<vector<float>> &descriptors, int &featureCount)
{
	Mat dictionary; 
	FileStorage fs("dictionary.SIFT.yml", FileStorage::READ);

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
		    detector.compute(input, keypoints, descriptor);
		    //put the all feature descriptors in a single Mat object 
		    featuresUnclustered.push_back(descriptor);    
		}    

		//Construct BOWKMeansTrainer
		//the number of bags
		int dictionarySize = DICTIONARY_SIZE;
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
		FileStorage fs("dictionary.SIFT.yml", FileStorage::WRITE);
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
    	string fileDescriptor = string(file + ".SIFT.yml");
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

void surfDescriptor(const vector<string> &files, vector<vector<float>> &descriptors, int &featureCount)
{
	Mat dictionary; 
	FileStorage fs("dictionary.SURF.yml", FileStorage::READ);

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
		SurfDescriptorExtractor detector;

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
		    detector.compute(input, keypoints, descriptor);
		    //put the all feature descriptors in a single Mat object 
		    featuresUnclustered.push_back(descriptor);    
		}    

		//Construct BOWKMeansTrainer
		//the number of bags
		int dictionarySize = DICTIONARY_SIZE;
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
		FileStorage fs("dictionary.SURF.yml", FileStorage::WRITE);
		fs << "vocabulary" << dictionary;
		fs.release();
	}
	else
	{
		fs["vocabulary"] >> dictionary;
		fs.release(); 
	}

	cout << "Creating SURF descriptors ..." << endl;

	//create a nearest neighbor matcher
    Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
    //create Surf feature point extracter
    Ptr<FeatureDetector> detector(new SurfFeatureDetector());
    //create Surf descriptor extractor
    Ptr<DescriptorExtractor> extractor(new SurfDescriptorExtractor);    
    //create BoF (or BoW) descriptor extractor
    BOWImgDescriptorExtractor bowDE(extractor,matcher);
    //Set the dictionary with the vocabulary we created in the first step
    bowDE.setVocabulary(dictionary);

	for(auto& file : files)
    {
    	//To store the BoW (or BoF) representation of the image
    	Mat bowDescriptor;
    	string fileDescriptor = string(file + ".SURF.yml");
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
			cout << "Warning: empty SURF descriptor for " << file << endl;
			for(int i=0; i<dictionary.rows; i++)
				features.push_back(1.0/dictionary.rows);
		}

        descriptors.push_back(features);
    }

    featureCount = dictionary.rows;
}