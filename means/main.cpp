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

int main(int argc, char** argv)
{
	if(argc < 2)
	{
		printf("usage: %s img1 img2 ...\n", argv[0]);
		return 0;
	}

    vector<float> means;
    for(int h=1 ; h<argc ; h++)
    {
        Mat src = imread(argv[h]);

        Mat srcGray;
        cvtColor(src, srcGray, CV_BGR2GRAY);
        blur(srcGray, srcGray, Size(3, 3));

        float m = mean(srcGray)[0];

        means.push_back(m);
    }

    printf("Means:\n");
    for(float m : means)
        printf("\t%f.2\n", m);

    // Création de la matrice d'entré pour K-Means (un point par ligne)
    Mat samples(means.size(), 1, DataType<float>::type);
    for(unsigned int i=0 ; i<means.size() ; i++)
        samples.at<float>(i, 0) = means[i];

    // Nombre de clusters
    int K = 5;

    // Nombre d'essais (renvoie le meilleur résultat)
    int attempts = 100;

    Mat labels, centers;

    // Critère d'arrêt fixé à 100 itération max avec une précision de 1.0
    TermCriteria termCriteria(CV_TERMCRIT_EPS|CV_TERMCRIT_ITER, 250, 1.0);

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

    // Check results
    int good = 0;
    int total = argc-1;
    for(int h=1 ; h<argc ; h++)
    {
    	Mat src = imread(argv[h]);

        Mat srcGray;
        cvtColor(src, srcGray, CV_BGR2GRAY);
        blur(srcGray, srcGray, Size(3, 3));

        float m = mean(srcGray)[0];

        int cls;
        float min = 1000.0f;
        for(int i=0; i<centers.rows; i++)
        {
        	float d = abs(m-centers.at<float>(i, 0));
        	if(d < min)
        	{
        		min = d;
        		cls = i;
        	}
        }

    	string img = "img" + std::to_string(h) + ".jpg";

    	cout << img << " is in " << cls;;

    	if(checkClass(img, cls))
    	{
    		cout << " [TRUE]" << endl;
    		good++;
    	}
    	else
    		cout << " [FALSE]" << endl;
    }

    cout << "RESULT: " << ((float)good/total) * 100 << "%" << endl;

    cout << "Done" << endl;

    return 0;
}


