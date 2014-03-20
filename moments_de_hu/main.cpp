#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <array>

using namespace cv;
using namespace std;


int main(int argc, char** argv)
{
    Mat src = imread(argv[1]);

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
    cout.precision(2);
    cout.setf(ios::fixed, ios::floatfield);
    cout << "Résultats: (" << hu.size() << " descripteurs)" << endl;
    for(int i=0 ; i<hu.size() ; i++)
    {
        cout << "\t";
        for(int j=0 ; j<7 ; j++)
            cout << hu[i][j] << " ";
        cout << endl;
    }

    return 0;
}


