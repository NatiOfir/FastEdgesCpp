#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "tools.h"
#include "Detector.h"
#include <stdio.h>
#include <direct.h>
#include <string>

using namespace std;
using namespace cv;

void myRunIm(const Mat& I, Mat& E, MyParam& prm);
void myWrapper(const Mat& I, Mat& E, const MyParam& prm, const Range& ry, const Range& rx);

int main( int argc, char** argv )
{
	//Noisy Image Demo
	Mat I;
	MyParam prm;
	cout << "Noisy Image Demo:" << endl;	
	string str(argv[1]);
	cout << str << endl;
	I = readImage(str);
	I.convertTo(I, TYPE);
	I = I / 255;
	Mat E;
	prm.slidingWindow = 129;
	prm.noisyImage = true;
	prm.parallel = false;
	prm.splitPoints = 0;
	// First Iteration, all Image
	myRunIm(I, E, prm);
	E = 1 - E;
	//showImage(E, 1, 2, true);
	imwrite(argv[2], 255*E);
	/*
	//Real Image Demo
	cout << "Real Image Demo:" << endl;
	I = readImage("real/2.png");
	I.convertTo(I, TYPE);
	I = I / 255;
	prm.slidingWindow = 129;
	prm.noisyImage = false;
	prm.parallel = true;
	prm.splitPoints = 0;
	// First Iteration, all Image
	myRunIm(I, E, prm);
	E = 1 - E;
	showImage(E, 1, 2, true);

	println("Finished");
	return 0;
	*/
}

void myRunIm(const Mat& I, Mat& E, MyParam& prm){
	if (!prm.slidingWindow){
		E = Mat(I.rows, I.cols, TYPE, ZERO);
		myWrapper(I, E, prm, Range::all(), Range::all());
	}
	else{
		prm.parallel = true;
		prm.printToScreen = true;
		int s = min(I.cols, I.rows);
		double j = log2(s);
		j = j == floor(j) ? floor(j) - 1 : floor(j);
		s = (int)pow(2,j) + 1;
		s = min(s, (int)prm.slidingWindow);
		E = Mat(I.rows, I.cols, TYPE, ZERO);
		int ds = (s - 1) / 2;
		ds = s;
		Range rx, ry;
		double start = tic();
		int ITER = 0;
		cout << (I.cols/ds+1)*(I.rows/ds+1) << " ITERATIONS" << endl;
		cout << s << " BLOCK" << endl;
		for (int x = 0; x < I.cols; x += ds){
			for (int y = 0; y < I.rows; y += ds){
				rx = x + s >= I.cols ? Range(I.cols - s, I.cols) : Range(x, x + s);
				ry = y + s >= I.rows ? Range(I.rows - s, I.rows) : Range(y, y + s);
				cout << "ITER " << ++ITER << endl;
				//cout << rx.end << endl;
				//cout << ry.end << endl;
				Mat curI = I(ry, rx);
				//cout << curI.rows << ',' << curI.cols << endl;
				myWrapper(curI, E, prm, ry, rx);
			}
		}
		toc(start);
	}
	E = E / maxValue(E);
}

void myWrapper(const Mat& I, Mat& E, const MyParam& prm, const Range& ry, const Range& rx){
	Detector d(I, prm);
	Mat curE = d.runIm();
	E(ry, rx) = max(E(ry, rx), curE);
}
