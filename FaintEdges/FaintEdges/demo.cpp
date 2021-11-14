#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "tools.h"
#include "Detector.h"
#include <thread>
#include <stdio.h>
#include <direct.h>
//#include "mex.h"

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
	I = readImage("Simulations/myCurves4.png");
	I.convertTo(I, TYPE);
	I = I / 255;
	Mat E;
	prm.slidingWindow = 0;
	prm.noisyImage = true;
	prm.parallel = true;
	prm.splitPoints = 0;
	// First Iteration, all Image
	myRunIm(I, E, prm);
	E = 1 - E;
	showImage(E, 1, 2, true);

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
}

void myRunIm(const Mat& I, Mat& E, MyParam& prm){
	if (!prm.slidingWindow){
		E = Mat(I.rows, I.cols, TYPE, ZERO);
		myWrapper(I, E, prm, Range::all(), Range::all());
	}
	else{
		prm.parallel = true;
		prm.printToScreen = false;
		int s = min(I.cols, I.rows);
		double j = log2(s);
		j = j == floor(j) ? floor(j) - 1 : floor(j);
		s = (int)pow(2,j) + 1;
		s = min(s, (int)prm.slidingWindow);
		E = Mat(I.rows, I.cols, TYPE, ZERO);
		int ds = (s - 1) / 2;
		Range rx, ry;
		double start = tic();
		int ITER = 0;
		cout << (I.cols/ds+1)*(I.rows/ds+1) << " ITERATIONS" << endl;
		cout << s << " BLOCK" << endl;
		vector<thread> tasks;
		bool parallel = false; 
		for (int x = 0; x < I.cols; x += ds){
			for (int y = 0; y < I.rows; y += ds){
				rx = x + s >= I.cols ? Range(I.cols - s, I.cols) : Range(x, x + s);
				ry = y + s >= I.rows ? Range(I.rows - s, I.rows) : Range(y, y + s);
				cout << "ITER " << ++ITER << endl;
				//cout << rx.end << endl;
				//cout << ry.end << endl;
				Mat curI = I(ry, rx);
				//cout << curI.rows << ',' << curI.cols << endl;
				if (parallel){
					tasks.push_back(thread(myWrapper, curI, E, prm, ry, rx));
				}
				else{
					myWrapper(curI, E, prm, ry, rx);
				}
			}
		}
		if (parallel){
			for (uint i = 0; i < tasks.size(); ++i)
				tasks[i].join();
		}
		toc(start);
	}
	E = E / maxValue(E);
}

std::mutex E_mutex;

void myWrapper(const Mat& I, Mat& E, const MyParam& prm, const Range& ry, const Range& rx){
	Detector d(I, prm);
	Mat curE = d.runIm();
	E_mutex.lock();
	E(ry, rx) = max(E(ry, rx), curE);
	E_mutex.unlock();
}

/*
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	mexPrintf("Run Edge Detection\n");
	if (nrhs != 6) {
		mexErrMsgIdAndTxt("MATLAB:mexcpp:nargin", "MEXCPP requires six input arguments.");
	}
	else if (nlhs != 1) {
		mexErrMsgIdAndTxt("MATLAB:mexcpp:nargout", "MEXCPP requires one output argument.");
	}

	if (!mxIsDouble(prhs[0])) {
		mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble", "Input Matrix must be a double.");
	}

	for (int i = 1; i < 6; ++i){
		if (!mxIsDouble(prhs[i]) || mxGetNumberOfElements(prhs[i]) != 1) {
			mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notScalar", "Input multiplier must be a scalar.");
		}
	}

	MyParam prm;
	double* img1 = (double *)mxGetPr(prhs[0]);
	int cols = (int)mxGetN(prhs[0]);
	int rows = (int)mxGetM(prhs[0]);
	mexPrintf(format("Image Size: %d, %d\n", rows,cols).c_str());
	prm.removeEpsilon = mxGetScalar(prhs[1]);
	prm.maxTurn = mxGetScalar(prhs[2]);
	prm.nmsFact = mxGetScalar(prhs[3]);
	prm.splitPoints = (int)mxGetScalar(prhs[4]);
	prm.minContrast = (int)mxGetScalar(prhs[5]);

	mexPrintf(format("Params: %2.2f, %2.2f, %2.2f, %d, %d\n", prm.removeEpsilon, prm.maxTurn, prm.nmsFact, prm.splitPoints, prm.minContrast).c_str());
	Mat I(rows, cols, TYPE);
	memcpy(I.data, img1, I.rows * I.cols * sizeof(double));
	Detector d(I, prm);
	Mat E = d.runIm();
	plhs[0] = mxCreateDoubleMatrix(E.rows, E.cols, mxREAL);
	memcpy(mxGetPr(plhs[0]), E.data, E.rows * E.cols * sizeof(double));
}
*/