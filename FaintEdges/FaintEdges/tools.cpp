#include <iostream>
#include <string>
#include <ctime>
#include <fstream>
#include <queue>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

const double PI = 3.141592653589793238463;

void println(string str){
	cout << str << endl;
}

void println(int n){
	cout << n << endl;
}

double tic(){
	return clock();
}

double toc(double start){
	double elapsed = (clock() - start)/1000.0;
	println(format("Elapsed %2.10f Seconds", elapsed));
	return elapsed;
}

int endRun(int exitCode){
	getchar();
	return exitCode;
}

int sign(int x){
	if (x > 0) return 1;
	else if(x < 0) return -1;
	else return 0;
}

int nchoosek(int n, int k)
{
	if (k == 0) return 1;
	if (n == 0) return 0;
	return nchoosek(n - 1, k - 1) + nchoosek(n - 1, k);
}

int sub2ind(const int rows, const int cols, const int row, const int col)
{
	return cols*row + col;
}

void matSub2ind(const int rows, const int cols, const Mat& row, const Mat& col, Mat& dest){
	assert(row.size == col.size);
	
	if (true){
		//Mat temp;
		//multiply(row, cols, temp);
		dest = cols*row + col;
	}
	else{
		dest = row.clone();
		assert(row.isContinuous() && col.isContinuous() && dest.isContinuous());

		double* p = (double*)row.data;
		double* cp = (double*)col.data;
		double* dp = (double*)dest.data;

		for (int i = 0; i < row.size().area(); ++i){
			*dp++ = sub2ind(rows, cols, (int)*p++, (int)*cp++);
		}
	}
}

void matSub2ind(const Size size, const Mat& row, const Mat& col, Mat& dest){
	matSub2ind(size.height, size.width, row, col, dest);
}

void ind2sub(const int ind, const int cols, const int rows, int &row, int &col)
{
	row = ind / cols;
	col = ind%cols;
}

bool exists(const string& name) {
	ifstream infile("thefile.txt");
	infile.close();
	return true;
}

Mat readImage(string img, bool kill = false){
	Mat image;
    image = imread(img, CV_LOAD_IMAGE_GRAYSCALE);
	
    if(! image.data )
    {
        println("Could not open or find the image");
		if(kill){
			exit(-1);
		}
    }
	return image;
}

void showImage(Mat& image, int fig, double scale = 1, bool wait = false){
	Mat iBig;
	resize(image, iBig, Size(0, 0), scale, scale);
	string winName = format("Window %d", fig);
	namedWindow( winName, WINDOW_AUTOSIZE );
    imshow( winName, iBig );
	if(wait){
		waitKey(0);
	}
}

void findIndices(const Mat& M, Mat& ind){
	assert(M.isContinuous());
	double* p = (double*)M.data;
	for (int i = 0; i < M.size().area(); ++i){
		if (*p++!= 0){
			ind.push_back(i);
		}
	}
	assert(ind.isContinuous());
	ind = ind.reshape(0, 1);
}

void copyIndices(const Mat& D, const Mat& ind, Mat& dest){
	Mat values = D.clone();
	dest = ind.clone();
	assert(values.isContinuous() && ind.isContinuous() && dest.isContinuous());
	values = values.reshape(0, 1);
	double* ip = (double*)ind.data;
	double* dp = (double*)dest.data;
	double* vp = (double*)values.data;

	for (int i = 0; i < ind.size().area(); ++i){
		int curInd = (int)*ip++;
		if (curInd >= 0){
			*dp++ = vp[curInd];
		}
		else{
			dp++;
		}
	}
}

void setValueIfTrue(const double value, Mat& dst, const Mat& flag){
	assert(dst.size().area() == flag.size().area());
	assert(dst.isContinuous() && flag.isContinuous());
	bool* fp = (bool*)flag.data;
	double* dp = (double*)dst.data;

	for (int i = 0; i < flag.size().area(); ++i){
		if (*fp++){
			*dp++ = value;
		}
		else{
			dp++;
		}
	}
}

void setValueIfTrue(const Mat& src, Mat& dst, const Mat& flag){
	assert(dst.size().area() == flag.size().area() && src.size().area() == flag.size().area());
	assert(src.isContinuous() && dst.isContinuous() && flag.isContinuous());
	bool* fp = (bool*)flag.data;
	double* dp = (double*)dst.data;
	double* sp = (double*)src.data;

	for (int i = 0; i < flag.size().area(); ++i){
		if (*fp++){
			*dp++ = *sp++;
		}
		else{
			dp++;
			sp++;
		}
	}
}

void keepSelectedRows(const Mat& src, const Mat& goodCols, Mat& dst){
	assert(src.rows == goodCols.cols && goodCols.rows == 1);
	assert(goodCols.isContinuous());

	Mat values = src.clone();
	Mat d;
	bool* p = (bool*)goodCols.data;

	for (int i = 0; i < goodCols.cols; ++i){
		if (*p++){
			d.push_back(values.row(i));
		}
	}
	d.copyTo(dst);
}

void keepSelectedColumns(const Mat& src, const Mat& goodCols, Mat& dst){
	assert(src.cols == goodCols.cols && goodCols.rows == 1);	
	assert(goodCols.isContinuous());

	Mat values = src.clone();
	Mat d;
	uchar* p = goodCols.data;

	for (int i = 0; i < goodCols.cols; ++i){
		if (*p++){
			d.push_back(values.col(i));
		}
	}
	d = d.reshape(0,values.rows);
	d.copyTo(dst);
}

void keepTrue(const Mat& src, const Mat& keep, Mat& dst){
	assert(src.size().area() == keep.size().area());
	assert(keep.isContinuous());
	Mat d;
	uchar* kp = keep.data;
	double* sp = (double*)src.data;
	for (int i = 0; i < keep.size().area(); ++i){
		double v = *sp++;
		if (*kp++){
			d.push_back(v);
		}
	}
	if (d.size().area()){
		d = d.reshape(0, 1);
		d.copyTo(dst);
	}
	else{
		dst.release();
	}
}

void setValuesInInd(const Mat& values, const Mat& ind, Mat& dst){
	assert(values.size().area() == ind.size().area());
	assert(values.isContinuous() && ind.isContinuous() && dst.isContinuous());

	double* vp = (double*)values.data;
	double* ip = (double*)ind.data;
	double* dp = (double*)dst.data;

	for (int i = 0; i < values.size().area(); ++i){
		dp[(int)*ip++] = *vp++;
	}
}

double maxValue(const Mat& m){
	double minVal;
	double maxVal;
	Point minLoc;
	Point maxLoc;
	minMaxLoc(m, &minVal, &maxVal, &minLoc, &maxLoc);
	return maxVal;
}

void getHighestKValues(const Mat& src, Mat& dst,Mat& idx,int k){
	Mat values = src.clone();
	priority_queue<pair<double, double>> q;

	assert(values.isContinuous());
	double* p = (double*)values.data;

	for (int i = 0; i < values.size().area(); ++i) {
		q.push(pair<double, double>(*p++,(double)i));
	}
	for (int i = 0; i < k; ++i) {
		idx.push_back(q.top().second);
		dst.push_back(q.top().first);
		q.pop();
	}
}

void reorder(const Mat& idx, const Mat& src, Mat& dst){
	assert(idx.size().area() == src.size().area());
	assert(src.isContinuous() && idx.isContinuous());
	Mat d(src.size(), src.type());
	assert(d.isContinuous());
	int* ip = (int*)idx.data;
	double* dp = (double*)d.data;
	double* sp = (double*)src.data;

	for (int i = 0; i < idx.size().area(); ++i){
		*dp++ = sp[*ip++];
	}
	d.copyTo(dst);
}

void reorderCols(const Mat& idx, const Mat& src, Mat& dst){
	assert(idx.size().area() == src.cols);
	assert(idx.isContinuous());
	Mat d(src.size(), src.type());
	assert(d.isContinuous());
	int* ip = (int*)idx.data;
	
	for (int i = 0; i < idx.size().area(); ++i){
		src.col(*ip++).copyTo(d.col(i));
	}
	d.copyTo(dst);
}

int indToAngle(int rows, int cols, int ind0, int ind1){
	int x0, y0, x1, y1;
	ind2sub(ind0, rows, cols, x0, y0);
	ind2sub(ind1, rows, cols, x1, y1);
	int v1 = x1 - x0;
	int v2 = y1 - y0;

	double angle = (v1 == 0) ? sign(v2)*PI/2 : atan(v2 / v1);
	angle *= 180 / PI;
	int ang = (int)(angle)+360;

	if (v1 < 0){
		ang += 180;
	}
	ang = ang%360;

	assert(ang >= 0 && ang <360);
	return ang;
}

void indToAngle(int rows, int cols, Mat& ind0, Mat& ind1, Mat& dst){
	Mat d(ind0.size(), ind0.type());
	assert(ind1.isContinuous() && ind0.isContinuous() && d.isContinuous());
	double* dp = (double*)d.data;
	double* p0 = (double*)ind0.data;
	double* p1 = (double*)ind1.data;

	for (int i = 0; i < d.size().area(); ++i){
		*dp++ = indToAngle(rows, cols, (int)*p0++, (int)*p1++);
	}
	d.copyTo(dst);
}