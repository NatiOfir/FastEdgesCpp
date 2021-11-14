#include <iostream>
#include <iterator>
#include <string>
#include <queue>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>
#include <thread>
#include "Detector.h"
#include "tools.h"

using namespace std;
using namespace cv;

/* 
Constructor
Stand alone function
*/
Detector::Detector(const Mat& I, const MyParam& prm){
	_prm = prm;
    _w = prm.filterWidth;
	Mat filter ;
	if (!prm.fibers){
		filter = Mat_<double>(1, 2 * _w + 1);
		assert(filter.isContinuous());
		double* p = (double*)filter.data;
		for (uint i = 0; i < (uint)filter.size().area(); ++i){
			if (i<_w)
				*p++ = 1;
			else if (i>_w)
				*p++ = -1;
			else
				*p++ = 0;
		}
	}
	else{
		filter = Mat_<double>(1, 2 * _w + 1);
		assert(filter.isContinuous());
		double* p = (double*)filter.data;
		for (uint i = 0; i < (uint)filter.size().area(); ++i){
			if (i<_w)
				*p++ = -1;
			else if (i>_w)
				*p++ = -1;
			else
				*p++ = _w*2;
		}
		//double s = _w * 4;
		filter = filter/2;
	}
	I.convertTo(_I, TYPE);
	filter.convertTo(_filter, TYPE);
	_E = Mat(_I.rows, _I.cols, TYPE, ZERO);
	filter2D(_I, _dX, TYPE, _filter, Point(-1, -1), 0.0, BORDER_REFLECT);
	filter2D(_I, _dY, TYPE, _filter.t(), Point(-1, -1), 0.0, BORDER_REFLECT);
	
	uint m = _I.rows;
	uint n = _I.cols;
	uint N = n*m;
	_handle.m = m;
	_handle.n = n;
	_handle.N = N;
	_handle.rSize = Size(_handle.N, _handle.N);
	int maxSize = (int)ceil(_handle.N/3.0);
	_pixelScores = new Mat[maxSize];
	_data = new unordered_map<uint, Mat>[maxSize];

	if (_debug){
		cout << _I << endl;
		cout << _filter << endl;
		Mat a = abs(_dY);
		cout << _dY << endl;
		showImage(a, 1, round(129 / n * 2), true);
		Mat b = abs(_dX);
		cout << _dX << endl;
		showImage(b, 1, round(129 / n * 2), true);
	}
}

/* 
Destructor
Stand alone function
*/
Detector::~Detector(){
	delete[] _pixelScores;
	delete[] _data;
}

/* 
Edge Detection Runner .
Calls to beamCurves() and getScores().
*/
Mat Detector::runIm(){
	uint m = _handle.m;
	uint n = _handle.n;
	//double j = log2(n - 1);
	double start = 0;
	if (_prm.printToScreen){
		println("Build Binary Tree");
		double start = tic();
	}
	Mat S(_handle.m,_handle.n,TYPE);
	assert(S.isContinuous());
	double* p = (double*)S.data;
	for (int i = 0; i < S.size().area(); ++i){
		*p++ = i;
	}

	uint w = _w;
	S.copyTo(_handle.S);
	beamCurves(0,0,&S);
	if (_prm.printToScreen){
		toc(start);
		println("Create Edge Image");
		start = tic();
	}
	getScores();

	if(_prm.printToScreen) toc(start);
	return _E;
}

/* 
Construct the Beam-Curve Binary-Tree of a rectangular image.
Recursic function, calls to beamCurves(), also in parallel mode.
Calls to getBottomLevelSimple() and to mergeTilesSimple().
*/
void Detector::beamCurves(uint index, uint level, Mat* S){
	//cout << "BeamCurves" << endl;
	uint m = S->rows;
	uint n = S->cols;

	_pixelScores[index] = Mat(_handle.m, _handle.n, TYPE,ZERO);

	if ( max(m,n) <= _prm.patchSize ){
		_maxLevel = level;
		//cout << "Before: " << _data.size() << endl;
		getBottomLevelSimple(*S, index);
		//cout << "After: " << _data.size() << endl;
	}
	else{
		Mat S0, S1, S2, S3;
		bool verticalSplit;
		uint s1len = 2;
		if (n>=m) {
			verticalSplit = true;
			uint mid1 = (uint)floor(n / 2);
			uint mid2 = (uint)floor(m / 2);
			subIm(*S, 0, 0, m-1, mid1, S0);
			subIm(*S, 0, mid1, m-1, n-1, S1);
			subIm(*S, 0, 0, mid2, n - 1, S2);
			subIm(*S, mid2, 0, m - 1, n - 1, S3);
		}
		else {
			verticalSplit = false;
			uint mid1 = (uint)floor(m / 2);
			uint mid2 = (uint)floor(n / 2);
			subIm(*S, 0, 0, mid1, n-1, S0);
			subIm(*S, mid1, 0, m-1, n-1, S1);
			subIm(*S, 0, 0, m - 1, mid2, S2);
			subIm(*S, 0, mid2, m - 1, n - 1, S3);
		}
		if (_debug){
			cout << "Tiling:" << endl;
			cout << *S << endl;
			cout << S0 << endl;
			cout << S1 << endl;
			//cout << S2 << endl;
			//cout << S3 << endl;
		}
		
		uint t[] = { 2 * index + 1, 2 * index + 2 };
		Mat* SArr1[] = { &S0, &S1 };
		Mat* SArr2[] = { &S2, &S3 };

		if (_prm.parallel && level%_prm.parallelJump == 0){
			vector<std::thread> tasks;
			for (uint i = 0; i < s1len; ++i)
				tasks.push_back(std::thread(std::bind(&Detector::beamCurves, this, t[i], level + 1, SArr1[i])));
			
			for (uint i = 0; i < tasks.size(); ++i)
				tasks[i].join();
		}
		else{
			for (uint i = 0; i < s1len; ++i)
				beamCurves(t[i], level + 1, SArr1[i]);
		}
		_prm.matrixMaximum(_pixelScores[t[0]], _pixelScores[t[1]], _pixelScores[index]);
		for (uint i = 0; i < s1len; ++i){
			_pixelScores[t[i]].release();
		}
		//cout << "Before: " << _data.size() << endl;
		if (_prm.minLevelOfStitching == 0 || level >= _prm.minLevelOfStitching){
			mergeTilesSimple(S0, S1, index, level, verticalSplit);
			if (verticalSplit)
				mergeTilesSimple(S2, S3, index, level, !verticalSplit);
		}
		
		for (uint i = 0; i < s1len; ++i){
			for (auto it = _data[t[i]].begin(); it != _data[t[i]].end(); ++it){
				insertValueToMap(index, (double)it->first, (Mat)it->second);
			}
			_data[t[i]].clear();
		}

		/*
		Bottom level insert:
		_data[index].insert(pair<uint, Mat>((uint)ind01, bestData.clone()));
		_data[index].insert(pair<uint, Mat>((uint)ind10, bestData2.clone()));

		Fast top level insert:
				
		for (uint i = 0; i < 2; ++i){
			//_data[index].insert(_data[t[i]].begin(), _data[t[i]].end());
			_data[t[i]].clear();
		}
		*/
		double j = log2(index + 2);
		if (_prm.printToScreen && ceil(j) == j){
			println(format("Level %d Complete", (int)j));
		}
	}
}

/* 
Merge two sibling tile responses, go over each pair of sides.
Calls to findBestResponse() and to getBestSplittingPoints().
*/
void Detector::mergeTilesSimple(const Mat& S1, const Mat& S2, uint index, uint level, bool verticalSplit){
	/*
	S1 is the upper or the leftmost rectangle.
	The recangle sides are numbered 0->1->2->3, such that 0 is the left side, 1 is up, 2 is right and 3 is bottom.
	*/
	vector<Mat> edgeS1; getEdgeIndices(S1, edgeS1);
	vector<Mat> edgeS2; getEdgeIndices(S2, edgeS2);
	Mat split;

	vector<std::thread> tasks;

	if (verticalSplit){
		getBestSplittingPoints(edgeS1[2], split, index);
		for (int i = 0; i < edgeS1.size(); ++i){
			if (i == 2) continue; // split of left rectangle
			findBestResponse(edgeS1[i], split, edgeS1[i], index, level);
			for (int j = 0; j < edgeS2.size(); ++j){
				if (j == 0) continue; // split of right rectangle
				if (i == j) continue; // same side of both rectangles
				findBestResponse(edgeS1[i], split, edgeS2[j], index, level);
			}
		}
	}
	else{
		getBestSplittingPoints(edgeS1[3], split, index);
		for (uint i = 0; i < edgeS1.size(); ++i){
			if (i == 3) continue; // split of upper rectangle
			findBestResponse(edgeS1[i], split, edgeS1[i], index, level);
			for (uint j = 0; j < edgeS2.size(); ++j){
				if (j == 1) continue; // split of lower rectangle
				if (i == j) continue; // same side of both rectangles
				findBestResponse(edgeS1[i], split, edgeS2[j], index, level);
			}
		}
	}
}

/* 
Find best responses between two fixed tile sides.
Stand alone function
*/
void Detector::findBestResponse(Mat& edge1, Mat& split, Mat& edge2, uint index, uint level){
	uint index1 = 2 * index + 1;
	uint index2 = 2 * index + 2;
	
	double* e1, *e2, *s;
	int i, j, k;
	double ind0, ind1, s0;
	for (e1 = (double*)edge1.data, i = 0; i < edge1.size().area(); ++i){
		ind0 = *e1++;
		for (e2 = (double*)edge2.data, j = 0; j < edge2.size().area(); ++j){
			ind1 = *e2++;
			if (ind0 == ind1){ continue; }
			if (_debug){
				cout << ind0 << "," << ind1 << endl;
			}
			Mat bestData(_handle.TOTAL, 1, TYPE, ZERO);
			Mat bestValues1;
			Mat bestValues2;
			assert(bestData.isContinuous());
			double* bd = (double*)bestData.data;
			double bestScore = 0;
			double bestS0 = -1;

			for (s = (double*)split.data, k = 0; k < split.size().area(); ++k){
				s0 = *s++;
				if (ind0 == s0 || ind1 == s0){ continue; }
				if (ind0 == 6 && ind1 == 78){// && s0 == 43){
					int ppp = -1;
				}

				double ind0s0 = (double)sub2ind(_handle.rSize.height, _handle.rSize.width, (int)ind0, (int)s0);
				double s0ind1 = (double)sub2ind(_handle.rSize.height, _handle.rSize.width, (int)s0, (int)ind1);
				if (_debug){
					cout << _data[index1].count((uint)ind0s0) << endl;
					cout << _data[index2].count((uint)s0ind1) << endl;
				}
				if (_data[index1].count((uint)ind0s0) != 1 || (_data[index2].count((uint)s0ind1) != 1)){continue;}

				Mat indValues1 = _data[index1].at((int)ind0s0).clone();
				Mat indValues2 = _data[index2].at((int)s0ind1).clone();

				int ang1_0 = (int)indValues1.at<double>(_handle.A0);
				int ang1_1 = (int)indValues1.at<double>(_handle.A1);
				int ang2_0 = (int)indValues2.at<double>(_handle.A0);
				int ang2_1 = (int)indValues2.at<double>(_handle.A1);
				int angDiff = abs(ang1_1 - ang2_0) % CIRCLE;

				if (angDiff > _prm.maxTurn && angDiff < (CIRCLE -_prm.maxTurn)){
					if (_debug){
						cout << "Angle not in range!" << endl;
					}
					continue;
				}

				double len1 = indValues1.at<double>(_handle.L);
				double len2 = indValues2.at<double>(_handle.L);
				double resp1 = indValues1.at<double>(_handle.R);
				double resp2 = indValues2.at<double>(_handle.R);


				double len = len1 + len2;
				assert(len1 >=1 && len2>=1);
				double resp = resp1 + resp2;
				double con;
				if (_w == 0) con = resp/len;
				else con = resp / (2 * _w * len);;
				double thresh = getThreshold(len);
				double score = abs(con) - thresh;

				if (score > bestScore || bestS0<0){
					bd[_handle.C] = con;
					bd[_handle.I0S0] = ind0s0;
					bd[_handle.L] = len;
					bd[_handle.R] = resp;
					bd[_handle.S0I1] = s0ind1;
					bd[_handle.SC] = score;
					bd[_handle.A0] = ang1_0;
					bd[_handle.A1] = ang2_1;
 					bestValues1 = indValues1.clone();
					bestValues2 = indValues2.clone();
					bestS0 = s0;
					bestScore = score;
				}
			}

			if (bestS0 >= 0){
				double ind01 = (double)sub2ind(_handle.rSize.height, _handle.rSize.width, (int)ind0, (int)ind1);
				double ind10 = (double)sub2ind(_handle.rSize.height, _handle.rSize.width, (int)ind1, (int)ind0);
				double ind1s0 = (double)sub2ind(_handle.rSize.height, _handle.rSize.width, (int)ind1, (int)bestS0);
				double s0ind0 = (double)sub2ind(_handle.rSize.height, _handle.rSize.width, (int)bestS0, (int)ind0);

				if (bd[_handle.L] <= _prm.minContrast){
					bd[_handle.minC] = bd[_handle.C];
					bd[_handle.maxC] = bd[_handle.C];
				}
				else{
					double minC = std::min(bestValues1.at<double>(_handle.minC), bestValues2.at<double>(_handle.minC));
					double maxC = std::max(bestValues1.at<double>(_handle.maxC), bestValues2.at<double>(_handle.maxC));
					bd[_handle.minC] = minC;
					bd[_handle.maxC] = maxC;
				}
				double bestValue = abs(bd[_handle.R]);
				if (bestValue > 0){
					Mat& px = _pixelScores[index];
					assert(px.isContinuous());
					double* p = (double*)px.data;
					uint i0 = (int)ind0, i1 = (uint)ind1;
					p[i0] = std::max(p[i0], bestValue);
					p[i1] = std::max(p[i1], bestValue);
				}

				Mat bestData2 = bestData.clone();
				assert(bestData2.isContinuous());
				double* bd2 = (double*)bestData2.data;
				bd2[_handle.R] = -bd[_handle.R];
				bd2[_handle.C] = -bd[_handle.C];
				bd2[_handle.minC] = -bd[_handle.maxC];
				bd2[_handle.maxC] = -bd[_handle.minC];
				bd2[_handle.I0S0] = ind1s0;
				bd2[_handle.S0I1] = s0ind0;
				bd2[_handle.A0] = (int)(bd[_handle.A1] + 0.5*CIRCLE) % CIRCLE;
				bd2[_handle.A1] = (int)(bd[_handle.A0] + 0.5*CIRCLE) % CIRCLE;

				insertValueToMap(index, ind01, bestData.clone());
				insertValueToMap(index, ind10, bestData2.clone());

				//_data[index][(uint)ind01] = bestData.clone();
				//_data[index][(uint)ind10] = bestData2.clone();
			}
		}
	}

}

/* 
Returns the best K points on the interface.
Calls to getHighestKValues from tools.
*/
void Detector::getBestSplittingPoints(Mat& split,Mat& dst,uint index){
	int len = split.size().area();
	if ( (_prm.splitPoints >= len) || (_prm.splitPoints == 0) ){
		split.copyTo(dst);
		return;
	}

	Mat splitScore;
	split.convertTo(split, TYPE);
	copyIndices(_pixelScores[index], split, splitScore);

	int P = _prm.splitPoints;
	Mat idx, dst2;
	getHighestKValues(splitScore, dst2, idx, P);
	
	for (int i = 0; i < idx.size().area(); ++i){
		dst.push_back(split.at<double>((int)idx.at<double>(i)));
	}
}

/* 
Returns the indices of the tile edges, int vector v.
Stand alone function
*/
void Detector::getEdgeIndices(const Mat& S, vector<Mat>& v){
	uint m = S.rows;
	uint n = S.cols;

	v.push_back(S.col(0).clone().reshape(0,1));
	v.push_back(S.row(0).clone());
	v.push_back(S.col(n - 1).clone().reshape(0,1));
	v.push_back(S.row(m - 1).clone());
}

/* 
Returns a sub image tile of the original image by Range().
Stand alone function
*/
void Detector::subIm(const Mat& Ssrc, uint x0, uint y0, uint x1, uint y1, Mat& Sdst){
	Sdst = Ssrc(Range(x0,x1+1), Range(y0,y1+1)).clone();
}

/* 
Returns the length of the side index e.
Stand alone function
*/
uint Detector::getSideLength(uint m, uint n, uint e){
	if (e % 2 == 1)
		return m;
	else
		return n;
}

/* 
Process the bottom level tile.
Calls to getVerticesFromPatchIndices(), getLine(), insertValueToMap and tools functions.
*/
void Detector::getBottomLevelSimple(Mat& S, uint index){
	//cout << S << endl;
	uint m = S.rows;
	uint n = S.cols;
	assert(S.isContinuous());
	int baseInd = (int)S.at<double>(0,0);
	int row, col;
	ind2sub(baseInd, _I.cols, _I.rows, row, col);
	//cout << _handle.S << endl;
	//cout << S << endl;
	//cout << row << "," << col << endl;
	Mat gx, gy, ss;
	subIm(_dX, row, col, row+m-1, col+n-1, gx);
	subIm(_dY, row, col, row+m-1, col+n-1, gy);
	subIm(_handle.S, row, col, row + m - 1, col + n - 1, ss);
	//cout << ss << endl;
	
	for (uint e0 = 1; e0 <= 3; ++e0){
		for (uint e1 = e0 + 1; e1 <= 4; ++e1){
			uint len0 = getSideLength(m, n, e0);
			for (uint v0 = 0; v0 < len0; ++v0){
				uint x0, y0;
				getVerticesFromPatchIndices(e0, v0, m, n, x0, y0);
				uint len1 = getSideLength(m, n, e1);
				for (uint v1 = 0; v1 < len1; ++v1){
					uint x1, y1;
					getVerticesFromPatchIndices(e1, v1, m, n, x1, y1);
					uint ind0 = (uint)S.at<double>(x0, y0);
					uint ind1 = (uint)S.at<double>(x1, y1);

					if (ind0 == ind1) continue;

					Mat P(m, n, TYPE, ZERO);
					Mat F(m, n, TYPE, ZERO);
					// TODO: if expensive, keep line images
					double len = getLine(x0, y0, x1, y1, P, F);
					double dx = (double)x1 - (double)x0;
					double dy = (double)y1 - (double)y0;
					int angle = getAngle(dx, dy);
					double adx = abs(dx);
					double ady = abs(dy);
					switch (_prm.normType)
					{
					case 0:
						len = max(adx, ady);
						break;
					case 1:
						len = adx + ady;
						break;
					case 2:
						len = sqrt(pow(adx,2)+pow(ady,2));
					default:
						break;
					}
					//cout << P << endl;
					//cout << F << endl;
					Mat curIndices;
					findIndices(P, curIndices);
					//cout << curIndices << endl;
					Mat curPixels;
					curIndices.convertTo(curIndices, TYPE);
					copyIndices(S, curIndices, curPixels);
					//cout << curPixels << endl;

					// F.dot for double filder, P.dot for binary filter
					double respX, respY;
					if (_prm.interpolation){
						respX = sign((int)dx)*F.dot(gx);
						respY = -sign((int)dy)*F.dot(gy);
					}
					else{
						respX = sign((int)dx)*P.dot(gx);
						respY = -sign((int)dy)*P.dot(gy);
					}
					double resp;
					//if (abs(dx) == abs(dy)){
					//	resp = 0.5*(respX + respY); 
					//}
					if (abs(dx) > abs(dy)){ 
						resp = respX;
					}
					else{ 
						resp = respY;
					}
					double con = resp/(2 * _w*len);
					bool good = abs(con) >= (_prm.removeEpsilon*_prm.sigma);
					if (!good) continue;
					double minC = con, maxC = con, score;
					if (len <= 0){
						score = numeric_limits<double>::min();
					}
					else{
						double thresh = getThreshold(len);
						score = abs(con) - thresh;
					}

					double ind01 = (double)sub2ind(_handle.rSize.height, _handle.rSize.width, (int)ind0, (int)ind1);
					double ind10 = (double)sub2ind(_handle.rSize.height, _handle.rSize.width, (int)ind1, (int)ind0);
					Mat bestData(_handle.TOTAL, 1, TYPE, ZERO);
					assert(bestData.isContinuous());
					double* bd = (double*)bestData.data;
					bd[_handle.C] = con;
					bd[_handle.I0S0] = 0;
					bd[_handle.L] = len;
					bd[_handle.R] = resp;
					bd[_handle.S0I1] = 0;
					bd[_handle.SC] = score;
					bd[_handle.minC] = con;
					bd[_handle.maxC] = con;
					bd[_handle.A0] = angle;
					bd[_handle.A1] = angle;

					Mat bestData2 = bestData.clone();
					assert(bestData2.isContinuous());
					double* bd2 = (double*)bestData2.data;
					bd2[_handle.R] = -bd[_handle.R];
					bd2[_handle.C] = -bd[_handle.C];
					bd2[_handle.minC] = -bd[_handle.maxC];
					bd2[_handle.maxC] = -bd[_handle.minC];
					int angle2 = (int)(angle + 0.5*CIRCLE) % CIRCLE;
					bd2[_handle.A0] = angle2;
					bd2[_handle.A1] = angle2;

					insertValueToMap(index, ind01, bestData.clone());
					insertValueToMap(index, ind10, bestData2.clone());

					//_data[index][(uint)ind01] = bestData.clone();
					//_data[index][(uint)ind10] = bestData2.clone();

					double value = abs(resp);
					if (value > 0){
						Mat& px = _pixelScores[index];
						assert(px.isContinuous());
						double* p = (double*)px.data;
						uint i0 = (int)ind0, i1 = (uint)ind1;
						p[i0] = std::max(p[i0], value);
						p[i1] = std::max(p[i1], value);
					}

					
					lock();
					_pixels[(uint)ind01] = curPixels.clone();
					_pixels[(uint)ind10] = curPixels.clone();

					//_pixels[(uint)ind01] = curPixels.clone();
					//_pixels[(uint)ind10] = curPixels.clone();
					unlock();
				}
			}
		}
	}
}

void Detector::lock(){
	std::chrono::milliseconds interval(0);
	while (true){
		if (_mtx.try_lock()){
			break;
		}
		else{
			this_thread::sleep_for(interval);
		}
	}
}

void Detector::unlock(){
	_mtx.unlock();
}

/*
Return 0-359 angle from the bottom level edge with directions.
Stand alone function.
*/
int Detector::getAngle(double dx, double dy){
	double v1 = dx;
	double v2 = dy;

	double angle = atan(v2/v1);
	angle *= CIRCLE / 2 / PI;

	if (v1 < 0){
		angle += CIRCLE / 2;
	}
	int a = (int)angle % CIRCLE;
	if (a < 0){
		a += CIRCLE;
	}
	if (!(a >= 0 && a <= CIRCLE)){
		int k = 2;
	}
	assert(a >= 0 && a <= CIRCLE);
	if (_debug){
		cout << v1 << endl;
		cout << v2 << endl;
		cout << a << endl;
	}
	return a;
}

/*
Insert value to key in map m.
Stand alone function.
*/
bool Detector::insertValueToMap(uint index, const double& key, const Mat& value){
	bool betterScore = false;
	if (_data[index].count((uint)key) == 0) {
		_data[index][(uint)key] = value;
		return true;
	}
	else{
		Mat curValue = _data[index][(uint)key];
		betterScore = curValue.at<double>(_handle.SC) <= value.at <double>(_handle.SC);
	}
	if (betterScore){
		_data[index][(uint)key] = value;
		return true;
	}
	else{
		return false;
	}
}

/*
Returns the line that goes through (x0,y0) and (x1,y1).
P is the binary lines image.
F is the double soft line images.
Calls to getLineFilter().
*/
int Detector::getLine(int x0, int y0, int x1, int y1, Mat& P, Mat& F){
	getLineFilter(x0, y0, x1, y1, F);
	int dx = abs(x1 - x0);
	int dy = abs(y1 - y0);
	int L = max(dx, dy);

	int sx;
	if (x0 < x1) sx = 1;
	else sx = -1;
	int sy;
	if(y0 < y1) sy = 1;
	else sy = -1;

	int err = dx - dy;
	bool first = true;

	double corner = 0.5;

	while(true){
		P.at<double>(x0, y0) = 1;
		if (first){
			P.at<double>(x0, y0) = corner;
			first = false;
		}
		if (x0 == x1 && y0 == y1){
			P.at<double>(x0, y0) = corner;
			break;
		}
		int e2 = 2 * err;
		if (e2 > -dy){
			err = err - dy;
			x0 = x0 + sx;
		}
		if (e2 < dx){
			err = err + dx;
			y0 = y0 + sy;
		}
	}
	//L = (int)sqrt(pow(dx,2)+pow(dy,2));
	return L;
}

/*
Returns the double soft line images F.
Calls to interpLine().
*/
void Detector::getLineFilter(int x0, int y0, int x1, int y1, Mat& F){
	double dx = x1 - x0;
	double dy = y1 - y0;

	int L = (int)max(abs(dx), abs(dy));
	bool biggerX = abs(dx) >= abs(dy);
	double xLoop = biggerX ? dx>0 : dx / abs(dy);
	double yLoop = !biggerX ? dy>0 : dy / abs(dx);
	double corner = 0.5;
	double curX = x0;
	double curY = y0;
	for (int j = 0; j <= L; ++j){
		if (j == 0 || j == L){
			interpLine(round(curX), round(curY), F, corner);
		}
		else{
			interpLine(curX, curY, F, 1);
		}

		curX += xLoop;
		curY += yLoop;
	}
}

/*
Update the values of F according to linear interpolation.
Stand alone function.
*/
void Detector::interpLine(double x, double y, Mat& F, double value){
	if (round(x) == x){
		double dy = y - floor(y);
		F.at<double>((int)x, (int)floor(y)) = value*(1 - dy);
		if (dy>0) F.at<double>((int)x, (int)ceil(y)) = value*dy;
	}
	else if (round(y) == y){
		double dx = x - floor(x);
		F.at<double>((int)floor(x), (int)y) = value*(1 - dx);
		if (dx>0) F.at<double>((int)ceil(x), (int)y) = value*dx;
	}
	else{
		cout << "Interp Error" << endl;
	}
}

/*
Translates edge index e and vertex number v to (x,y) coordinates in patch of size m x n.
Stand alone function.
*/
void Detector::getVerticesFromPatchIndices(uint e, uint  v, uint m, uint n, uint& x, uint& y){
	/*
	e is the edge index, v is the vertex index
	the function converet the pair(e,v) to the coordinates (x,y)
	e = 1 is the left patch side
	e = 2 is the top patch side
	*/

	switch (e){
	case 1:
		x = v;
		y = 0;
		break;
	case 2:
		x = 0;
		y = v;
		break;
	case 3:
		x = v;
		y = n-1;
		break;
	case 4:
		x = m-1;
		y = v;
		break;
	default:
		x = -1;
		y = -1;
	}
}

/*
Produces soft edge map from the Beam-Curve Binary-Tree of the image.
Calls to removeKey(), and addEdge().
*/
void Detector::getScores(){
	uint m = _handle.m;
	uint n = _handle.n;
	
	Mat selected(m,n,BOOL,FALSE);

	unordered_map<uint,Mat>& data = _data[0];
	priority_queue<pair<double,uint>> q;
	unordered_map<uint, Mat>::iterator it;
	for (it = data.begin(); it != data.end(); ++it){
		Mat tuple = it->second;
		double* p = (double*)tuple.data;

		double sc = p[_handle.SC], con = p[_handle.C], minC = p[_handle.minC], maxC = p[_handle.maxC], L = p[_handle.L];
		if (sc > 0){
			bool minTest = L <=_prm.minContrast || _prm.minContrast == 0 || (con > 0 && minC >= (con / 2)) || (con < 0 && maxC <= (con / 2));
			if (minTest){
				q.push(pair<double, uint>(sc, it->first));
			}
		}
	}
	//cout << _handle.S << endl;
	uint counter = 0;
	size_t before = q.size();
	while (!q.empty()){
		double curScore = q.top().first;
		uint curKey = q.top().second;
		int row, col;
		ind2sub(curKey, _handle.N, _handle.N, row, col);
		
		q.pop();
		Mat E(m, n, BOOL, FALSE);

		if (_debug){
			cout << "Ind0 = " << row << endl;
			cout << "Ind1 = " << col << endl;
			cout << curScore << endl;
		}
		if (!addEdge(data, curKey, E, 1)){
			continue;
		}
		else if (_prm.nmsFact == 0){
			Mat t;
			E.convertTo(t, TYPE);
			Mat cur = t*curScore;
			max(_E, cur, _E);
			++counter;
		}
		else{
			if (_debug){
				Mat e(_handle.m, _handle.n, TYPE, double(1));
				int r, c;
				ind2sub(curKey, _handle.N, _handle.N, r, c);
				double* p = (double*)e.data;
				p[r] = 0;
				p[c] = 0;
				Mat H;
				Mat k;
				E.convertTo(k, TYPE);
				hconcat(k, e, H);
				cout << e << endl;
				showImage(H, 1, round(129 / n * 2), true);
			}

			Mat curI = E.clone();
			double L = sum(curI)[0];

			Size imSize = curI.size();
			Mat sub1, sub2, sub3, sub4;
			Mat horFalse(1, imSize.width, BOOL, FALSE);
			Mat verFalse(imSize.height, 1, BOOL, FALSE);
			
			vconcat(horFalse, curI(Range(0,imSize.height-1),Range::all()), sub1);
			vconcat(curI(Range(1, imSize.height), Range::all()), horFalse, sub2);
			hconcat(curI(Range::all(), Range(1, imSize.width)), verFalse, sub3);
			hconcat(verFalse , curI(Range::all(), Range(0, imSize.width-1)), sub4);

			Mat curIdialate = curI | sub1 | sub2 | sub3 | sub4;

			Mat coor;
			bitwise_and(curIdialate, selected, coor);
			if (_debug){
				cout << selected << endl;
				cout << curIdialate << endl;
				cout << coor << endl;
			}
			double nmsScore = sum(coor)[0]/L;
			if (nmsScore < _prm.nmsFact){
				//cout << "NMS" << endl;
				if (_debug){
					cout << L << endl;
				}
				removeKey(data, curKey);
				++counter;
				selected = selected | curIdialate;
				Mat t;
				E.convertTo(t, TYPE);
				Mat cur = t*curScore;
				max(_E, cur , _E);
				if (counter > _prm.maxNumOfEdges){
					return;
				}
			}
		}
	}
	if (_prm.printToScreen) println(format("EdgesBeforeNMS = %d\nEdgesAfterNMS = %d", before, counter));
}

/* 
Removes the key from the hash map data.
It is used to remove edge responses from the hash map of the Beam-Curve Binary-Tree.
Stand alone function.
*/
void Detector::removeKey(unordered_map<uint, Mat>& data, uint key){
	int a1, a2;
	uint rows = _handle.rSize.height, cols = _handle.rSize.width;

	ind2sub(key, rows, cols, a1, a2);
	uint negKey = sub2ind(rows, cols, a2, a1);
	if (data.count(key) == 1){
		data.erase(key);
	}
	if (data.count(negKey)){
		data.erase(negKey);
	}
}

/*
Adds a specific edge to the image E.
Recursive function, calls to itself to produce the edge curve.
Stand alone recursive function.
*/
bool Detector::addEdge(unordered_map<uint, Mat>& data, uint curKey, Mat& E, uint level){
	//int maxLevel = (int)(2*log2(_handle.n)-log2(_prm.patchSize));
	if (data.count(curKey) == 0 || level == _maxLevel+1){
		return false;
	}
	Mat curData = data.at(curKey).clone();
	uint i0s0 = (uint)curData.at<double>(_handle.I0S0);
	uint s0i1 = (uint)curData.at<double>(_handle.S0I1);

	i0s0 = (1 - (curKey == i0s0))*i0s0;
	s0i1 = (1 - (curKey == s0i1))*s0i1;
	if (i0s0 > 0 && s0i1 > 0){
		if (!addEdge(data, (uint)i0s0, E, level + 1)) return false;
		if (!addEdge(data, (uint)s0i1, E, level + 1)) return false;		
	}
	else{
		if (_pixels.count(curKey) == 0){
			println("Pixels Problem");
			return false;
		}
		Mat& pixels = _pixels.at(curKey);
		assert(pixels.isContinuous());
		double* p = (double*)pixels.data;
		bool* e = (bool*)E.data;

		bool flag = false;
		for (int i = 0; i < pixels.size().area(); ++i){
			int curPixel = (int)*p++;
			if (curPixel >= 0){
				flag = true;
				e[curPixel] = true;
			}
		}
		if (!flag){
			return false;
		}
	}
	return true;
}
