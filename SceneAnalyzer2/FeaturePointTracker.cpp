#include <cmath>
#include <algorithm>
#include "FeaturePointTracker.h"

using namespace std;
using namespace cv;

typedef list<FeaturePoint*>::iterator FtrPtPtrItr;
static const double mathPi = asin(1.0) * 2;

void drawLineSegments(Mat& image, const vector<LineSegment>& lineSegments, const Scalar& color)
{
	for (int i = 0; i < lineSegments.size(); i++)
	{
		line(image, lineSegments[i].begPoint, lineSegments[i].endPoint, color);
		circle(image, lineSegments[i].begPoint, 3, color);
	}
}

FeaturePointTracker::~FeaturePointTracker(void)
{
	for (FtrPtPtrItr itr = pointList.begin(); itr != pointList.end(); ++itr)
	{
		delete (*itr);
	}
	pointList.clear();
}

void FeaturePointTracker::init(const cv::Size& originalSize, const cv::Size& normalSize, int ratio)
{
	origSize = originalSize;
	normSize = normalSize;
	scaleOrigToNorm = ratio;
	pointList.clear();
}

void FeaturePointTracker::track(std::vector<cv::KeyPoint>& currKeyPoints, std::vector<cv::DMatch>& matches,
	std::vector<LineSegment>& lineSegments, long long int timeStamp, int frameCount)
{
	lineSegments.clear();
	
	//if (!pointList.empty())
	//	printf("num of tracked points: %d\n", (int)distance(pointList.begin(), pointList.end()));
	int numOfCurrKeyPoints = currKeyPoints.size();

	// 如果当前帧检测不到特征点 将 pointList 中的点都删除
	if (numOfCurrKeyPoints == 0)
	{
		for (FtrPtPtrItr itr = pointList.begin(); itr != pointList.end(); ++itr)
		{
			delete (*itr);
		}
		pointList.clear();
		return;
	}
	
	// 如果当前帧 pointList 为空 则将 currKeyPoints 都填入 pointList 中
	if (pointList.empty())
	{		
		pointList.resize(numOfCurrKeyPoints);
		FtrPtPtrItr itr = pointList.begin();
		for (int i = 0; i < numOfCurrKeyPoints; i++)
		{
			(*itr) = new FeaturePoint(currKeyPoints[i], i, timeStamp, frameCount);
			++itr;
		}
		return;
	}

	vector<unsigned char> isKeyPointMatch(numOfCurrKeyPoints, 0);
	int numOfMatches = matches.size();
	for (FtrPtPtrItr itr = pointList.begin(); itr != pointList.end();)
	{
		int index = -1;
		// 找当前 FeaturePoint 在当前 matches 中的 trainIndex
		for (int i = 0; i < numOfMatches; i++)
		{
			if (matches[i].trainIdx == (*itr)->trainIndex)
			{
				index = i;
				break;
			}
		}

		if (index >= 0)
		{
			int currKeyPointIndex = matches[index].queryIdx;
			isKeyPointMatch[currKeyPointIndex] = 1;
			if ((*itr)->pointHistory.size() > 1000)
			{
				delete (*itr);
				(*itr) = new FeaturePoint(currKeyPoints[currKeyPointIndex], currKeyPointIndex, timeStamp, frameCount);
			}
			else
			{
				(*itr)->trainIndex = currKeyPointIndex;
				(*itr)->currKeyPoint = currKeyPoints[currKeyPointIndex];
				(*itr)->pointHistory.push_back(FeaturePoint::PointRecord(currKeyPoints[currKeyPointIndex].pt, timeStamp, frameCount));
			}
			++itr;
		}
		else
		{
			if ((*itr)->pointHistory.size() > 3)
			{
				int diffX = (*itr)->pointHistory.back().point.x - (*itr)->pointHistory.front().point.x;
				int diffY = (*itr)->pointHistory.back().point.y - (*itr)->pointHistory.front().point.y;
				if (abs(diffX) > 4 || abs(diffY) > 4)
				{
					LineSegment lineSeg;
					lineSeg.begPoint = (*itr)->pointHistory.front().point;
					lineSeg.endPoint = (*itr)->pointHistory.back().point;
					if (diffX == 0 && diffY > 0)
						lineSeg.angle = 90;
					else if (diffX == 0 && diffY < 0)
						lineSeg.angle = -90;
					else
						lineSeg.angle = atan2(double(diffY) , double(diffX)) * 180 / mathPi;
					if (lineSeg.angle < 0)
						lineSeg.angle += 360;
					lineSegments.push_back(lineSeg);
				}
			}
			delete (*itr);
			itr = pointList.erase(itr);
		}
	}

	// 处理没有能够被匹配的 currKeyPoints
	for (int i = 0; i < numOfCurrKeyPoints; i++)
	{
		if (!isKeyPointMatch[i])
		{
			pointList.push_back(new FeaturePoint(currKeyPoints[i], i, timeStamp, frameCount));
		}
	}
}

void FeaturePointTracker::drawCurrentPositions(cv::Mat& image, const cv::Scalar& color)
{
	if (image.size() != normSize || image.data == 0)
		return;

	if (image.type() == CV_8UC3)
	{
		for (FtrPtPtrItr itr = pointList.begin(); itr != pointList.end(); ++itr)
		{
			image.at<Vec3b>((*itr)->currKeyPoint.pt) = Vec3b(color[0], color[1], color[2]);
		}
	}
	else if (image.type() == CV_8UC1)
	{
		for (FtrPtPtrItr itr = pointList.begin(); itr != pointList.end(); ++itr)
		{
			image.at<unsigned char>((*itr)->currKeyPoint.pt) = color[0];
		}
	}
}

void FeaturePointTracker::drawHistories(cv::Mat& image, const cv::Scalar& color)
{
	for (FtrPtPtrItr itr = pointList.begin(); itr != pointList.end(); ++itr)
	{
		const vector<FeaturePoint::PointRecord>& refPointHistory = (*itr)->pointHistory;
		if (refPointHistory.size() > 1)
		{
			for (int i = 0; i < refPointHistory.size() - 1; i++)
			{
				line(image, refPointHistory[i].point, refPointHistory[i + 1].point, color);
			}
		}
	}
}

void FeaturePointTracker::drawCurrentDirections(cv::Mat& image, const cv::Scalar& color)
{
	for (FtrPtPtrItr itr = pointList.begin(); itr != pointList.end(); ++itr)
	{
		const vector<FeaturePoint::PointRecord>& refPointHistory = (*itr)->pointHistory;
		if (refPointHistory.size() > 1)
		{
			for (int i = 0; i < refPointHistory.size() - 1; i++)
			{
				line(image, refPointHistory.front().point, refPointHistory.back().point, color);
				circle(image, refPointHistory.front().point, 2, color);
			}
		}
	}
}

void FeaturePointTracker::fillCurrentDirections(cv::Mat& image)
{
	for (FtrPtPtrItr itr = pointList.begin(); itr != pointList.end(); ++itr)
	{
		if ((*itr)->pointHistory.size() > 3)
		{
			int diffX = (*itr)->pointHistory.back().point.x - (*itr)->pointHistory.front().point.x;
			int diffY = (*itr)->pointHistory.back().point.y - (*itr)->pointHistory.front().point.y;
			int currAngle;
			if (diffX == 0 && diffY > 0)
				currAngle = 90;
			else if (diffX == 0 && diffY < 0)
				currAngle = -90;
			else
				currAngle = atan2(double(diffY) , double(diffX)) * 180 / mathPi;
			if (currAngle < 0)
				currAngle += 360;
			currAngle /= 2;
			//anglesImage.at<unsigned char>((*itr)->pointHistory.back().point) = currAngle;
			circle(image, (*itr)->pointHistory.back().point, 4, currAngle + 1, -1);
		}
	}
}

static inline Rect div(const Rect& rect, int scale)
{
	return Rect(rect.x / scale, rect.y / scale, rect.width / scale, rect.height / scale);
}

bool FeaturePointTracker::checkRectsConnection(const cv::Rect& rect1, int frameCount1, const cv::Rect& rect2, int frameCount2)
{
	Rect normRect1 = div(rect1, scaleOrigToNorm), normRect2 = div(rect2, scaleOrigToNorm); 
	for (FtrPtPtrItr itr = pointList.begin(); itr != pointList.end(); ++itr)
	{
		const vector<FeaturePoint::PointRecord>& refPointHistory = (*itr)->pointHistory;
		bool hit1 = false, hit2 = false;
		for (int i = 0; i < refPointHistory.size(); i++)
		{
			if (refPointHistory[i].count == frameCount1 &&
				normRect1.contains(refPointHistory[i].point))
				hit1 = true;
			if (refPointHistory[i].count == frameCount2 &&
				normRect2.contains(refPointHistory[i].point))
				hit2 = true;
			if (hit1 && hit2)
				return true;
		}
	}
	return false;
}

void DirectionLookUpTable::init(const Size& imageSize, const Size& blockSize, int numberOfBins)
{
	learnRate = 0.01;
	initWeight = 0.05;
	maxCount = 1.0F / learnRate;
	imageWidth = imageSize.width;
	imageHeight = imageSize.height;
	stepX = blockSize.width;
	stepY = blockSize.height;
	numOfBins = numberOfBins;
	lutHeight = imageSize.height / stepY;
	lutWidth = imageSize.width / stepX;
	lut = Mat::zeros(lutHeight, lutWidth * (1 + numOfBins), CV_32FC1); 
}

static inline int getBinIndex(int angle, int numOfBins)
{
	return int(double(angle) / 360.0 * numOfBins) % numOfBins;
}

static void getLUTPointsCrspndToLineSegment(const LineSegment& lineSegment, vector<Point>& points, int stepX, int stepY)
{
	points.clear();
	// Bresenham 算法扫描直线上的点
	// http://rosettacode.org/wiki/Bitmap/Bresenham's_line_algorithm
	int x0 = lineSegment.begPoint.x / stepX;
	int y0 = lineSegment.begPoint.y / stepY;
	int x1 = lineSegment.endPoint.x / stepX;
	int y1 = lineSegment.endPoint.y / stepY;
	int dx = abs(x0 - x1);
	int dy = abs(y0 - y1);
	int sx = x0 < x1 ? 1 : -1;
	int sy = y0 < y1 ? 1 : -1;
	int err = (dx > dy ? dx : -dy) / 2;
	int e2;
	while (true)
	{
		points.push_back(Point(x0, y0));
		if (x0 == x1 && y0 == y1)
			break;
		e2 = err;
		if (e2 > -dx)
		{
			err -= dy;
			x0 += sx;
		}
		if (e2 < dy)
		{
			err += dx;
			y0 += sy;
		}
	}
}

void DirectionLookUpTable::update(const vector<LineSegment>& lineSegments)
{	
	for (int i = 0; i < lineSegments.size(); i++)
	{
		// 根据 angle 值确定应该修改哪个 bin
		int binIndex = getBinIndex(lineSegments[i].angle, numOfBins);
		vector<Point> points;
		// 根据起始点和终止点计算 LUT 中哪些位置的元素的直方图需要修改
		getLUTPointsCrspndToLineSegment(lineSegments[i], points, stepX, stepY);
		for (int j = 0; j < points.size(); j++)
		{
			float* ptrHist = &lut.at<float>(points[j].y, points[j].x * (1 + numOfBins));
			if (ptrHist[numOfBins] < maxCount)
			{
				ptrHist[numOfBins] += 1;
				ptrHist[binIndex] += 1.0F / ptrHist[numOfBins];
			}
			else
			{
				ptrHist[binIndex] += learnRate;
			}
			float acc = 0;
			for (int k = 0; k < numOfBins; k++)
			{
				acc += ptrHist[k];
			}
			for (int k = 0; k < numOfBins; k++)
			{
				ptrHist[k] /= acc;
			}
		}
	}
}

bool DirectionLookUpTable::getMainDirection(const Rect& imageRegion, int& angle)
{
	int accAngle = 0;
	int count = 0;
	int begX = imageRegion.x / stepX - 1; 
	int endX = (imageRegion.x + imageRegion.width) / stepX + 1;
	int begY = imageRegion.y / stepY - 1;
	int endY = (imageRegion.y + imageRegion.height) / stepY + 1;
	if (begX < 0) begX = 0;
	if (endX > lutWidth) endX = lutWidth;
	if (begY < 0) begY = 0;
	if (endY > lutHeight) endY = lutHeight;
	for (int i = begY; i < endY; i++)
	{
		float* ptr = lut.ptr<float>(i);
		for (int j = begX; j < endX; j++)
		{
			if (ptr[numOfBins] > 0)
			{
				int mainDirIndex;
				float currMaxRatio = -1.0F;
				for (int k = 0; k < numOfBins; k++)
				{
					if (ptr[k] > currMaxRatio)
					{
						currMaxRatio = ptr[k];
						mainDirIndex = k;
					}
				}
				accAngle += float(mainDirIndex) / numOfBins * 360.0F;
				count++;
			}
			ptr += (numOfBins + 1);
		}
	}
	if (count == 0)
	{
		angle = -1;
		return false;
	}
	else
	{
		angle = double(accAngle) / count;
		return true;
	}
}

void DirectionLookUpTable::drawMainDirection(Mat& image, const Scalar& color)
{
	const static int radius = 4;
	for (int i = 0; i < lutHeight; i++)
	{
		float* ptr = lut.ptr<float>(i);
		for (int j = 0; j < lutWidth; j++)
		{
			if (ptr[numOfBins] > 0)
			{
				int mainDirIndex;
				float currMaxRatio = -1.0F;
				for (int k = 0; k < numOfBins; k++)
				{
					if (ptr[k] > currMaxRatio)
					{
						currMaxRatio = ptr[k];
						mainDirIndex = k;
					}
				}
				double angle = mathPi * 2 * float(mainDirIndex) / numOfBins;
				Point beg, end, center;
				center.x = (j + 0.5F) * stepX;
				center.y = (i + 0.5F) * stepY;
				beg.x = center.x - radius * cos(angle);
				beg.y = center.y - radius * sin(angle);
				end.x = center.x + radius * cos(angle);
				end.y = center.y + radius * sin(angle);
				line(image, beg, end, color);
				circle(image, beg, 1, color);
			}
			ptr += (numOfBins + 1);
		}
	}
}

void DirectionLookUpTable::printMainDirection(void)
{
	printf("print main direction:-------------------------------");
	for (int i = 0; i < lutHeight; i++)
	{
		float* ptr = lut.ptr<float>(i);
		for (int j = 0; j < lutWidth; j++)
		{
			if (ptr[numOfBins] > 0)
			{
				int mainDirIndex;
				float currMaxRatio = -1.0F;
				for (int k = 0; k < numOfBins; k++)
				{
					if (ptr[k] > currMaxRatio)
					{
						currMaxRatio = ptr[k];
						mainDirIndex = k;
					}
				}
				int angle = 360 * float(mainDirIndex) / numOfBins;
				printf("%4d", angle);
			}
			else
				printf("    ");
			ptr += (numOfBins + 1);
		}
		printf("\n");
	}
	printf("end-------------------------------------------------\n");
}

typedef pair<int, float> PairIndexWeight;

static inline bool greaterByWeight(const PairIndexWeight& lhs, const PairIndexWeight& rhs)
{
	return lhs.second > rhs.second;
}

static inline Scalar operator*(const Scalar& scalar, float r)
{
	return Scalar(scalar[0] * r, scalar[1] * r, scalar[2] * r, scalar[3] * r);
}

void DirectionLookUpTable::drawDirections(vector<Mat>& images, const Scalar& color)
{
	const static int radius = 4;
	int numOfImages = images.size();
	for (int i = 0; i < lutHeight; i++)
	{
		float* ptr = lut.ptr<float>(i);
		for (int j = 0; j < lutWidth; j++)
		{
			if (ptr[numOfBins] > 0)
			{
				vector<PairIndexWeight> pairIndexWeights(numOfBins);
				for (int k = 0; k < numOfBins; k++)
				{
					pairIndexWeights[k].first = k;
					pairIndexWeights[k].second = ptr[k];
				}
				partial_sort(pairIndexWeights.begin(), pairIndexWeights.begin() + numOfImages, pairIndexWeights.end(), greaterByWeight);
				for (int k = 0; k < numOfImages; k++)
				{
					if (pairIndexWeights[k].second == 0)
						continue;
					double angle = mathPi * 2 * float(pairIndexWeights[k].first) / numOfBins;
					Point beg, end, center;
					center.x = (j + 0.5F) * stepX;
					center.y = (i + 0.5F) * stepY;
					beg.x = center.x - radius * cos(angle);
					beg.y = center.y - radius * sin(angle);
					end.x = center.x + radius * cos(angle);
					end.y = center.y + radius * sin(angle);
					line(images[k], beg, end, color * pairIndexWeights[k].second);
					circle(images[k], beg, 2, color * pairIndexWeights[k].second);
				}				
			}
			ptr += (numOfBins + 1);
		}
	}
}

void DirectionLookUpTable::getDirections(cv::Mat* images, int numOfImages)
{
	vector<vector<unsigned char> > angles(numOfImages);	
	for (int k = 0; k < numOfImages; k++)
	{
		images[k].create(imageHeight, imageWidth, CV_8UC1);
		angles[k].resize(lutHeight * lutWidth, 255);
	}
		
	for (int i = 0; i < lutHeight; i++)
	{
		float* ptr = lut.ptr<float>(i);
		vector<unsigned char*> ptrAngles(numOfImages);
		for (int k = 0; k < numOfImages; k++)
			ptrAngles[k] = &(angles[k][i * lutWidth]);
		for (int j = 0; j < lutWidth; j++)
		{
			if (ptr[numOfBins] > 0)
			{
				vector<PairIndexWeight> pairIndexWeights(numOfBins);
				for (int k = 0; k < numOfBins; k++)
				{
					pairIndexWeights[k].first = k;
					pairIndexWeights[k].second = ptr[k];
				}
				partial_sort(pairIndexWeights.begin(), pairIndexWeights.begin() + numOfImages, pairIndexWeights.end(), greaterByWeight);
				for (int k = 0; k < numOfImages; k++)
				{
					if (pairIndexWeights[k].second != 0)
						ptrAngles[k][j] = 180 * float(pairIndexWeights[k].first) / numOfBins;
				}				
			}
			ptr += (numOfBins + 1);
		}
	}

	for (int k = 0; k < numOfImages; k++)
	{
		for (int i = 0; i < imageHeight; i++)
		{
			unsigned char* ptrAngle = &(angles[k][i / stepY * lutWidth]);
			unsigned char* ptrImage = images[k].ptr<unsigned char>(i);
			for (int j = 0; j < imageWidth; j++)
				ptrImage[j] = ptrAngle[j / stepX];
		}
	}
}

void LocalDirectionHistogram::init(const Size& imageSize, const Size& blockSize, int numberOfBins)
{
	imageWidth = imageSize.width;
	imageHeight = imageSize.height;
	stepX = blockSize.width;
	stepY = blockSize.height;
	histHeight = imageSize.height / stepY;
	histWidth = imageSize.width / stepX;
	numOfBins = numberOfBins;
	hist.init(Size(histWidth, histHeight), numberOfBins);
}

void LocalDirectionHistogram::update(const vector<LineSegment>& lineSegments)
{	
	for (int i = 0; i < lineSegments.size(); i++)
	{
		// 根据 angle 值确定应该修改哪个 bin
		int binIndex = getBinIndex(lineSegments[i].angle, numOfBins);
		vector<Point> points;
		// 根据起始点和终止点计算 LUT 中哪些位置的元素的直方图需要修改
		getLUTPointsCrspndToLineSegment(lineSegments[i], points, stepX, stepY);
		for (int j = 0; j < points.size(); j++)
		{
			hist.update(points[j].x, points[j].y, binIndex);
		}
	}
}

void LocalDirectionHistogram::getMainDirection(Mat& image)
{
	vector<int> angles;
	hist.getLocalMaximun(angles);
	for (int i = 0; i < histWidth * histHeight; i++)
		angles[i] = (angles[i] == numOfBins ? 255 : 180 * float(angles[i]) / numOfBins);
	image.create(imageHeight, imageWidth, CV_8UC1);

	for (int i = 0; i < imageHeight; i++)
	{
		int* ptrAngle = &(angles[i / stepY * histWidth]);
		unsigned char* ptrImage = image.ptr<unsigned char>(i);
		for (int j = 0; j < imageWidth; j++)
			ptrImage[j] = ptrAngle[j / stepX];
	}
}

void LocalDirectionHistogram::drawMainDirection(Mat& image, const Scalar& color)
{
	const static int radius = 4;
	vector<IndexedFloat> data;
	hist.getLocalMaximun(data);
	for (int i = 0; i < histHeight; i++)
	{
		IndexedFloat* ptrData = &data[i * histWidth];
		for (int j = 0; j < histWidth; j++)
		{
			if (ptrData[j].first < numOfBins)
			{
				double angle = mathPi * 2 * float(ptrData[j].first) / numOfBins;
				Point beg, end, center;
				center.x = (j + 0.5F) * stepX;
				center.y = (i + 0.5F) * stepY;
				beg.x = center.x - radius * cos(angle);
				beg.y = center.y - radius * sin(angle);
				end.x = center.x + radius * cos(angle);
				end.y = center.y + radius * sin(angle);
				line(image, beg, end, color * ptrData[j].second);
				circle(image, beg, 1, color * ptrData[j].second);
			}
		}
	}
}

void LocalDirectionHistogram::getDirections(cv::Mat* images, int numOfImages)
{
	vector<vector<int> > angles;
	hist.getLocalMaxima(angles, numOfImages);
	for (int k = 0; k < numOfImages; k++)
	{
		for (int i = 0; i < histWidth * histHeight; i++)
		{
			angles[k][i] = (angles[k][i] == numOfBins ? 255 : 180 * float(angles[k][i]) / numOfBins);
		}
	}
	for (int k = 0; k < numOfImages; k++)
	{
		images[k].create(imageHeight, imageWidth, CV_8UC1);
	}	

	for (int k = 0; k < numOfImages; k++)
	{
		for (int i = 0; i < imageHeight; i++)
		{
			int* ptrAngle = &(angles[k][i / stepY * histWidth]);
			unsigned char* ptrImage = images[k].ptr<unsigned char>(i);
			for (int j = 0; j < imageWidth; j++)
				ptrImage[j] = ptrAngle[j / stepX];
		}
	}
}

void LocalDirectionHistogram::drawDirections(cv::Mat* images, int numOfImages, const Scalar& color)
{
	const static int radius = 4;
	vector<vector<IndexedFloat> > data;
	hist.getLocalMaxima(data, numOfImages);
	for (int k = 0; k < numOfImages; k++)
	{		
		for (int i = 0; i < histHeight; i++)
		{
			IndexedFloat* ptrData = &data[k][i * histWidth];
			for (int j = 0; j < histWidth; j++)
			{
				if (ptrData[j].first < numOfBins)
				{					
					double angle = mathPi * 2 * float(ptrData[j].first) / numOfBins;
					Point beg, end, center;
					center.x = (j + 0.5F) * stepX;
					center.y = (i + 0.5F) * stepY;
					beg.x = center.x - radius * cos(angle);
					beg.y = center.y - radius * sin(angle);
					end.x = center.x + radius * cos(angle);
					end.y = center.y + radius * sin(angle);
					line(images[k], beg, end, color * ptrData[j].second);
					circle(images[k], beg, 2, color * ptrData[j].second);				
				}
			}
		}
	}
}