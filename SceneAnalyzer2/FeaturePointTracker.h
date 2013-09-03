#pragma once

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <string>
#include <sstream>
#include <list>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

struct LineSegment
{
	cv::Point begPoint, endPoint;
	int angle;
};

void drawLineSegments(cv::Mat& image, const std::vector<LineSegment>& lineSegments, const cv::Scalar& color);

struct FeaturePoint
{
	FeaturePoint(void) 
	{

	};
	FeaturePoint(const cv::KeyPoint& keyPoint, int idx, long long int timeStamp, int frameCount) 
	{
		trainIndex = idx; 
		currKeyPoint = keyPoint; 
		pointHistory.push_back(PointRecord(keyPoint.pt, timeStamp, frameCount));
	};

	int trainIndex;
	cv::KeyPoint currKeyPoint;

	struct PointRecord
	{
		PointRecord(void) {};
		PointRecord(const cv::Point& pt, long long int timeStamp, int frameCount)
			: point(pt), time(timeStamp), count(frameCount) {};

		cv::Point point;
		long long int time;
		int count;
	};
	std::vector<PointRecord> pointHistory;
};

class FeaturePointTracker
{
public:
	~FeaturePointTracker(void);
	void init(const cv::Size& originalSize, const cv::Size& normalSize, int ratio);
	void track(std::vector<cv::KeyPoint>& currKeyPoints, std::vector<cv::DMatch>& matches,
		std::vector<LineSegment>& lineSegments, long long int timeStamp, int frameCount);
	void drawCurrentPositions(cv::Mat& image, const cv::Scalar& color);
	void drawHistories(cv::Mat& image, const cv::Scalar& color);
	// 以下两个函数 运动方向 1 表示 0 度，180 表示 358 度，255 表示运动方向未知
    // 画出特征点的运动方向 anglesImage 不需要预先分配内存
	void drawCurrentDirections(cv::Mat& image, const cv::Scalar& color);	
	// 把特征点的运动方向填充到 image 需要预先分配内存 图片的类型是 CV_8UC1
	void fillCurrentDirections(cv::Mat& image);
	// 给定两个帧号和两个矩形 判定区域内是否存在连通的跟踪点
	bool checkRectsConnection(const cv::Rect& rect1, int frameCount1, const cv::Rect& rect2, int frameCount2);
private:
	cv::Size origSize, normSize;
	int scaleOrigToNorm;
public:
	std::list<FeaturePoint*> pointList;
};

class DirectionLookUpTable
{
public:
	void init(const cv::Size& imageSize, 
		      const cv::Size& blockSize = cv::Size(16, 16), 
			  int numberOfBins = 8);
	void update(const std::vector<LineSegment>& lineSegments);
	bool getMainDirection(const cv::Rect& rect, int& angle);
	void drawMainDirection(cv::Mat& image, const cv::Scalar& color);
	void printMainDirection(void);
	void drawDirections(std::vector<cv::Mat>& images, const cv::Scalar& color);
	void getDirections(cv::Mat* images, int numOfImages);

private:
	// 表示每个像素的运动方向分布情况 每个像素由 numOfBins + 1 个浮点数组成 最后一个浮点数表示
	// 查找表表示的每个点 有 numOfBins 个 bin，起始 bin 以零度为中心，随后的中心沿着逆时针旋转
	cv::Mat lut;    
	int lutWidth, lutHeight;
	int imageWidth, imageHeight;
	int stepX, stepY;
	int numOfBins;
	float learnRate;
	float initWeight;
	int maxCount;
};
