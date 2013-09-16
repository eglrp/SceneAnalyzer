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
	void clear(void);
	void drawCurrentPositions(cv::Mat& image, const cv::Scalar& color);
	void drawHistories(cv::Mat& image, const cv::Scalar& color);
	// 以下两个函数 运动方向 1 表示 0 度，180 表示 358 度，255 表示运动方向未知
    // 画出特征点的运动方向 anglesImage 需要预先分配内存
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

typedef std::pair<int, float> IndexedFloat;

inline bool greaterByValue(const IndexedFloat& lhs, const IndexedFloat& rhs)
{
	return lhs.second > rhs.second;
}

class LocalHistogram
{
public:
	void init(const cv::Size& size, int numberOfBins)
	{
		histWidth = size.width;
		histHeight = size.height;
		numOfBins = numberOfBins;
		learnRate = 0.01F;
		maxCount = 1.0F / learnRate; 
		hist = cv::Mat::zeros(histHeight, histWidth * (numOfBins + 1), CV_32FC1);
	};
	void update(int x, int y, int binIndex)
	{
		float* ptrHist = &hist.at<float>(y, x * (1 + numOfBins));
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
	};
	double calcUpdatedRatio(void)
	{
		int updateCount = 0;
		for (int i = 0; i < histHeight; i++)
		{
			float* ptrHist = hist.ptr<float>(i);
			for (int j = 0; j < histWidth; j++)
			{
				if (ptrHist[numOfBins] > 0)
					updateCount++;
				ptrHist += (1 + numOfBins);
			}
		}
		return double(updateCount) / (histWidth * histHeight);
	}
	void clear(void)
	{
		hist.setTo(0);
	};
	void getLocalMaximun(std::vector<int>& data)
	{
		data.resize(histWidth * histHeight, numOfBins);
		for (int i = 0; i < histHeight; i++)
		{
			float* ptrHist = hist.ptr<float>(i);
			int* ptrData = &data[i * histWidth];
			for (int j = 0; j < histWidth; j++)
			{
				if (ptrHist[numOfBins] > 0)
				{
					int mainIndex;
					float currMaxRatio = -1.0F;
					for (int k = 0; k < numOfBins; k++)
					{
						if (ptrHist[k] > currMaxRatio)
						{
							currMaxRatio = ptrHist[k];
							mainIndex = k;
						}
					}
					ptrData[j] = mainIndex;
				}
				ptrHist += (numOfBins + 1);
			}
		}
	};
	void getLocalMaximun(std::vector<IndexedFloat>& data)
	{
		data.resize(histWidth * histHeight, IndexedFloat(numOfBins, 0));
		for (int i = 0; i < histHeight; i++)
		{
			float* ptrHist = hist.ptr<float>(i);
			IndexedFloat* ptrData = &data[i * histWidth];
			for (int j = 0; j < histWidth; j++)
			{
				if (ptrHist[numOfBins] > 0)
				{
					int mainIndex;
					float currMaxRatio = -1.0F;
					for (int k = 0; k < numOfBins; k++)
					{
						if (ptrHist[k] > currMaxRatio)
						{
							currMaxRatio = ptrHist[k];
							mainIndex = k;
						}
					}
					ptrData[j].first = mainIndex;
					ptrData[j].second = currMaxRatio;
				}
				ptrHist += (numOfBins + 1);
			}
		}
	};
	void getLocalMaxima(std::vector<std::vector<int> >& data, int numOfPlanes)
	{
		data.resize(numOfPlanes);
		for (int k = 0; k < numOfPlanes; k++)
			data[k].resize(histWidth * histHeight, numOfBins);
		for (int i = 0; i < histHeight; i++)
		{
			float* ptrHist = hist.ptr<float>(i);
			std::vector<int*> ptrData(numOfPlanes);
			for (int k = 0; k < numOfPlanes; k++)
				ptrData[k] = &data[k][i * histWidth];
			for (int j = 0; j < histWidth; j++)
			{
				if (ptrHist[numOfBins] > 0)
				{
					std::vector<IndexedFloat> pairIndexWeights(numOfBins);
					for (int k = 0; k < numOfBins; k++)
					{
						pairIndexWeights[k].first = k;
						pairIndexWeights[k].second = ptrHist[k];
					}
					partial_sort(pairIndexWeights.begin(), pairIndexWeights.begin() + numOfPlanes, pairIndexWeights.end(), greaterByValue);
					for (int k = 0; k < numOfPlanes; k++)
					{
						if (pairIndexWeights[k].second == 0)
							continue;
						ptrData[k][j] = pairIndexWeights[k].first;
					}				
				}
				ptrHist += (numOfBins + 1);
			}
		}
	};
	void getLocalMaxima(std::vector<std::vector<IndexedFloat> >& data, int numOfPlanes)
	{
		data.resize(numOfPlanes);
		for (int k = 0; k < numOfPlanes; k++)
			data[k].resize(histWidth * histHeight, IndexedFloat(numOfBins, 0));
		for (int i = 0; i < histHeight; i++)
		{
			float* ptrHist = hist.ptr<float>(i);
			std::vector<IndexedFloat*> ptrData(numOfPlanes);
			for (int k = 0; k < numOfPlanes; k++)
				ptrData[k] = &data[k][i * histWidth];
			for (int j = 0; j < histWidth; j++)
			{
				if (ptrHist[numOfBins] > 0)
				{
					std::vector<IndexedFloat> pairIndexWeights(numOfBins);
					for (int k = 0; k < numOfBins; k++)
					{
						pairIndexWeights[k].first = k;
						pairIndexWeights[k].second = ptrHist[k];
					}
					partial_sort(pairIndexWeights.begin(), pairIndexWeights.begin() + numOfPlanes, pairIndexWeights.end(), greaterByValue);
					for (int k = 0; k < numOfPlanes; k++)
					{
						if (pairIndexWeights[k].second == 0)
							continue;
						ptrData[k][j] = pairIndexWeights[k];
					}				
				}
				ptrHist += (numOfBins + 1);
			}
		}
	};
private:
	int histWidth, histHeight;
	int numOfBins;
	float learnRate;
	int maxCount;
	cv::Mat hist;
};

class LocalDirectionHistogram
{
public:
	void init(const cv::Size& imageSize, 
		      const cv::Size& blockSize = cv::Size(16, 16), 
			  int numberOfBins = 8);
	void update(const std::vector<LineSegment>& lineSegments);
	double calcUpdatedRatio(void);
	void clear(void);
	void getMainDirection(cv::Mat& image);
	void drawMainDirection(cv::Mat& image, const cv::Scalar& color);
	void getDirections(cv::Mat* images, int numOfImages);
	void drawDirections(cv::Mat* images, int numOfImages, const cv::Scalar& color);	

private:
	LocalHistogram hist;
	int histWidth, histHeight;
	int imageWidth, imageHeight;
	int stepX, stepY;
	int numOfBins;
};
