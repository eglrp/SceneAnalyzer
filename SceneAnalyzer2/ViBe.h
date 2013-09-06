#pragma once

#define RUN_VIBE_WITH_BACKGROUND_CONSTRUCTION 0	

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

class RandIntUniform
{
public:
	RandIntUniform();
	~RandIntUniform();
	void init(int size, int minInc, int maxExc, long long int seed);
	void update(long long int seed);
	void release(void);
	inline int getNext(void);

private:
	int minValInc, maxValExc;
	Mat mat;
	int* data;
	int index;
	int capacity;
};

int RandIntUniform::getNext(void)
{
	index++;
	if (index >= capacity || index < 0)
		index = 0;
	return data[index];
}

class ViBe
{
public:
	ViBe();
	~ViBe();

	void init(Mat& image, const string& path = "", const string& signature = "");
	void update(Mat& image, Mat& foregroundImage,
		const vector<Rect>& rectsNoUpdate = vector<Rect>());
#if RUN_VIBE_WITH_BACKGROUND_CONSTRUCTION
	void update(Mat& image, Mat& foregroundImage, Mat& backgroundImage, 
		const vector<Rect>& rectsNoUpdate = vector<Rect>());
#endif
	void refill(Mat& image);
	void release(void);

private:
	void init(void);
	void config(const string& path = "", const string& signature = "");
	void fill8UC3(Mat& image);
	void fill8UC1(Mat& image);
	void update(Mat& image, const vector<Rect>& rectsNoUpdate = vector<Rect>());
	void showSamples(int count);	

	int imageWidth, imageHeight;
	int imageChannels;

	int numOfSamples;                       // 每个像素保存的样本数量
	int minMatchDist;                       // 判定前景背景的距离
	int minNumOfMatchCount;                 // 判定为背景的最小匹配成功次数
	int subSampleInterval;                  // 它的倒数等于更新保存像素值的概率

	Mat foreImage;                          // 前景图
	unsigned char** ptrFore;                // 前景图首行地址
	unsigned char* samples;                 // 保存先前像素值 即样本
	unsigned char** ptrSamples;             // 样本的行首地址

#if RUN_VIBE_WITH_BACKGROUND_CONSTRUCTION
	Mat backImage;                          // 背景图
	float** ptrBack;                        // 背景图首行地址
	float learnRate;                        // 学习速率
	float compLearnRate;
#endif

	Mat noUpdateImage;                      // 不更新的像素
	unsigned char** ptrNoUpdate;            // 行首地址

	RandIntUniform rndReplaceCurr;          // 确定是否更新当前像素的样本
	RandIntUniform rndIndexCurr;            // 确定需替换的样本下标
	RandIntUniform rndReplaceAdj;           // 确定是否更新邻域像素样本
	RandIntUniform rndPositionAdj;          // 确定需要更新的邻域位置
	RandIntUniform rndIndexAdj;             // 确定续替换的样本的下标
	
	static char adjPositions[8][2];
};
