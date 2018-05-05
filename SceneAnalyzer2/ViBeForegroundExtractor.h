#pragma once

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <fstream>
#include <sstream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "ViBe.h"

using namespace std;
using namespace cv;

class StaticRectExtractor
{
public:
    void init(const Size& imageSize);
    void extract(const vector<Rect>& rects, vector<Rect>& stableRects);
    void clear(void);

private:
    Size size;

    struct RectInfo
    {
        Rect rect;
        int matchCount;
        int missCount;
        int startFrameCount;

        RectInfo() {};
        RectInfo(const Rect& rectVal, int currFrameCount)
            : rect(rectVal), matchCount(0), missCount(0), startFrameCount(currFrameCount) {}; 
    };
    vector<RectInfo> staticRectInfos;

    int currFrameCount;
    int thresFrameCount;  // startFrameCount 小于 thresFrameCount 的静态矩形不会输出到 stableRects 中
    int maxFrameCount;    // currFrameCount 自增的最大值
};

class ViBeForegroundExtractor
{
public:
    struct State
    {
        enum {BEGIN = 0, NORMAL = 1, ABNORMAL = 2};
    };
    void init(Mat& image, Mat& gradImage, const string& configFilePath);
    int apply(Mat& image, Mat& gradImage, Mat& foregroundImage, vector<Rect>& rects);

private:
    Mat colorForeImage;
    Mat gradForeImage;
    Mat testChangeForeImage;

    ViBe colorBackModel;
    ViBe gradBackModel;

    StaticRectExtractor rectExtractor;
    vector<Rect> staticRects;

    int foreLargeCount;
    int imageWidth, imageHeight;

    double ratioRectWidthLarge, ratioRectHeightLarge;
    int rectWidthForUnion, rectHeightForUnion;
    double ratioUnionRectAreaLarge;
    int foreLargeCountForRefill;
};