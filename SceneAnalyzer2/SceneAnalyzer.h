#pragma once

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <string>
#include <sstream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "ObjectInfo.h"
#include "ViBeForegroundExtractor.h"
#include "FeaturePointTracker.h"

class SceneAnalyzer
{
public:
    SceneAnalyzer(const string& configFilePath = "param/ConfigSceneAnalyzer.txt")
        : path(configFilePath),
          hasInit(false),
          orb(500, 1.41F, 2, 15, 0, 2, 0, 15), 
          matcher(NORM_HAMMING, true) {};
    struct State
    {
        enum {BEGIN = 0, LEARNING = 1, NORMAL = 2, ABNORMAL = 3};
    };
    int analyze(cv::Mat& frame, 
                long long int timeStamp, int frameCount,
                cv::Mat& foregroundImage, 
                cv::Mat& mainDirectionImage,
                std::vector<Rect>& foreRects,
                FeaturePointTracker& pointTracker);
    int analyze(DetectTask& detectTask, FeaturePointTracker& pointTracker)
    {
        return analyze(detectTask.fpkg.imSrc,
                       detectTask.fpkg.rawtime,
                       detectTask.fpkg.frameIndex,
                       detectTask.fpkg.imMask,
                       detectTask.mainFLow,
                       detectTask.recs,
                       pointTracker);
    };
private:
    string path;
    bool hasInit;
    int state;
    cv::Size origSize;
    cv::Size normSize;
    int scaleOrigToNorm;
    cv::Mat image, blurImage;
    cv::Mat grayImage, grayBlurImage;
    cv::Mat gradImage;
    cv::ORB orb;
    cv::BFMatcher matcher;
    std::vector<cv::DMatch> matches, filteredMatches;
    std::vector<LineSegment> lineSegs;
    std::vector<cv::KeyPoint> currKeyPoints, lastKeyPoints;
    cv::Mat currDescriptors, lastDescriptors;
    ViBeForegroundExtractor foreExtractor;
    LocalDirectionHistogram dirHist;
};
