﻿#include "SceneAnalyzer.h"
using namespace std;
using namespace cv;

static void calcThresholdedGradient(Mat& src, Mat& dst, double thres);
static void calcGradient(Mat& src, Mat& dst, double scale);
static inline Rect mul(const Rect& rect, int scale);

int SceneAnalyzer::analyze(Mat& frame, long long int timeStamp, int frameCount,
    Mat& foregroundImage, Mat& mainDirImage, vector<Rect>& foregroundRects, FeaturePointTracker& pointTracker)
{
    if (!hasInit)
    {
        hasInit = true;
        scaleOrigToNorm = frame.cols / 400.0 + 0.5;
        normSize.width = frame.cols / scaleOrigToNorm;
        normSize.height = frame.rows / scaleOrigToNorm;

        resize(frame, image, normSize);
        cvtColor(image, grayImage, CV_BGR2GRAY);
        GaussianBlur(image, blurImage, Size(3, 3), 1);
        GaussianBlur(grayImage, grayBlurImage, Size(3, 3), 1);
        calcGradient(grayBlurImage, gradImage, 0.25);
        foreExtractor.init(blurImage, gradImage, path);
        dirHist.init(normSize, Size(8, 8), 32);
        foregroundImage = Mat::zeros(normSize, CV_8UC1);
        mainDirImage = Mat::zeros(normSize, CV_8UC1);
        foregroundRects.clear();
        pointTracker.init(Size(frame.cols, frame.rows), normSize, scaleOrigToNorm);
        state = State::BEGIN;
        return state;
    }

    resize(frame, image, normSize);
    cvtColor(image, grayImage, CV_BGR2GRAY);
    GaussianBlur(image, blurImage, Size(3, 3), 1);
    GaussianBlur(grayImage, grayBlurImage, Size(3, 3), 1);
    calcGradient(grayBlurImage, gradImage, 0.25);

    int extractorState;
    vector<Rect> normRects;
    extractorState = foreExtractor.apply(blurImage, gradImage, foregroundImage, normRects);
    if (extractorState == ViBeForegroundExtractor::State::BEGIN)
    {
        lastKeyPoints.clear();
        lastDescriptors.release();
        dirHist.clear();
        pointTracker.clear();
        foregroundRects.clear();
        foregroundImage = Mat::zeros(normSize, CV_8UC1);
        mainDirImage = Mat::zeros(normSize, CV_8UC1);
        return State::BEGIN;
    }
    foregroundRects.resize(normRects.size());
    for (int i = 0; i < normRects.size(); i++)
        foregroundRects[i] = mul(normRects[i], scaleOrigToNorm);

    orb(grayImage, foregroundImage, currKeyPoints, currDescriptors);
    if (currDescriptors.data && lastDescriptors.data)
        matcher.match(currDescriptors, lastDescriptors, matches);
    filteredMatches.clear();
    filteredMatches.reserve(matches.size());
    int numOfMatches = matches.size();
    if (numOfMatches > 2)
    {
        vector<float> dist(numOfMatches);
        for (int i = 0; i < numOfMatches; i++)
        {
            const Point2f& queryPoint = currKeyPoints[matches[i].queryIdx].pt;
            const Point2f& trainPoint = lastKeyPoints[matches[i].trainIdx].pt;
            dist[i] = fabs(queryPoint.x - trainPoint.x) + fabs(queryPoint.y - trainPoint.y);
        }
        vector<float> distSorted = dist;
        partial_sort(distSorted.begin(), distSorted.begin() + numOfMatches / 2 + 1, distSorted.end());
        float maxDist = distSorted[numOfMatches / 2] * 2;
        maxDist = maxDist > 30 ? 30 : maxDist;
        maxDist = maxDist < 10 ? 10 : maxDist;
        for (int i = 0; i < numOfMatches; i++)
        {
            if (dist[i] > 0 && dist[i] < maxDist)
                filteredMatches.push_back(matches[i]);
        }
    }
    pointTracker.track(currKeyPoints, filteredMatches, lineSegs, timeStamp, frameCount);
    pointTracker.fillCurrentDirections(foregroundImage);
    dirHist.update(lineSegs);
    dirHist.getMainDirection(mainDirImage);
    lastKeyPoints = currKeyPoints;
    currDescriptors.copyTo(lastDescriptors);

    if (state == State::BEGIN)
    {
        if (extractorState == ViBeForegroundExtractor::State::NORMAL)
            state = State::LEARNING;
        else if (extractorState == ViBeForegroundExtractor::State::ABNORMAL)
            state = State::ABNORMAL;
    }
    else if (state == State::LEARNING)
    {
        if (extractorState == ViBeForegroundExtractor::State::NORMAL)
        {
            if (dirHist.calcUpdatedRatio() < 0.25)
                state = State::LEARNING;
            else
                state = State::NORMAL;
        }
        else if (extractorState == ViBeForegroundExtractor::State::ABNORMAL)
            state = State::ABNORMAL;        
    }
    else
    {
        if (extractorState == ViBeForegroundExtractor::State::NORMAL)
            state = State::NORMAL;
        else if (extractorState == ViBeForegroundExtractor::State::ABNORMAL)
            state = State::ABNORMAL;
    }

    //Mat ftrPoints = Mat::zeros(normSize, CV_8UC1);
    //pointTracker.drawCurrentPositions(ftrPoints, 255);
    //blur(ftrPoints, ftrPoints, Size(11, 11));
    //imshow("feature points region", ftrPoints > 0);
    Mat result = image.clone();
    //pointTracker.drawHistories(result, Scalar(0, 255, 0));
    pointTracker.drawCurrentDirections(result, Scalar(0, 255, 0));
    //for (int i = 0; i < normRects.size(); i++)
    //  rectangle(result, normRects[i], Scalar(0, 0, 255));
    imshow("curr dir", result);
    //waitKey(30);

    //Mat mainDir;
    //dirHist.getMainDirection(mainDir);
    //imshow("main dir image", mainDir);

    //Mat dirs[3];
    //dirHist.getDirections(dirs, 3);
    //imshow("dir image 0", dirs[0]);
    //imshow("dir image 1", dirs[1]);
    //imshow("dir image 2", dirs[2]);

    Mat mainDirection = image.clone();
    dirHist.drawMainDirection(mainDirection, Scalar(0, 255, 0));
    imshow("main dir", mainDirection);

    //Mat directions[3];
    //directions[0] = image.clone();
    //directions[1] = image.clone();
    //directions[2] = image.clone();
    //dirHist.drawDirections(directions, 3, Scalar(0, 255, 0));
    //imshow("direction 0", directions[0]);
    //imshow("direction 1", directions[1]);
    //imshow("direction 2", directions[2]);

    //waitKey(30);

    return state;
}

void calcThresholdedGradient(Mat& src, Mat& dst, double thres)
{
    //static float horiArray[3][3] = {{1, 0, -1}, {sqrt(2.0F), 0, -sqrt(2.0F)}, {1, 0, -1}};
    //static float vertArray[3][3] = {{1, sqrt(2.0F), 1}, {0, 0, 0}, {-1, -sqrt(2.0F), -1}};
    static float horiArray[3][3] = {{3, 0, -3}, {10, 0, -10}, {3, 0, -3}};
    static float vertArray[3][3] = {{3, 10, 3}, {0, 0, 0}, {-3, -10, -3}};
    static Mat horiKernel = Mat(3, 3, CV_32F, horiArray);
    static Mat vertKernel = Mat(3, 3, CV_32F, vertArray);

    Mat horiGrad = Mat::zeros(src.rows, src.cols, CV_32FC1);
    Mat vertGrad = Mat::zeros(src.rows, src.cols, CV_32FC1);
    Mat grad = Mat::zeros(src.rows, src.cols, CV_32FC1);

    filter2D(src, horiGrad, horiGrad.depth(), horiKernel);
    filter2D(src, vertGrad, vertGrad.depth(), vertKernel);
    grad = abs(horiGrad) + abs(vertGrad);
    grad.convertTo(dst, CV_8UC1); 

    unsigned char* ptrDstData = (unsigned char*)dst.data;
    for (int i = 0; i < dst.rows; i++)
    {
        ptrDstData = dst.ptr<unsigned char>(i);
        ptrDstData[0] = 0;
        ptrDstData[dst.cols - 1] = 0;
    }
    ptrDstData = dst.ptr<unsigned char>(0);
    for (int i = 0; i < dst.cols; i++)
    {
        ptrDstData[i] = 0;
    }
    ptrDstData = dst.ptr<unsigned char>(dst.rows - 1);
    for (int i = 0; i < dst.cols; i++)
    {
        ptrDstData[i] = 0;
    }
    for (int i = 0; i < dst.rows; i++)
    {
        ptrDstData = dst.ptr<unsigned char>(i);
        for (int j = 0; j < dst.cols; j++)
        {
            if (ptrDstData[j] > thres)
                ptrDstData[j] = 255;
            else
                ptrDstData[j] = 0;
        }
    }
}

void calcGradient(Mat& src, Mat& dst, double scale)
{
    //static float horiArray[3][3] = {{1, 0, -1}, {sqrt(2.0F), 0, -sqrt(2.0F)}, {1, 0, -1}};
    //static float vertArray[3][3] = {{1, sqrt(2.0F), 1}, {0, 0, 0}, {-1, -sqrt(2.0F), -1}};
    static float horiArray[3][3] = {{3, 0, -3}, {10, 0, -10}, {3, 0, -3}};
    static float vertArray[3][3] = {{3, 10, 3}, {0, 0, 0}, {-3, -10, -3}};
    static Mat horiKernel = Mat(3, 3, CV_32F, horiArray);
    static Mat vertKernel = Mat(3, 3, CV_32F, vertArray);
    
    Mat horiGrad = Mat::zeros(src.rows, src.cols, CV_32FC1);
    Mat vertGrad = Mat::zeros(src.rows, src.cols, CV_32FC1);
    Mat grad = Mat::zeros(src.rows, src.cols, CV_32FC1);

    filter2D(src, horiGrad, horiGrad.depth(), horiKernel);
    filter2D(src, vertGrad, vertGrad.depth(), vertKernel);
    grad = abs(horiGrad) + abs(vertGrad);
    if (scale != 1.0) grad *= scale;
    grad.convertTo(dst, CV_8UC1); 

    unsigned char* ptrDstData = (unsigned char*)dst.data;
    for (int i = 0; i < dst.rows; i++)
    {
        ptrDstData = dst.ptr<unsigned char>(i);
        ptrDstData[0] = 0;
        ptrDstData[dst.cols - 1] = 0;
    }
    ptrDstData = dst.ptr<unsigned char>(0);
    for (int i = 0; i < dst.cols; i++)
    {
        ptrDstData[i] = 0;
    }
    ptrDstData = dst.ptr<unsigned char>(dst.rows - 1);
    for (int i = 0; i < dst.cols; i++)
    {
        ptrDstData[i] = 0;
    }
}

Rect mul(const Rect& rect, int scale)
{
    return Rect(rect.x * scale, rect.y * scale, rect.width * scale, rect.height * scale);
}