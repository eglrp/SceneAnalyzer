#include <iostream>
#include <string>
#include <vector>
#include <cstdio>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "SceneAnalyzer.h"

using namespace std;
using namespace cv;

int main(void)
{
    string path = "D:/SHARED/BeijingVideo/B9-2_�ϳ��֣�81��_20130814100000_20130814103000_6733890.mp4"
        /*"D:/SHARED/TaicangVideo/11/72(00h11m41s-00h15m50s).flv"*/
        /*"D:/SHARED/beijingVideo/B9-1_���巹�꣨98��_20130814113000_20130814114733_5544859.mp4"*/
        /*"d:/shared/taicangvideo/1/70.flv"*/;
    SceneAnalyzer analyzer;
    FeaturePointTracker tracker;
    //Mat image, foreImage,mainDirImage;
    //vector<Rect> rects;
    long long int beg, end;
    double freq = getTickFrequency();
    double timeElapsed;
    VideoCapture cap;
    cap.open(path);
    if (!cap.isOpened())
    {
        printf("cannot open file %s\n", path.c_str());
        return 0;
    }
    while (true)
    {
        Mat image, foreImage,mainDirImage;
        vector<Rect> rects;
        
        long long int currTime = cap.get(CV_CAP_PROP_POS_MSEC);
        int currCount = cap.get(CV_CAP_PROP_POS_FRAMES);
        if (!cap.read(image))
            break;

        beg = getTickCount();
        int state = analyzer.analyze(image, currTime, currCount, foreImage, mainDirImage, rects, tracker);
        end = getTickCount();
        timeElapsed = double(end - beg) / freq;
        printf("frame count: %d, time elapsed: %.4f, ", currCount, timeElapsed);
        switch (state)
        {
        case SceneAnalyzer::State::BEGIN :      // ģ�����³�ʼ�� �����½�ʶ������ 
            {
                printf("begin\n");
                break;
            }
        case SceneAnalyzer::State::LEARNING :   // ������ģ���ȶ����������ѧϰ�׶� �����½�ʶ������
            {
                printf("learning\n");
                break;
            }
        case SceneAnalyzer::State::NORMAL :     // ����״̬ �����½�ʶ������
            {
                printf("normal\n");
                break;
            }
        case SceneAnalyzer::State::ABNORMAL :   // �쳣״̬ ���ܴ�������ͷת�� �����½�ʶ������
            {
                printf("abnormal\n");
                break;
            }
        }
        Mat result = image.clone();
        for (int i = 0; i < rects.size(); i++)
            rectangle(result, rects[i], Scalar(0, 0, 255));
        imshow("result", result);
        imshow("foreground", foreImage);
        imshow("main direction", mainDirImage);
        waitKey(15);
    }
    cap.release();
}