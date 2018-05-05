#ifndef _OBJECTINFO_H__
#define _OBJECTINFO_H__

#include <deque>

#include "opencv2/opencv.hpp"
//#include "pthread.h"

#include "FeaturePointTracker.h"

using namespace cv;
using namespace std;




/************************************************************************/
// Frame Package ����ṹ���壬Ҳ�Ǽ�¼֡��Ϣ�Ľṹ��                                           
/************************************************************************/
class FramePackage
{
public:
    int frameIndex;         // ֡��
    long long int rawtime;  // ʱ���
    Mat imSrc;              // ԭʼ֡
    Mat imMask;             // Mask��ͬʱ����ʵʱ������Ϣ
    Rect roi;               // ԭʼͼ��roi
};


/************************************************************************/
// ��������      
/************************************************************************/
class ItemInfo
{
public:
    int frameIndex;                     // ���ֵ�֡��
    int color;                          // ��ɫ
    int movingDirection;                // �˶�����
    Rect loc;                           // ��С��λ��
};


/************************************************************************/
// �������� 
/************************************************************************/
class PedestrianInfo:public ItemInfo
{
public:
    int pedestrian_basic_color;         // ��0 ��ɫ, 1 ��ɫ, 2 ��ɫ, 3 ��ɫ, 4 ��ɫ, 5 ��ɫ, 6 ��ɫ,7 ��ɫ, 8 ��ɫ,9 ��ɫ, 10 ��ɫ,11 �ۺ�ɫ, 12 δ֪��
    int pedestrian_upperbody_color ;    // �ϰ�����ɫ // ��0 ��ɫ, 1 ��ɫ, 2 ��ɫ, 3 ��ɫ, 4 ��ɫ, 5 ��ɫ, 6 ��ɫ,7 ��ɫ, 8 ��ɫ,9 ��ɫ, 10 ��ɫ,11 �ۺ�ɫ, 12 δ֪��
    int pedestrian_lowerbody_color ;    // �°�����ɫ // ��0 ��ɫ, 1 ��ɫ, 2 ��ɫ, 3 ��ɫ, 4 ��ɫ, 5 ��ɫ, 6 ��ɫ,7 ��ɫ, 8 ��ɫ,9 ��ɫ, 10 ��ɫ,11 �ۺ�ɫ, 12 δ֪��

    Rect recUpperBody;                  // �����ϰ������
    Rect recLowerBody;                  // �����°������
};


/************************************************************************/
// �������������ٽṹ��      
/************************************************************************/
class VehicleInfo:public ItemInfo
{
public:
    int color1;                         // ������ɫ1��0 ��ɫ, 1 ��ɫ, 2 ��ɫ, 3 ��ɫ, 4 ��ɫ, 5 ��ɫ, 6 ��ɫ,7 ��ɫ, 8 ��ɫ,9 ��ɫ, 10 ��ɫ,11 �ۺ�ɫ, 12 δ֪��
    int color2;                         // ������ɫ2��0 ��ɫ, 1 ��ɫ, 2 ��ɫ, 3 ��ɫ, 4 ��ɫ, 5 ��ɫ, 6 ��ɫ,7 ��ɫ, 8 ��ɫ,9 ��ɫ, 10 ��ɫ,11 �ۺ�ɫ, 12 δ֪��
    double weight1;                     // ������ɫȨ��1
    double weight2;                     // ������ɫȨ��2
    int vehicleType;                    // ��������
};


/************************************************************************/
// ���ٶ���
/************************************************************************/
class TrackingInfo
{
public:
    deque<FramePackage> fpkgs;
    deque<Rect> footprints;
public:
    virtual int tracking(ItemInfo* ii, FeaturePointTracker& tracker) {return 0;};       // ���ٷ��������ٳɹ�����1�����򷵻�0
};


class ItemTracking:public TrackingInfo
{
public:
    deque<ItemInfo> squence;
};

class VehicleTracking:public TrackingInfo
{
public:
    deque<VehicleInfo> squence;
    int tracking(ItemInfo* ii, FeaturePointTracker& tracker);
};

class PedestrianTracking:public TrackingInfo
{
public:
    deque<PedestrianInfo> squence;
    int tracking(ItemInfo* ii, FeaturePointTracker& tracker);
};


/************************************************************************/
// ��¼���ʶ�����������Ľṹ��       
/************************************************************************/
class DetectTask
{
public:
    FramePackage fpkg;      // �����Ӧ��ͼ����Ϣ
    vector<Rect> recs;      // �����еõ���һЩǰ����Ӿ���

    Mat mainFLow;           // ��¼�˶�������������

    vector<ItemInfo>        items;
    vector<PedestrianInfo>  pedestrians;
    vector<VehicleInfo>     vehicles;

    DetectTask(){};
    DetectTask(const DetectTask& _dt){
        this->fpkg = _dt.fpkg;
        // �����
        for (int i = 0; i < _dt.recs.size(); i++)
        {
            this->recs.push_back(_dt.recs[i]);  
        }
        for (int i = 0; i < _dt.pedestrians.size(); i++)
        {
            this->pedestrians.push_back(_dt.pedestrians[i]);
        }
        for (int i = 0; i < _dt.vehicles.size(); i++)
        {
            this->vehicles.push_back(_dt.vehicles[i]);
        }   
    };
};

#endif