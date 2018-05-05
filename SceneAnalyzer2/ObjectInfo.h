#ifndef _OBJECTINFO_H__
#define _OBJECTINFO_H__

#include <deque>

#include "opencv2/opencv.hpp"
//#include "pthread.h"

#include "FeaturePointTracker.h"

using namespace cv;
using namespace std;




/************************************************************************/
// Frame Package 输入结构定义，也是记录帧信息的结构体                                           
/************************************************************************/
class FramePackage
{
public:
    int frameIndex;         // 帧号
    long long int rawtime;  // 时间戳
    Mat imSrc;              // 原始帧
    Mat imMask;             // Mask，同时保存实时方向信息
    Rect roi;               // 原始图像roi
};


/************************************************************************/
// 对象属性      
/************************************************************************/
class ItemInfo
{
public:
    int frameIndex;                     // 出现的帧号
    int color;                          // 颜色
    int movingDirection;                // 运动方向
    Rect loc;                           // 大小及位置
};


/************************************************************************/
// 行人描述 
/************************************************************************/
class PedestrianInfo:public ItemInfo
{
public:
    int pedestrian_basic_color;         // （0 黑色, 1 蓝色, 2 棕色, 3 青色, 4 灰色, 5 绿色, 6 红色,7 银色, 8 白色,9 黄色, 10 紫色,11 粉红色, 12 未知）
    int pedestrian_upperbody_color ;    // 上半身颜色 // （0 黑色, 1 蓝色, 2 棕色, 3 青色, 4 灰色, 5 绿色, 6 红色,7 银色, 8 白色,9 黄色, 10 紫色,11 粉红色, 12 未知）
    int pedestrian_lowerbody_color ;    // 下半身颜色 // （0 黑色, 1 蓝色, 2 棕色, 3 青色, 4 灰色, 5 绿色, 6 红色,7 银色, 8 白色,9 黄色, 10 紫色,11 粉红色, 12 未知）

    Rect recUpperBody;                  // 行人上半身矩形
    Rect recLowerBody;                  // 行人下半身矩形
};


/************************************************************************/
// 车辆描述及跟踪结构体      
/************************************************************************/
class VehicleInfo:public ItemInfo
{
public:
    int color1;                         // 车身颜色1（0 黑色, 1 蓝色, 2 棕色, 3 青色, 4 灰色, 5 绿色, 6 红色,7 银色, 8 白色,9 黄色, 10 紫色,11 粉红色, 12 未知）
    int color2;                         // 车身颜色2（0 黑色, 1 蓝色, 2 棕色, 3 青色, 4 灰色, 5 绿色, 6 红色,7 银色, 8 白色,9 黄色, 10 紫色,11 粉红色, 12 未知）
    double weight1;                     // 车身颜色权重1
    double weight2;                     // 车身颜色权重2
    int vehicleType;                    // 车辆类型
};


/************************************************************************/
// 跟踪对象
/************************************************************************/
class TrackingInfo
{
public:
    deque<FramePackage> fpkgs;
    deque<Rect> footprints;
public:
    virtual int tracking(ItemInfo* ii, FeaturePointTracker& tracker) {return 0;};       // 跟踪方法，跟踪成功返回1，否则返回0
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
// 记录检测识别任务与结果的结构体       
/************************************************************************/
class DetectTask
{
public:
    FramePackage fpkg;      // 任务对应的图像信息
    vector<Rect> recs;      // 任务中得到的一些前景外接矩形

    Mat mainFLow;           // 记录运动场景的主方向

    vector<ItemInfo>        items;
    vector<PedestrianInfo>  pedestrians;
    vector<VehicleInfo>     vehicles;

    DetectTask(){};
    DetectTask(const DetectTask& _dt){
        this->fpkg = _dt.fpkg;
        // 做深拷贝
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