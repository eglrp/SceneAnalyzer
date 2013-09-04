#include "ViBeForegroundExtractor.h"

void StaticRectExtractor::init(const Size& imageSize)
{
	size = imageSize;
	currFrameCount = 0;
	thresFrameCount = 1000;
	maxFrameCount = 1000;
}

void StaticRectExtractor::extract(const vector<Rect>& rects, vector<Rect>& stableRects)
{
	// 前面 maxFrameCount 帧 
	if (currFrameCount < maxFrameCount)
	{
		currFrameCount++;
	}
	else if (currFrameCount == maxFrameCount)
	{
		currFrameCount++;
		// 删除 startFrameCount <= thresFrameCount 的矩形
		for (vector<RectInfo>::iterator itr = staticRectInfos.begin(); itr != staticRectInfos.end();)
		{
			if (itr->startFrameCount <= thresFrameCount)
				itr = staticRectInfos.erase(itr);
			else
				++itr;
		}
	}

	vector<RectInfo> currRectInfos(rects.size());
	for (int i = 0; i < rects.size(); i++)
		currRectInfos[i] = RectInfo(rects[i], currFrameCount);	

	// 如果 staticRectInfos 为空，则将当前帧找到的所有 rects 放到 staticRectInfos 中
	if (staticRectInfos.empty())
	{
		for (int i = 0; i < currRectInfos.size(); i++)
			staticRectInfos.push_back(currRectInfos[i]);
	}
	// 否则
	else
	{
		for (int i = 0; i < currRectInfos.size(); i++)
			currRectInfos[i].matchCount = 0;
			
		// 如果 staticRectInfos 中的元素和某个 currRectInfos 中的元素能够完美匹配 则修改 staticRectInfos 中元素的 matchCount
		for (vector<RectInfo>::iterator itr = staticRectInfos.begin(); itr != staticRectInfos.end();)
		{
			bool match = false;
			for (int i = 0; i < currRectInfos.size(); i++)
			{
				if (currRectInfos[i].matchCount)
					continue;

				Rect intersectRect = itr->rect & currRectInfos[i].rect;
				Rect unionRect = itr->rect | currRectInfos[i].rect;
				if (intersectRect.area() > 0.95 * unionRect.area())
				{
					currRectInfos[i].matchCount = 1;
					itr->rect = currRectInfos[i].rect;
					match = true;
					(itr->matchCount)++;
					if (itr->missCount > 0)
						itr->missCount = 0;
					break;
				}
			}

			if (!match)
			{
				(itr->missCount)++;
				if (itr->missCount > 15)
					itr = staticRectInfos.erase(itr);
				else
					++itr;
			}
			else
				++itr;
		}

		// 未能和 staticRectInfos 中任何元素匹配的 currRectInfos 中的元素直接放到 staticRectInfos 中
		for (int i = 0; i < currRectInfos.size(); i++)
		{
			if (!currRectInfos[i].matchCount)
				staticRectInfos.push_back(currRectInfos[i]);
		}
					
		// 检查 staticRectInfos 中所有元素 matchCount 的值
		// 大于阈值且面积较大的矩形放到 rectsNoUpdate 中
		stableRects.clear();
		for (int i = 0; i < staticRectInfos.size(); i++)
		{
			if (staticRectInfos[i].matchCount > 20 &&
				staticRectInfos[i].startFrameCount > thresFrameCount/* && staticRectInfos[i].rect.area() > 100*/)
			{
				//printf("stable rect: x = %d, y = %d, w = %d, h = %d\n", 
				//	staticRectInfos[i].rect.x, staticRectInfos[i].rect.y,
				//	staticRectInfos[i].rect.width, staticRectInfos[i].rect.height);
				stableRects.push_back(staticRectInfos[i].rect);
			}
		}
	}
}

void StaticRectExtractor::clear(void)
{
	staticRectInfos.clear();
	currFrameCount = 0;
}

void ViBeForegroundExtractor::init(Mat& image, Mat& gradImage, const string& path)
{
	imageWidth = image.cols;
	imageHeight = image.rows;
	colorBackModel.init(image, path, "[color]");
	gradBackModel.init(gradImage, path, "[gradient]");
	rectExtractor.init(Size(imageWidth, imageHeight));
	foreLargeCount = 0;
}

void ViBeForegroundExtractor::apply(Mat& image, Mat& gradImage, 
	Mat& foregroundImage, vector<Rect>& foregroundRects)
{
	colorBackModel.update(image, colorForeImage, staticRects);
	gradBackModel.update(gradImage, gradForeImage, staticRects);
	medianBlur(colorForeImage, colorForeImage, 3);
	medianBlur(gradForeImage, gradForeImage, 3);

	static const Mat kern = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
	dilate(colorForeImage, colorForeImage, kern);
	dilate(gradForeImage, gradForeImage, kern);
	vector<vector<Point> > contours, refinedContours;
	//colorForeImage.copyTo(testChangeForeImage);
	testChangeForeImage = colorForeImage + gradForeImage;
	findContours(testChangeForeImage, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	bool hasLargeForeground = false;
	vector<Rect> rects;
	for (int i = 0; i < contours.size(); i++)
	{
		Rect rect = boundingRect(contours[i]);
		//if (rect.width > 0.8 * imageWidth && rect.height > 0.8 * imageHeight)
		if (rect.width > 1.8 * imageWidth && rect.height > 1.8 * imageHeight)
			hasLargeForeground = true;
		if (rect.width > 20 && rect.height > 20)
		{
			rects.push_back(rect);
			refinedContours.push_back(contours[i]);
		}
	}
	rectExtractor.extract(rects, staticRects);

	//for (int i = 0; i < rects.size(); i++)
	//	rectangle(image, rects[i], Scalar(255, 0, 0));
	//for (int i = 0; i < staticRects.size(); i++)
	//	rectangle(image, staticRects[i], Scalar(0, 255, 0));
	//imshow("image", image);
	//imshow("grad image", gradImage);
	//imshow("color fore image", colorForeImage);
	//imshow("grad fore image", gradForeImage);
	//waitKey(5);

	if (hasLargeForeground)
	{
		foreLargeCount++;
		printf("foreground large, count is %d\n", foreLargeCount);
		//waitKey(0);
	}
	else 
		foreLargeCount = 0;

	if (foreLargeCount > 10)
	{
		printf("refille foreground\n");
		//waitKey(0);
		foreLargeCount = 0;
		colorBackModel.refill(image);
		gradBackModel.refill(gradImage);
		rectExtractor.clear();
	}

	foregroundImage = Mat::zeros(imageHeight, imageWidth, CV_8UC1);
	drawContours(foregroundImage, refinedContours, -1, Scalar(255, 255, 255), -1);
	foregroundRects = rects;
}