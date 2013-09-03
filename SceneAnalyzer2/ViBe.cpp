#include "ViBe.h"
#include <fstream>

RandIntUniform::RandIntUniform()
{

}

RandIntUniform::~RandIntUniform()
{

}

void RandIntUniform::init(int size, int minInc, int maxExc, long long int seed)
{
	capacity = size;
	minValInc = minInc;
	maxValExc = maxExc;
	mat.create(1, capacity, CV_32SC1);	
	RNG rng(seed);
	rng.fill(mat, RNG::UNIFORM, minValInc, maxValExc);
	data = (int*)mat.data;
	index = -1;
}

void RandIntUniform::update(long long int seed)
{
	RNG rng(seed);
	rng.fill(mat, RNG::UNIFORM, minValInc, maxValExc);
	index = -1;
}

void RandIntUniform::release(void)
{

}

char ViBe::adjPositions[8][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};

ViBe::ViBe()
{
	init();
}

ViBe::~ViBe()
{
	release();
}

void ViBe::init(void)
{
	config();
	
	ptrFore = 0;
#if RUN_VIBE_WITH_BACKGROUND_SUBTRACTION
	ptrBack = 0;
#endif
	ptrNoUpdate = 0;
	ptrSamples = 0;
	samples = 0;
}

static int countNonZero8UC3(Mat& image)
{
	if (image.type() != CV_8UC3)
		throw string("ERROR in function countNonZero8UC3(), image .type() != CV_8UC3");

	int count = 0;
	for (int i = 0; i < image.rows; i++)
	{
		unsigned char* ptr = image.ptr<unsigned char>(i);
		for (int j = 0; j < image.cols; j++)
		{
			if (ptr[j * 3] != 0 || ptr[j * 3 + 1] != 0 || ptr[j * 3 + 2] != 0)
				count++;
		}
	}
	return count;
}

static bool isNonZeroLarge(Mat& image, float ratioNonZero)
{
	if (image.type() == CV_8UC1)
		return countNonZero(image) > ratioNonZero * image.cols * image.rows;
	else if (image.type() == CV_8UC3)
		return countNonZero8UC3(image) > ratioNonZero * image.cols * image.rows;
}

void ViBe::init(Mat& image, const string& path, const string& signature)
{
	if (image.type() != CV_8UC3 && image.type() != CV_8UC1)
		throw string("ERROR in ViBe::init(), unsupported image format");

	imageWidth = image.cols;
	imageHeight = image.rows;
	imageChannels = image.channels();	

	config(path, signature);

	// 初始化随机数
	rndReplaceCurr.init(imageWidth * imageHeight, 0, subSampleInterval, getTickCount());
	rndIndexCurr.init(imageWidth * imageHeight, 0, numOfSamples, getTickCount() / 2);
	rndReplaceAdj.init(imageWidth * imageHeight, 0, subSampleInterval, getTickCount() / 3);
	rndPositionAdj.init(imageWidth * imageHeight, 0, 8, getTickCount() / 4);
	rndIndexAdj.init(imageWidth * imageHeight, 0, numOfSamples, getTickCount() / 5); 

	// 初始化前景图
	foreImage = Mat::zeros(imageHeight, imageWidth, CV_8UC1);
	ptrFore = new unsigned char* [imageHeight];
	for (int i = 0; i < imageHeight; i++)
		ptrFore[i] = foreImage.ptr<unsigned char>(i);

#if RUN_VIBE_WITH_BACKGROUND_SUBTRACTION
	// 初始化背景图
	if (imageChannels == 3)
		backImage = Mat::zeros(imageHeight, imageWidth, CV_32FC3);
	else if (imageChannels == 1)
		backImage = Mat::zeros(imageHeight, imageWidth, CV_32FC1);
	ptrBack = new float* [imageHeight];
	for (int i = 0; i < imageHeight; i++)
		ptrBack[i] = backImage.ptr<float>(i);
#endif

	// 不更新区域图
	noUpdateImage = Mat::zeros(imageHeight, imageWidth, CV_8UC1);
	ptrNoUpdate = new unsigned char* [imageHeight];
	for (int i = 0; i < imageHeight; i++)
		ptrNoUpdate[i] = noUpdateImage.ptr<unsigned char>(i);

	
	// 分配保存样本空间 标记行首地址
	samples = new unsigned char [imageWidth * imageHeight * imageChannels * numOfSamples];
	ptrSamples = new unsigned char * [imageHeight];
	for (int i = 0; i < imageHeight; i++)
		ptrSamples[i] = samples + imageWidth * numOfSamples * imageChannels * i;

	// 填充背景样本和背景图片
	if (imageChannels == 3)
		fill8UC3(image);
	else if (imageChannels == 1)
		fill8UC1(image);
}

void ViBe::release(void)
{
	delete [] ptrFore;       ptrFore = 0;
#if RUN_VIBE_WITH_BACKGROUND_SUBTRACTION
	delete [] ptrBack;       ptrBack = 0;
#endif
	delete [] ptrNoUpdate;   ptrNoUpdate = 0;
	delete [] samples;       samples = 0;
	delete [] ptrSamples;    ptrSamples = 0;
}

void ViBe::update(Mat& image, Mat& foregroundImage, const vector<Rect>& rectsNoUpdate)
{
	update(image, rectsNoUpdate);
	foreImage.copyTo(foregroundImage);
}

#if RUN_VIBE_WITH_BACKGROUND_SUBTRACTION
void ViBe::update(Mat& image, Mat& foregroundImage, Mat& backgroundImage, const vector<Rect>& rectsNoUpdate)
{
	update(image, rectsNoUpdate);
	foreImage.copyTo(foregroundImage);
	if (imageChannels == 3)
		backImage.convertTo(backgroundImage, CV_8UC3);
	else if (imageChannels == 1)
		backImage.convertTo(backgroundImage, CV_8UC1);
}
#endif

void ViBe::refill(Mat& image)
{
	if (image.channels() != imageChannels)
		throw string("ERROR in ViBe::refill(), image.channels() != imageChannels");
	if (image.cols != imageWidth || image.rows != imageHeight)
		throw string("ERROR in ViBe::refill(), image size does not match");

	if (imageChannels == 3)
		fill8UC3(image);
	else if (imageChannels == 1)
		fill8UC1(image);
}

void ViBe::fill8UC3(Mat& image)
{
	RandIntUniform rndInit;
	rndInit.init(imageWidth * imageHeight * numOfSamples, 0, 8, getTickCount());
	// 输入图片的行首地址
	unsigned char** ptrImage = new unsigned char* [imageHeight];
	for (int i = 0; i < imageHeight; i++)
	{
		ptrImage[i] = image.ptr<unsigned char>(i);
	}
	for (int i = 1; i < imageHeight - 1; i++)
	{
		for (int j = 1; j < imageWidth - 1; j++)
		{
			for (int k = 0; k < numOfSamples; k++)
			{
				int index = rndInit.getNext();
				memcpy(ptrSamples[i] + (j * numOfSamples + k) * 3, 
						ptrImage[i + adjPositions[index][0]] + (j + adjPositions[index][1]) * 3,
						sizeof(unsigned char) * 3);
			}
		} 
	}
#if RUN_VIBE_WITH_BACKGROUND_SUBTRACTION
	image.convertTo(backImage, CV_32FC3);
#endif
	delete [] ptrImage;
}

void ViBe::fill8UC1(Mat& image)
{
	RandIntUniform rndInit;
	rndInit.init(imageWidth * imageHeight * numOfSamples, 0, 8, getTickCount());
	// 输入图片的行首地址
	unsigned char** ptrImage = new unsigned char* [imageHeight];
	for (int i = 0; i < imageHeight; i++)
	{
		ptrImage[i] = image.ptr<unsigned char>(i);
	}
	for (int i = 1; i < imageHeight - 1; i++)
	{
		for (int j = 1; j < imageWidth - 1; j++)
		{
			for (int k = 0; k < numOfSamples; k++)
			{
				int index = rndInit.getNext();
				memcpy(ptrSamples[i] + (j * numOfSamples + k), 
						ptrImage[i + adjPositions[index][0]] + (j + adjPositions[index][1]),
						sizeof(unsigned char));
			}
		} 
	}
#if RUN_VIBE_WITH_BACKGROUND_SUBTRACTION
	image.convertTo(backImage, CV_32FC1);
#endif
	delete [] ptrImage;
}

void ViBe::update(Mat& image, const vector<Rect>& rectsNoUpdate)
{
	if (image.type() != CV_8UC3 && image.type() != CV_8UC1)
		throw string("ERROR in ViBe::update(), unsupported image format");
	
	if (image.channels() != imageChannels)
		throw string("ERROR in ViBe::update(), image.channels() != imageChannels");

	if (image.cols != imageWidth || image.rows != imageHeight)
		throw string("ERROR in ViBe::update(), image size does not match");

	noUpdateImage.setTo(0);
	if (!rectsNoUpdate.empty())
	{
		for (int i = 0; i < rectsNoUpdate.size(); i++)
		{
			Mat noUpdateMatROI = noUpdateImage(rectsNoUpdate[i] & Rect(0, 0, imageWidth, imageHeight));
			noUpdateMatROI.setTo(255);
		}
	}

	if (imageChannels == 3)
	{
		unsigned char** ptrImage = new unsigned char* [imageHeight];
		for (int i = 0; i < imageHeight; i++)
		{
			ptrImage[i] = image.ptr<unsigned char>(i);
		}

		for (int i = 1; i < imageHeight - 1; i++)
		{
			for (int j = 1; j < imageWidth - 1; j++)
			{
				// 统计当前像素能够和多少个已存储的样本匹配
				int matchCount = 0;
				unsigned char* ptrInput = ptrImage[i] + j * 3;
				unsigned char* ptrStore = ptrSamples[i] + j * numOfSamples * 3;
				for (int k = 0; k < numOfSamples && matchCount < minNumOfMatchCount; k++)
				{
					int dist = abs(int(ptrInput[0]) - int(ptrStore[k * 3])) +
						   abs(int(ptrInput[1]) - int(ptrStore[k * 3 + 1])) +
						   abs(int(ptrInput[2]) - int(ptrStore[k * 3 + 2]));
					if (dist < minMatchDist)
						matchCount++;
				}

				// 是前景
				if (matchCount < minNumOfMatchCount)
				{
					ptrFore[i][j] = 255;
					continue;
				}

				// 是背景
				ptrFore[i][j] = 0;

				if (ptrNoUpdate[i][j])
					continue;

				// 更新当前像素的存储样本
				if (rndReplaceCurr.getNext() == 0)
				{
					memcpy(ptrStore + rndIndexCurr.getNext() * 3, 
						   ptrInput, sizeof(unsigned char) * 3);
				}

				// 更新邻域像素的存储样本
				if (rndReplaceAdj.getNext() == 0)
				{
					int posAdj = rndPositionAdj.getNext();
					memcpy(ptrSamples[i + adjPositions[posAdj][0]] + 
						   ((j + adjPositions[posAdj][1]) * numOfSamples + rndIndexAdj.getNext()) * 3,
						   ptrInput, sizeof(unsigned char) * 3);
				}
			}
		}

#if RUN_VIBE_WITH_BACKGROUND_SUBTRACTION
		// 更新背景图
		accumulateWeighted(image, backImage, learnRate, ~(foreImage | noUpdateImage));
#endif
		delete [] ptrImage;
	}
	else if (imageChannels == 1)
	{
		unsigned char** ptrImage = new unsigned char* [imageHeight];
		for (int i = 0; i < imageHeight; i++)
		{
			ptrImage[i] = image.ptr<unsigned char>(i);
		}

		for (int i = 1; i < imageHeight - 1; i++)
		{
			for (int j = 1; j < imageWidth - 1; j++)
			{
				// 统计当前像素能够和多少个已存储的样本匹配
				int matchCount = 0;
				unsigned char* ptrInput = ptrImage[i] + j;
				unsigned char* ptrStore = ptrSamples[i] + j * numOfSamples;
				for (int k = 0; k < numOfSamples && matchCount < minNumOfMatchCount; k++)
				{
					int dist = abs(int(ptrInput[0]) - int(ptrStore[k]));
					if (dist < minMatchDist)
						matchCount++;
				}

				// 是前景
				if (matchCount < minNumOfMatchCount)
				{
					ptrFore[i][j] = 255;
					continue;
				}

				// 是背景
				ptrFore[i][j] = 0;

				if (ptrNoUpdate[i][j])
					continue;

				// 更新当前像素的存储样本
				if (rndReplaceCurr.getNext() == 0)
				{
					memcpy(ptrStore + rndIndexCurr.getNext(), 
						   ptrInput, sizeof(unsigned char));
				}

				// 更新邻域像素的存储样本
				if (rndReplaceAdj.getNext() == 0)
				{
					int posAdj = rndPositionAdj.getNext();
					memcpy(ptrSamples[i + adjPositions[posAdj][0]] + 
						   ((j + adjPositions[posAdj][1]) * numOfSamples + rndIndexAdj.getNext()),
						   ptrInput, sizeof(unsigned char));
				}
			}
		}

#if RUN_VIBE_WITH_BACKGROUND_SUBTRACTION
		// 更新背景图
		accumulateWeighted(image, backImage, learnRate, ~(foreImage | noUpdateImage));
#endif
		delete [] ptrImage;
	}	
}

void ViBe::showSamples(int count)
{
	if (count <= 0 || count > numOfSamples)
		return;

	Mat* samplesMat = new Mat [count];
	if (imageChannels == 3)
	{
		
		for (int i = 0; i < count; i++)
		{
			samplesMat[i].create(imageHeight, imageWidth, CV_8UC3);
		}

		for (int i = 0; i < imageHeight; i++)
		{
			unsigned char** ptr = new unsigned char* [count];
			for (int k = 0; k < count; k++)
			{
				ptr[k] = samplesMat[k].ptr<unsigned char>(i);
			}
			for (int j = 0; j < imageWidth; j++)
			{			
				for (int k = 0; k < count; k++)
				{
					memcpy(ptr[k] + j * 3, 
						   ptrSamples[i] + (j * numOfSamples + k) * 3, 
						   sizeof(unsigned char) * 3);
				}			
			}
			delete [] ptr;
		}
	}
	else if (imageChannels == 1)
	{
		for (int i = 0; i < count; i++)
		{
			samplesMat[i].create(imageHeight, imageWidth, CV_8UC1);
		}

		for (int i = 0; i < imageHeight; i++)
		{
			unsigned char** ptr = new unsigned char* [count];
			for (int k = 0; k < count; k++)
			{
				ptr[k] = samplesMat[k].ptr<unsigned char>(i);
			}
			for (int j = 0; j < imageWidth; j++)
			{			
				for (int k = 0; k < count; k++)
				{
					memcpy(ptr[k] + j, 
						   ptrSamples[i] + (j * numOfSamples + k), 
						   sizeof(unsigned char));
				}			
			}
			delete [] ptr;
		}
	}

	for (int i = 0; i < count; i++)
	{
		char imageName[100];
		sprintf(imageName, "samples %d", i);
		imshow(imageName, samplesMat[i]);
	}
	delete [] samplesMat;
}

void ViBe::config(const string& path, const string& signature)
{
	if (path.size() == 0 || signature.size() == 0)
	{
		numOfSamples = 20;
		minMatchDist = 40;
		minNumOfMatchCount = 2;
		subSampleInterval = 16;
#if RUN_VIBE_WITH_BACKGROUND_SUBTRACTION
		learnRate = 0.02F;
		compLearnRate = 1.F - learnRate;
#endif
	}
	else
	{
		fstream initFileStream;
		initFileStream.open(path.c_str());
		if (!initFileStream.is_open())
		{
			stringstream sstr;
			sstr << "ERROR in ViBe::config(), cannot open file " << path << " for configuration";
			throw sstr.str();
		}
		char stringNotUsed[500];
		do
		{
			initFileStream >> stringNotUsed;
			if (initFileStream.eof())
			{
				stringstream sstr;
				sstr << "ERROR in ViBe::config(), cannot find label " << signature << " for configuration";
				throw sstr.str();
			}
		}
		while(string(stringNotUsed) != string(signature));

		initFileStream >> stringNotUsed >> numOfSamples;
		initFileStream >> stringNotUsed >> minMatchDist;
		initFileStream >> stringNotUsed >> minNumOfMatchCount;
		initFileStream >> stringNotUsed >> subSampleInterval;
#if RUN_VIBE_WITH_BACKGROUND_SUBTRACTION
		initFileStream >> stringNotUsed >> learnRate;
#endif

		initFileStream.close();
		initFileStream.clear();
#if RUN_VIBE_WITH_BACKGROUND_SUBTRACTION
		compLearnRate = 1.F - learnRate;
#endif

		printf("display ViBe config for %s:\n", signature.c_str());
		printf("  numOfSamples = %d\n", numOfSamples);
		printf("  minMatchDist = %d\n", minMatchDist);
		printf("  minNumOfMatchCount = %d\n", minNumOfMatchCount);
		printf("  subSampleInterval = %d\n", subSampleInterval);
#if RUN_VIBE_WITH_BACKGROUND_SUBTRACTION
		printf("  learnRate = %.4f\n", learnRate);
		printf("  compLearnRate = %.4f\n", compLearnRate);
#endif
		printf("\n");
	}
}