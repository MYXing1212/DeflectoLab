#include "stdafx.h"
#include "Rectifier.h"
#include "CalibParas.h"

CRectifier::CRectifier()
{
}

CRectifier::~CRectifier()
{
}

bool CRectifier::init(const cv::Size &imgSize, CCalibParas *calib)
{
	if (!calib || !calib->isBinocLoaded())
	{
		std::cerr << "极线校正初始化失败: 双目标定参数未载入" << std::endl;
		return false;
	}
	m_srcSize = imgSize;

	cv::Mat K1 = calib->intrinsicLeft();
	cv::Mat D1 = calib->distortLeft();
	cv::Mat K2 = calib->intrinsicRight();
	cv::Mat D2 = calib->distortRight();
	cv::Mat R  = calib->rmatRight();
	cv::Mat T  = calib->tvecRight();

	cv::Mat R1, R2, P1, P2, Q;
	cv::stereoRectify(K1, D1, K2, D2, imgSize, R, T,
		R1, R2, P1, P2, Q,
		cv::CALIB_ZERO_DISPARITY, 0, imgSize);

	// 校正映射
	cv::initUndistortRectifyMap(K1, D1, R1, P1, imgSize, CV_32FC1, m_mapLx, m_mapLy);
	cv::initUndistortRectifyMap(K2, D2, R2, P2, imgSize, CV_32FC1, m_mapRx, m_mapRy);
	m_newSize = imgSize;

	// 校正后主点 = P1/P2 的 cx,cy 相对图像中心的偏移
	m_newCLeft  = glm::vec2((float)(P1.at<double>(0, 2) - imgSize.width  * 0.5),
		(float)(P1.at<double>(1, 2) - imgSize.height * 0.5));
	m_newCRight = glm::vec2((float)(P2.at<double>(0, 2) - imgSize.width  * 0.5),
		(float)(P2.at<double>(1, 2) - imgSize.height * 0.5));

	m_bReady = true;
	std::cout << "极线校正初始化完成, 新尺寸=" << m_newSize.width << "x" << m_newSize.height << std::endl;
	std::cout << "newCLeft =(" << m_newCLeft.x << "," << m_newCLeft.y << ") newCRight=(" << m_newCRight.x << "," << m_newCRight.y << ")" << std::endl;
	return true;
}

void CRectifier::rectifyGray(const cv::Mat &src, cv::Mat &dst, bool isLeft)
{
	if (!m_bReady) { dst = src.clone(); return; }
	cv::remap(src, dst, isLeft ? m_mapLx : m_mapRx, isLeft ? m_mapLy : m_mapRy, cv::INTER_LINEAR);
}

void CRectifier::rectifyPhase(const cv::Mat &phase, const cv::Mat &mask,
	cv::Mat &newPhase, cv::Mat &newMask, bool isLeft)
{
	if (!m_bReady)
	{
		newPhase = phase.clone();
		newMask = mask.clone();
		return;
	}
	const cv::Mat &mapX = isLeft ? m_mapLx : m_mapRx;
	const cv::Mat &mapY = isLeft ? m_mapLy : m_mapRy;
	int W = m_newSize.width, H = m_newSize.height;
	newPhase = cv::Mat::zeros(H, W, CV_32FC1);
	newMask  = cv::Mat::zeros(H, W, CV_8UC1);

	for (int i = 0; i < H; i++)
	{
		const float *px = mapX.ptr<float>(i);
		const float *py = mapY.ptr<float>(i);
		float *op = newPhase.ptr<float>(i);
		uchar *om = newMask.ptr<uchar>(i);
		for (int j = 0; j < W; j++)
		{
			float fx = px[j], fy = py[j];
			int x0 = (int)floor(fx), x1 = x0 + 1;
			int y0 = (int)floor(fy), y1 = y0 + 1;
			if (x0 < 0 || y0 < 0 || x1 >= m_srcSize.width || y1 >= m_srcSize.height)
				continue;
			float dx = fx - x0, dy = fy - y0;
			// 四角掩膜
			uchar m00 = mask.at<uchar>(y0, x0);
			uchar m01 = mask.at<uchar>(y0, x1);
			uchar m10 = mask.at<uchar>(y1, x0);
			uchar m11 = mask.at<uchar>(y1, x1);
			if (m00 && m01 && m10 && m11)
			{
				float p00 = phase.at<float>(y0, x0);
				float p01 = phase.at<float>(y0, x1);
				float p10 = phase.at<float>(y1, x0);
				float p11 = phase.at<float>(y1, x1);
				op[j] = p00 + dx * (p01 - p00) + dy * (p10 - p00) + dx * dy * (p00 + p11 - p01 - p10);
				om[j] = 255;
			}
		}
	}
}