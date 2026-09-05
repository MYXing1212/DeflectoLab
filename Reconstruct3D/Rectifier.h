#pragma once

// 极线校正类
// 将左右相机图像校正到共面且极线水平的配置
// 使用 Bouguet 算法 (OpenCV stereoRectify) 计算左右相机的校正映射
// 校正后左右相机内参一致, 极线水平, 便于按行进行立体匹配

class CRectifier
{
public:
	CRectifier();
	~CRectifier();

	// 初始化 (需先载入双目标定参数)
	bool init(const cv::Size &imgSize, class CCalibParas *calib);

	// 对灰度图像进行极线校正
	void rectifyGray(const cv::Mat &src, cv::Mat &dst, bool isLeft);

	// 对相位图进行极线校正 (带掩膜, 双线性插值)
	void rectifyPhase(const cv::Mat &phase, const cv::Mat &mask,
		cv::Mat &newPhase, cv::Mat &newMask, bool isLeft);

	// 校正后图像尺寸
	cv::Size newSize() const { return m_newSize; }

	// 校正后左右主点偏移 (相对于图像中心)
	glm::vec2 newCLeft()  const { return m_newCLeft; }
	glm::vec2 newCRight() const { return m_newCRight; }

	// 是否已初始化
	bool isReady() const { return m_bReady; }

private:
	cv::Size m_srcSize;
	cv::Size m_newSize;
	cv::Mat  m_mapLx, m_mapLy;   // 左相机校正映射
	cv::Mat  m_mapRx, m_mapRy;   // 右相机校正映射
	glm::vec2 m_newCLeft = glm::vec2(0);
	glm::vec2 m_newCRight = glm::vec2(0);
	bool m_bReady = false;
};