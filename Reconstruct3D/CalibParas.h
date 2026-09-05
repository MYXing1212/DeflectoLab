#pragma once

// 双目Deflectometry系统标定参数类
// 标定参数文件格式:
//   cal_res.txt (33个数): 双目相机外参(om,T) + 左相机内参+畸变 + 右相机内参+畸变 + F矩阵
//   cal_Projector.txt (10个数): 显示屏外参(om,T) + 像素尺寸psW psH(mm/px) + 分辨率prjW prjH
//     旧格式(15个数, fc/cc/dist)仍兼容读取: ps=1/fc, res=round(2*cc)

class CCalibParas
{
public:
	CCalibParas();
	~CCalibParas();

	// 读取双目相机标定参数 cal_res.txt
	bool readBinocularCalib(const std::string &path);

	// 读取显示屏标定参数 cal_Projector.txt (相对于左相机)
	bool readProjectorCalib(const std::string &path);

	// 左相机内参 fc=(fx,fy) cc=(cx,cy)
	glm::vec2 fcLeft()  const { return m_fcLeft; }
	glm::vec2 ccLeft()  const { return m_ccLeft; }
	cv::Mat  intrinsicLeft() const;
	cv::Mat  distortLeft()   const { return m_distortLeft.clone(); }

	// 右相机内参
	glm::vec2 fcRight() const { return m_fcRight; }
	glm::vec2 ccRight() const { return m_ccRight; }
	cv::Mat  intrinsicRight() const;
	cv::Mat  distortRight()   const { return m_distortRight.clone(); }

	// 右相机相对于左相机的外参
	cv::Mat rvecRight() const { return m_rvecRight.clone(); }
	cv::Mat tvecRight() const { return m_tvecRight.clone(); }
	cv::Mat rmatRight() const;

	// 显示屏(投影屏)参数: 外参(相对左相机) + 像素尺寸 + 分辨率
	// 屏坐标系约定(与 Match::displayTo3D / Blender screen_frame_from_uv 一致):
	//   原点 = 屏左上角(图像(0,0)像素处), +X=像素u+(图像右), +Y=像素v+(图像下), 屏面z=0
	// 重建中显示屏像素坐标 → 3D: P = Rprj * [u*psW, v*psH, 0] + Tprj
	cv::Mat rvecPrj()   const { return m_rvecPrj.clone(); }
	cv::Mat tvecPrj()   const { return m_tvecPrj.clone(); }
	cv::Mat rmatPrj()   const;

	double pixelSizeW() const { return m_pixelSizeW; }   // mm/px 水平
	double pixelSizeH() const { return m_pixelSizeH; }   // mm/px 垂直
	int    prjResW()    const { return m_prjResW; }      // 显示屏分辨率宽
	int    prjResH()    const { return m_prjResH; }      // 显示屏分辨率高

	// 基线长度(左右相机距离) mm
	double baseline() const;

	// 是否已载入
	bool isBinocLoaded() const { return m_bBinocLoaded; }
	bool isPrjLoaded()   const { return m_bPrjLoaded; }

	// 最近一次读取的错误列表(明确到具体参数名), 为空表示读取成功
	const std::vector<std::string> &lastReadErrors() const { return m_lastReadErrors; }

	// 标定参数有效性校验: 返回问题列表(每条明确指出是哪个参数有问题)
	// imgW/imgH: 相机分辨率(<=0 则跳过与分辨率相关的检查)
	std::vector<std::string> validate(int imgW = 0, int imgH = 0) const;

private:
	glm::vec2 m_fcLeft,  m_ccLeft;
	glm::vec2 m_fcRight, m_ccRight;

	cv::Mat m_distortLeft;   // 5x1
	cv::Mat m_distortRight;  // 5x1

	cv::Mat m_rvecRight;     // 3x1  右相机相对于左相机
	cv::Mat m_tvecRight;     // 3x1
	cv::Mat m_rvecPrj;       // 3x1  显示屏相对于左相机
	cv::Mat m_tvecPrj;       // 3x1

	double m_pixelSizeW = 0.0;   // mm/px
	double m_pixelSizeH = 0.0;
	int    m_prjResW = 0;
	int    m_prjResH = 0;

	cv::Mat m_F;             // 3x3 基本矩阵

	bool m_bBinocLoaded = false;
	bool m_bPrjLoaded   = false;

	std::vector<std::string> m_lastReadErrors;
};