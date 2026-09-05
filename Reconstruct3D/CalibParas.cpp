#include "stdafx.h"
#include "CalibParas.h"
#include <cmath>

CCalibParas::CCalibParas()
{
	m_fcLeft = m_ccLeft = glm::vec2(0);
	m_fcRight = m_ccRight = glm::vec2(0);
	m_distortLeft  = cv::Mat::zeros(5, 1, CV_64FC1);
	m_distortRight = cv::Mat::zeros(5, 1, CV_64FC1);
	m_rvecRight = cv::Mat::zeros(3, 1, CV_64FC1);
	m_tvecRight = cv::Mat::zeros(3, 1, CV_64FC1);
	m_rvecPrj   = cv::Mat::zeros(3, 1, CV_64FC1);
	m_tvecPrj   = cv::Mat::zeros(3, 1, CV_64FC1);
	m_F = cv::Mat::zeros(3, 3, CV_64FC1);
}

CCalibParas::~CCalibParas()
{
}

namespace
{
	// 逐个数读取并检查: 失败时记录具体参数名, 便于定位标定文件中的问题参数
	bool readD(std::istream &fs, double &v, const std::string &name,
		std::vector<std::string> &errs)
	{
		if (!(fs >> v))
		{
			errs.push_back("读取失败(文件缺失数值或格式错误): " + name);
			v = 0.0;
			return false;
		}
		if (!std::isfinite(v))
		{
			errs.push_back("数值非法(NaN/Inf): " + name);
			v = 0.0;
			return false;
		}
		return true;
	}

	bool readVec(std::istream &fs, double *arr, int n, const std::string &prefix,
		std::vector<std::string> &errs)
	{
		bool ok = true;
		for (int i = 0; i < n; i++)
		{
			char idx[16];
			sprintf_s(idx, "[%d]", i);
			if (!readD(fs, arr[i], prefix + idx, errs)) ok = false;
		}
		return ok;
	}

	bool checkFiniteVec(const cv::Mat &m, const std::string &name,
		std::vector<std::string> &issues)
	{
		for (int i = 0; i < m.rows; i++)
			for (int j = 0; j < m.cols; j++)
			{
				double v = m.at<double>(i, j);
				if (!std::isfinite(v))
				{
					char buf[256];
					sprintf_s(buf, "%s 含非法数值 NaN/Inf (第%d个分量=%.1e)",
						name.c_str(), i * m.cols + j, v);
					issues.push_back(buf);
					return false;
				}
			}
		return true;
	}
}

// cal_res.txt 格式:
//   om(3) T(3)  fcLeft(2) ccLeft(2) distortLeft(5)  fcRight(2) ccRight(2) distortRight(5)  F(9)
bool CCalibParas::readBinocularCalib(const std::string &path)
{
	m_lastReadErrors.clear();
	std::fstream fs(path, std::ios_base::in);
	if (!fs.is_open())
	{
		std::cerr << "无法打开双目标定参数文件: " << path << std::endl;
		m_lastReadErrors.push_back("无法打开文件: " + path);
		return false;
	}
	std::cout << "载入双目标定参数: " << path << std::endl;

	auto &errs = m_lastReadErrors;
	double om[3], T[3];
	bool ok = true;
	ok = readVec(fs, om, 3, "om_R(右相机相对左相机旋转向量)", errs) && ok;
	ok = readVec(fs, T, 3, "T_R(右相机相对左相机平移)", errs) && ok;

	double fcL[2], ccL[2], distL[5];
	ok = readVec(fs, fcL, 2, "fc_L(左相机焦距)", errs) && ok;
	ok = readVec(fs, ccL, 2, "cc_L(左相机主点)", errs) && ok;
	ok = readVec(fs, distL, 5, "dist_L(左相机畸变)", errs) && ok;

	double fcR[2], ccR[2], distR[5];
	ok = readVec(fs, fcR, 2, "fc_R(右相机焦距)", errs) && ok;
	ok = readVec(fs, ccR, 2, "cc_R(右相机主点)", errs) && ok;
	ok = readVec(fs, distR, 5, "dist_R(右相机畸变)", errs) && ok;

	double F[9];
	ok = readVec(fs, F, 9, "F(基本矩阵)", errs) && ok;

	if (!ok)
	{
		m_bBinocLoaded = false;
		std::cerr << "cal_res.txt 读取失败, 共" << errs.size() << "处错误" << std::endl;
		for (auto &e : errs) std::cerr << "  " << e << std::endl;
		return false;
	}

	m_rvecRight = (cv::Mat_<double>(3, 1) << om[0], om[1], om[2]);
	m_tvecRight = (cv::Mat_<double>(3, 1) << T[0], T[1], T[2]);
	m_fcLeft = glm::vec2(fcL[0], fcL[1]);
	m_ccLeft = glm::vec2(ccL[0], ccL[1]);
	for (int i = 0; i < 5; i++) m_distortLeft.at<double>(i, 0) = distL[i];
	m_fcRight = glm::vec2(fcR[0], fcR[1]);
	m_ccRight = glm::vec2(ccR[0], ccR[1]);
	for (int i = 0; i < 5; i++) m_distortRight.at<double>(i, 0) = distR[i];
	for (int i = 0; i < 9; i++)
		m_F.at<double>(i / 3, i % 3) = F[i];

	m_bBinocLoaded = true;
	std::cout << "左相机 fc=(" << m_fcLeft.x << "," << m_fcLeft.y << ") cc=(" << m_ccLeft.x << "," << m_ccLeft.y << ")" << std::endl;
	std::cout << "右相机 fc=(" << m_fcRight.x << "," << m_fcRight.y << ") cc=(" << m_ccRight.x << "," << m_ccRight.y << ")" << std::endl;
	std::cout << "基线长度 = " << baseline() << " mm" << std::endl;
	return true;
}

// cal_Projector.txt 格式:
//   新格式(10个数): om(3) T(3)  psW psH(像素尺寸mm/px)  prjW prjH(分辨率)
//   旧格式(15个数): om(3) T(3)  fcPrj(2) ccPrj(2) distortPrj(5)  [兼容: ps=1/fc, res=round(2*cc)]
//   显示屏坐标约定: 原点=屏左上角(图像(0,0)像素处), +X沿像素u+(图像右), +Y沿像素v+(图像下), 屏幕平面z=0
bool CCalibParas::readProjectorCalib(const std::string &path)
{
	m_lastReadErrors.clear();
	std::fstream fs(path, std::ios_base::in);
	if (!fs.is_open())
	{
		std::cerr << "无法打开显示屏标定参数文件: " << path << std::endl;
		m_lastReadErrors.push_back("无法打开文件: " + path);
		return false;
	}
	std::cout << "载入显示屏标定参数: " << path << std::endl;

	auto &errs = m_lastReadErrors;
	double om[3], T[3];
	bool ok = true;
	ok = readVec(fs, om, 3, "om_S(显示屏相对左相机旋转向量)", errs) && ok;
	ok = readVec(fs, T, 3, "T_S(显示屏相对左相机平移)", errs) && ok;

	// 通过后续数字个数区分新旧格式: 新10个/旧15个 (前面共用 om3+T3)
	double v4[4], v5th = 0.0;
	ok = readVec(fs, v4, 4, "显示屏参数(psW psH prjW prjH)", errs) && ok;
	bool legacy = (bool)(fs >> v5th);   // 能读到第11个数 → 旧格式(dist_P[0])

	if (legacy)
	{
		// 旧格式: v4 = fcP(2)+ccP(2), 继续读 distP 剩余4个
		double distRest[4];
		ok = readVec(fs, distRest, 4, "dist_P(投影仪畸变)", errs) && ok;
		if (!ok)
		{
			m_bPrjLoaded = false;
			return false;
		}
		double fcPx = v4[0], fcPy = v4[1];
		double ccPx = v4[2], ccPy = v4[3];
		if (fcPx <= 0 || fcPy <= 0)
		{
			errs.push_back("旧格式投影仪焦距 fc_P 非正, 无法换算像素尺寸");
			m_bPrjLoaded = false;
			return false;
		}
		m_pixelSizeW = 1.0 / fcPx;
		m_pixelSizeH = 1.0 / fcPy;
		m_prjResW = (int)std::lround(2.0 * ccPx);
		m_prjResH = (int)std::lround(2.0 * ccPy);
		std::cout << "  检测到旧格式(15数), 已换算: ps=(" << m_pixelSizeW << "," << m_pixelSizeH
			<< ") mm/px, res=" << m_prjResW << "x" << m_prjResH << std::endl;
	}
	else
	{
		if (fs.bad()) errs.push_back("文件流错误");
		m_pixelSizeW = v4[0];
		m_pixelSizeH = v4[1];
		m_prjResW = (int)std::lround(v4[2]);
		m_prjResH = (int)std::lround(v4[3]);
	}

	if (!ok)
	{
		m_bPrjLoaded = false;
		std::cerr << "cal_Projector.txt 读取失败, 共" << errs.size() << "处错误" << std::endl;
		for (auto &e : errs) std::cerr << "  " << e << std::endl;
		return false;
	}

	m_rvecPrj = (cv::Mat_<double>(3, 1) << om[0], om[1], om[2]);
	m_tvecPrj = (cv::Mat_<double>(3, 1) << T[0], T[1], T[2]);

	m_bPrjLoaded = true;
	std::cout << "显示屏 ps=(" << m_pixelSizeW << "," << m_pixelSizeH << ") mm/px, 分辨率="
		<< m_prjResW << "x" << m_prjResH << std::endl;
	return true;
}

// 标定参数有效性校验: 逐项检查并明确指出问题参数
std::vector<std::string> CCalibParas::validate(int imgW, int imgH) const
{
	std::vector<std::string> issues;
	char buf[512];

	if (!m_bBinocLoaded) issues.push_back("[参数缺失] 双目标定参数未加载(cal_res.txt)");
	if (!m_bPrjLoaded)   issues.push_back("[参数缺失] 显示屏标定参数未加载(cal_Projector.txt)");

	auto checkIntrinsics = [&](const glm::vec2 &fc, const glm::vec2 &cc,
		const std::string &tag, int resW, int resH)
	{
		if (!(fc.x > 0.0) || !(fc.y > 0.0))
		{
			sprintf_s(buf, "[内参异常] %s焦距 fc 无效(必须>0): fx=%.4g fy=%.4g, "
				"请检查标定文件内参数值或单位", tag.c_str(), fc.x, fc.y);
			issues.push_back(buf);
		}
		else if (resW > 0 && (fc.x < resW * 0.05 || fc.x > resW * 20.0))
		{
			sprintf_s(buf, "[内参异常] %s焦距 fc.x=%.4g 与图像宽度%d量级不符, "
				"请确认内参是否与当前分辨率匹配", tag.c_str(), fc.x, resW);
			issues.push_back(buf);
		}
		if (!(cc.x >= 0.0) || !(cc.y >= 0.0) ||
			(resW > 0 && (cc.x > resW || cc.y > resH)))
		{
			sprintf_s(buf, "[内参异常] %s主点 cc=(%.4g, %.4g) 超出合理范围(图像 %dx%d)",
				tag.c_str(), cc.x, cc.y, resW, resH);
			issues.push_back(buf);
		}
	};

	auto checkExtrinsics = [&](const cv::Mat &rvec, const cv::Mat &tvec,
		const std::string &tag, double minT, double maxT)
	{
		if (!checkFiniteVec(rvec, tag + "旋转向量 om", issues)) return;
		if (!checkFiniteVec(tvec, tag + "平移 T", issues)) return;

		double omNorm = cv::norm(rvec);
		if (omNorm > CV_PI + 0.2)
		{
			sprintf_s(buf, "[外参异常] %s旋转向量 om 范数=%.4f 大于PI, "
				"请确认是否为罗德里格斯旋转向量", tag.c_str(), omNorm);
			issues.push_back(buf);
		}

		double tNorm = cv::norm(tvec);
		if (tNorm < 1e-9)
		{
			sprintf_s(buf, "[外参异常] %s平移 T 全零, 两坐标系重合, 外参无效", tag.c_str());
			issues.push_back(buf);
		}
		else if (tNorm < minT)
		{
			sprintf_s(buf, "[单位错误] %s平移 T 模长=%.4f 过小, 疑似单位为米(m)而非毫米(mm), "
				"Blender导出应乘以1000", tag.c_str(), tNorm);
			issues.push_back(buf);
		}
		else if (tNorm > maxT)
		{
			sprintf_s(buf, "[单位错误] %s平移 T 模长=%.1f 过大, 疑似单位异常"
				"(典型值: 双目基线50~500mm, 相机到屏幕100~2000mm)", tag.c_str(), tNorm);
			issues.push_back(buf);
		}
	};

	if (m_bBinocLoaded)
	{
		checkIntrinsics(m_fcLeft, m_ccLeft, "左相机 ", imgW, imgH);
		checkIntrinsics(m_fcRight, m_ccRight, "右相机 ", imgW, imgH);
		checkExtrinsics(m_rvecRight, m_tvecRight, "右相机", 10.0, 5000.0);

		if (checkFiniteVec(m_distortLeft, "左相机畸变系数 dist_L", issues))
		{
			double k1 = m_distortLeft.at<double>(0, 0);
			if (fabs(k1) > 1.0)
			{
				sprintf_s(buf, "[畸变异常] 左相机畸变 k1=%.4g 绝对值过大(模拟数据应接近0)", k1);
				issues.push_back(buf);
			}
		}

		if (!checkFiniteVec(m_F, "基本矩阵 F", issues) || cv::norm(m_F) < 1e-12)
		{
			sprintf_s(buf, "[外参异常] 基本矩阵 F 全零或无效, 请检查 cal_res.txt 末尾9个数");
			issues.push_back(buf);
		}

		// 旋转矩阵正交性
		cv::Mat R = rmatRight();
		cv::Mat err = R.t() * R - cv::Mat::eye(3, 3, CV_64FC1);
		double orthoErr = cv::norm(err);
		if (orthoErr > 1e-3)
		{
			sprintf_s(buf, "[外参异常] 右相机旋转矩阵非正交(误差=%.2e), om 值可能损坏", orthoErr);
			issues.push_back(buf);
		}
	}

	if (m_bPrjLoaded)
	{
		checkExtrinsics(m_rvecPrj, m_tvecPrj, "显示屏", 50.0, 5000.0);

		// 像素尺寸校验: 物理屏幕像素尺寸典型 0.01~10 mm/px
		if (!(m_pixelSizeW > 0.0) || !(m_pixelSizeH > 0.0))
		{
			sprintf_s(buf, "[屏幕参数] 像素尺寸 ps=(%.4g, %.4g) mm/px 无效(必须>0), "
				"请检查 cal_Projector.txt 第7~8个数", m_pixelSizeW, m_pixelSizeH);
			issues.push_back(buf);
		}
		else if (m_pixelSizeW < 0.005 || m_pixelSizeW > 20.0 ||
			m_pixelSizeH < 0.005 || m_pixelSizeH > 20.0)
		{
			sprintf_s(buf, "[屏幕参数] 像素尺寸 ps=(%.4g, %.4g) mm/px 量级异常"
				"(典型 0.05~1 mm/px), 请确认单位是否为mm", m_pixelSizeW, m_pixelSizeH);
			issues.push_back(buf);
		}

		// 分辨率校验: 需>0
		if (m_prjResW <= 0 || m_prjResH <= 0)
		{
			sprintf_s(buf, "[屏幕参数] 显示屏分辨率 res=%dx%d 无效, "
				"请检查 cal_Projector.txt 第9~10个数", m_prjResW, m_prjResH);
			issues.push_back(buf);
		}

		// 屏外参四角 z>0 校验: 屏坐标系原点=左上角, 四角局部坐标
		// (0,0,0)/(W*psW,0,0)/(0,H*psH,0)/(W*psW,H*psH,0), 经R*p+T变换到左相机系后 z 必须>0
		if (m_pixelSizeW > 0.0 && m_pixelSizeH > 0.0 && m_prjResW > 0 && m_prjResH > 0 &&
			checkFiniteVec(m_rvecPrj, "显示屏旋转向量 om", issues) &&
			checkFiniteVec(m_tvecPrj, "显示屏平移 T", issues))
		{
			cv::Mat R = rmatPrj();
			double corners[4][2] = {
				{0.0, 0.0},
				{m_prjResW * m_pixelSizeW, 0.0},
				{0.0, m_prjResH * m_pixelSizeH},
				{m_prjResW * m_pixelSizeW, m_prjResH * m_pixelSizeH}
			};
			for (int i = 0; i < 4; ++i)
			{
				cv::Mat p = (cv::Mat_<double>(3, 1) << corners[i][0], corners[i][1], 0.0);
				cv::Mat pc = R * p + m_tvecPrj;
				double z = pc.at<double>(2);
				if (z <= 0.0)
				{
					sprintf_s(buf, "[外参异常] 显示屏第%d角变换到左相机系 z=%.4f<=0, "
						"屏在相机后方或外参错误(屏坐标系原点应为左上角)", i + 1, z);
					issues.push_back(buf);
				}
			}
		}
	}

	return issues;
}

cv::Mat CCalibParas::intrinsicLeft() const
{
	return (cv::Mat_<double>(3, 3) <<
		m_fcLeft.x, 0, m_ccLeft.x,
		0, m_fcLeft.y, m_ccLeft.y,
		0, 0, 1);
}

cv::Mat CCalibParas::intrinsicRight() const
{
	return (cv::Mat_<double>(3, 3) <<
		m_fcRight.x, 0, m_ccRight.x,
		0, m_fcRight.y, m_ccRight.y,
		0, 0, 1);
}


cv::Mat CCalibParas::rmatRight() const
{
	cv::Mat R;
	cv::Rodrigues(m_rvecRight, R);
	return R;
}

cv::Mat CCalibParas::rmatPrj() const
{
	cv::Mat R;
	cv::Rodrigues(m_rvecPrj, R);
	return R;
}

double CCalibParas::baseline() const
{
	return cv::norm(m_tvecRight, cv::NORM_L2);
}