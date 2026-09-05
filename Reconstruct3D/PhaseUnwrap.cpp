#include "stdafx.h"
#include "PhaseUnwrap.h"
#include <cstdio>
#include <ctime>
#include <sstream>
#include <iomanip>
#include <windows.h>

CPhaseUnwrap::CPhaseUnwrap()
{
}

void CPhaseUnwrap::setDebugLog(bool on, const std::string &logFile)
{
	m_debugLog = on;
	m_logFile = logFile.empty() ? std::string("phase_unwrap_debug.log") : logFile;
}

// 线程安全的诊断日志: 追加到日志文件, 同时发送到调试器输出窗口
void CPhaseUnwrap::dbg(const std::string &msg)
{
	if (!m_debugLog) return;
	std::lock_guard<std::mutex> lk(m_logMutex);

	// 时间戳
	std::time_t t = std::time(nullptr);
	std::tm tmv;
	localtime_s(&tmv, &t);
	char ts[32];
	std::strftime(ts, sizeof(ts), "%H:%M:%S", &tmv);

	std::string line = std::string("[PHASE ") + ts + "] " + msg + "\n";

	FILE *fp = nullptr;
	if (fopen_s(&fp, m_logFile.c_str(), "a") == 0 && fp)
	{
		fputs(line.c_str(), fp);
		fclose(fp);
	}
	OutputDebugStringA(line.c_str());
}

CPhaseUnwrap::~CPhaseUnwrap()
{
}

void CPhaseUnwrap::init(int imgWidth, int imgHeight,
	int prjWidth, int prjHeight,
	int numFreq, int numStep,
	const std::vector<float> &pitchesV,
	const std::vector<float> &pitchesH,
	float offsetFringe, float offsetFringeH,
	float maxFringeIntensity)
{
	m_width = imgWidth;
	m_height = imgHeight;
	m_prjWidth = prjWidth;
	m_prjHeight = prjHeight;
	m_numFreq = numFreq;
	m_numStep = numStep;
	m_pitches = pitchesV;
	m_pitchesH = pitchesH.size() >= 3 ? pitchesH : pitchesV;   // 未提供H则与V相同
	m_offsetFringe = offsetFringe;
	m_offsetFringeH = offsetFringeH;
	m_maxFringeIntensity = maxFringeIntensity;

	m_phaseLeftV.resize(m_numFreq);
	m_phaseRightV.resize(m_numFreq);
	m_phaseLeftH.resize(m_numFreq);
	m_phaseRightH.resize(m_numFreq);

	if (m_debugLog)
	{
		std::ostringstream os;
		os << "==== init ====\n"
		   << "  cam=" << m_width << "x" << m_height
		   << " prj=" << m_prjWidth << "x" << m_prjHeight
		   << " numFreq=" << m_numFreq << " numStep=" << m_numStep << "\n"
		   << "  offsetFringe(V)=" << m_offsetFringe
		   << " offsetFringe(H)=" << m_offsetFringeH
		   << " maxFringeIntensity=" << m_maxFringeIntensity << "\n"
		   << "  屏幕 " << m_prjWidth << "x" << m_prjHeight;
		// 对 V(沿屏宽) 和 H(沿屏高) 分别打印周期与合成波长/歧义检查
		auto logPitch = [&](const std::vector<float> &pit, const char *dir, int span)
		{
			os << "\n  [" << dir << "] pitches=[";
			for (size_t i = 0; i < pit.size(); i++) os << pit[i] << (i + 1 < pit.size() ? "," : "");
			os << "]";
			if (pit.size() >= 3)
			{
				float P1 = pit[0], P2 = pit[1], P3 = pit[2];
				float P12 = P1 * P2 / std::fabs(P2 - P1);
				float P23 = P2 * P3 / std::fabs(P3 - P2);
				float P123 = P12 * P23 / std::fabs(P23 - P12);
				os << " 合成波长 P123=" << P123 << " (需>=该方向屏幅 " << span << ")";
				if (P123 < span)
					os << " [警告] P123<屏幅, 存在整周期歧义, 解相会出错!";
				os << " | 展开链放大 P123/P12=" << (P123 / P12) << " P12/P1=" << (P12 / P1)
				   << " 最细周期=" << P1 << "px";
				if (P1 < 4.0f) os << " [警告]<4px采样不足";
			}
		};
		logPitch(m_pitches, "V沿屏宽", m_prjWidth);
		logPitch(m_pitchesH, "H沿屏高", m_prjHeight);
		dbg(os.str());
	}
}

bool CPhaseUnwrap::loadImages(const std::string &folder, int poseIndex,
	const std::string &side, std::vector<cv::Mat> &imgs)
{
	imgs.clear();
	for (int i = 0; i < 24; i++)
	{
		char buf[512];
		sprintf_s(buf, "%s\\pose%d_%s_img%d.bmp", folder.c_str(), poseIndex, side.c_str(), i);
		cv::Mat img = cv::imread(buf, cv::IMREAD_GRAYSCALE);
		if (img.empty()) break;
		if (img.cols != m_width || img.rows != m_height)
			cv::resize(img, img, cv::Size(m_width, m_height));
		imgs.push_back(img);
	}
	return imgs.size() >= 12;
}

bool CPhaseUnwrap::calculatePhasesFromFile(const std::string &folder, int poseIndex)
{
	std::vector<cv::Mat> imgsLeft, imgsRight;
	if (!loadImages(folder, poseIndex, "left", imgsLeft))  return false;
	if (!loadImages(folder, poseIndex, "right", imgsRight)) return false;
	bool hasV = true;
	bool hasH = (imgsLeft.size() >= 24 && imgsRight.size() >= 24);
	return calculatePhases(imgsLeft, imgsRight, hasV, hasH);
}

bool CPhaseUnwrap::calculatePhases(const std::vector<cv::Mat> &leftImgs, const std::vector<cv::Mat> &rightImgs,
	bool hasV, bool hasH)
{
	int nPerCamera = (int)leftImgs.size();
	if (nPerCamera < 12 || (int)rightImgs.size() < 12)
	{
		std::cerr << "图像数量不足, 需要12幅/相机, 实际左:" << leftImgs.size() << " 右:" << rightImgs.size() << std::endl;
		return false;
	}

	m_hasV = hasV;
	m_hasH = hasH;

	if (!m_hasV && !m_hasH)
	{
		std::cerr << "纵向和横向条纹均不可用" << std::endl;
		return false;
	}

	bool okL = false, okR = false;
	m_bCancelled = false;
	dbg("==== calculatePhases (双目) ====  hasV=" + std::to_string(m_hasV) +
		" hasH=" + std::to_string(m_hasH) +
		" 左图数=" + std::to_string(leftImgs.size()) +
		" 右图数=" + std::to_string(rightImgs.size()));
	std::thread tL([&]() {
		okL = processOneCamera(leftImgs, m_phaseLeftV, m_phaseLeftH, m_maskLeftV, m_maskLeftH, m_hasV, m_hasH, "left");
	});
	std::thread tR([&]() {
		okR = processOneCamera(rightImgs, m_phaseRightV, m_phaseRightH, m_maskRightV, m_maskRightH, m_hasV, m_hasH, "right");
	});
	tL.join();
	tR.join();
	if (m_progressCb) m_progressCb(100);
	return okL && okR;
}

bool CPhaseUnwrap::calculatePhasesSingle(const std::vector<cv::Mat> &imgs,
	bool hasV, bool hasH)
{
	if (imgs.size() < 12)
	{
		std::cerr << "图像数量不足, 需要12幅, 实际:" << imgs.size() << std::endl;
		return false;
	}

	m_hasV = hasV;
	m_hasH = hasH;

	if (!m_hasV && !m_hasH)
	{
		std::cerr << "纵向和横向条纹均不可用" << std::endl;
		return false;
	}

	m_bCancelled = false;
	dbg("==== calculatePhasesSingle (单相机) ====  hasV=" + std::to_string(m_hasV) +
		" hasH=" + std::to_string(m_hasH) + " 图数=" + std::to_string(imgs.size()));
	bool ok = processOneCamera(imgs, m_phaseLeftV, m_phaseLeftH, m_maskLeftV, m_maskLeftH, m_hasV, m_hasH, "single");
	if (!ok) return false;

	m_phaseRightV.resize(m_numFreq);
	m_phaseRightH.resize(m_numFreq);
	for (int i = 0; i < m_numFreq; i++)
	{
		m_phaseRightV[i] = cv::Mat::zeros(m_height, m_width, CV_32FC1);
		m_phaseRightH[i] = cv::Mat::zeros(m_height, m_width, CV_32FC1);
	}
	m_maskRightV = cv::Mat::zeros(m_height, m_width, CV_8UC1);
	m_maskRightH = cv::Mat::zeros(m_height, m_width, CV_8UC1);
	return true;
}

bool CPhaseUnwrap::processOneCamera(const std::vector<cv::Mat> &imgs,
	std::vector<cv::Mat> &phaseV, std::vector<cv::Mat> &phaseH,
	cv::Mat &maskV, cv::Mat &maskH,
	bool hasV, bool hasH, const std::string &side)
{
	int indexOffset = m_numFreq * m_numStep;   // 12 横向条纹起始索引
	int nImgs = (int)imgs.size();

	// 逐图统计: 均值/最小/最大 (用于检测图像顺序错乱、过曝、图像内容异常)
	if (m_debugLog)
	{
		std::ostringstream os;
		os << "---- [" << side << "] 逐幅图像强度统计 (共" << nImgs << "幅) ----";
		for (int k = 0; k < nImgs; k++)
		{
			if (imgs[k].empty()) { os << "\n  img" << k << ": <空>"; continue; }
			double mn, mx; cv::Scalar mean = cv::mean(imgs[k]);
			cv::minMaxLoc(imgs[k], &mn, &mx);
			// 频/步/方向标注
			int grp = k / m_numStep;          // 第几组4步
			int step = k % m_numStep;
			const char *dir = (k < indexOffset) ? "V" : "H";
			int freq = (k < indexOffset) ? grp : (grp - m_numFreq);
			os << "\n  img" << std::setw(2) << k
			   << " [" << dir << " f" << freq << " step" << step << "]"
			   << " mean=" << std::fixed << std::setprecision(1) << mean[0]
			   << " min=" << mn << " max=" << mx;
			if (mx >= m_maxFringeIntensity)
				os << "  <过曝: max>=maxFringeIntensity=" << m_maxFringeIntensity
				   << ", 该阈值会把亮条纹峰当作无效点剔除>";
		}
		dbg(os.str());
	}

	// 纵向条纹 3频
	if (hasV)
	{
		std::vector<cv::Mat> maskVTmp(m_numFreq);
		for (int i = 0; i < m_numFreq; i++)
		{
			if (isCancelled()) { m_bCancelled = true; return false; }
			int base = i * m_numStep;
			if (base + 3 < nImgs && !imgs[base + 0].empty() && !imgs[base + 3].empty())
			{
				getWrappedPhase(
					imgs[base + 0], imgs[base + 1],
					imgs[base + 2], imgs[base + 3],
					phaseV[i], maskVTmp[i], 100,
					side + " V f" + std::to_string(i) + " (img" +
					std::to_string(base) + "~" + std::to_string(base + 3) + ")");
			}
			else
			{
				phaseV[i] = cv::Mat::zeros(m_height, m_width, CV_32FC1);
				maskVTmp[i] = cv::Mat::zeros(m_height, m_width, CV_8UC1);
			}
		}
		unwrapPhase(phaseV, maskVTmp, maskV, m_pitches, side + " V");
	}
	else
	{
		maskV = cv::Mat::zeros(m_height, m_width, CV_8UC1);
		for (int i = 0; i < m_numFreq; i++)
			phaseV[i] = cv::Mat::zeros(m_height, m_width, CV_32FC1);
	}

	// 横向条纹 3频
	if (hasH)
	{
		int hBase = (nImgs >= 24) ? indexOffset : 0;
		std::vector<cv::Mat> maskHTmp(m_numFreq);
		for (int i = 0; i < m_numFreq; i++)
		{
			if (isCancelled()) { m_bCancelled = true; return false; }
			int base = hBase + i * m_numStep;
			if (base + 3 < nImgs && !imgs[base + 0].empty() && !imgs[base + 3].empty())
			{
				getWrappedPhase(
					imgs[base + 0], imgs[base + 1],
					imgs[base + 2], imgs[base + 3],
					phaseH[i], maskHTmp[i], 100,
					side + " H f" + std::to_string(i) + " (img" +
					std::to_string(base) + "~" + std::to_string(base + 3) + ")");
			}
			else
			{
				phaseH[i] = cv::Mat::zeros(m_height, m_width, CV_32FC1);
				maskHTmp[i] = cv::Mat::zeros(m_height, m_width, CV_8UC1);
			}
		}
		unwrapPhase(phaseH, maskHTmp, maskH, m_pitchesH, side + " H");
	}
	else
	{
		maskH = cv::Mat::zeros(m_height, m_width, CV_8UC1);
		for (int i = 0; i < m_numFreq; i++)
			phaseH[i] = cv::Mat::zeros(m_height, m_width, CV_32FC1);
	}
	return true;
}

// 四步相移: I_n = I0 + Im*cos(phi + 2*pi*n/4)
//   tan(phi) = (I4 - I2) / (I1 - I3)
// 调制度平方 B = (I4-I2)^2 + (I1-I3)^2 (未开方, 阈值 modThresh 按平方量纲比较)
bool CPhaseUnwrap::getWrappedPhase(const cv::Mat &img1, const cv::Mat &img2,
	const cv::Mat &img3, const cv::Mat &img4,
	cv::Mat &phase, cv::Mat &mask, int modThresh,
	const std::string &label)
{
	phase = cv::Mat::zeros(m_height, m_width, CV_32FC1);
	mask = cv::Mat::zeros(m_height, m_width, CV_8UC1);

	// 诊断累计量
	long long nValid = 0, nSat = 0, nLowMod = 0;
	double sumMod = 0.0, maxMod = 0.0;
	float phMin = 1e9f, phMax = -1e9f;

	for (int i = 0; i < m_height; i++)
	{
		const uchar *p1 = img1.ptr<uchar>(i);
		const uchar *p2 = img2.ptr<uchar>(i);
		const uchar *p3 = img3.ptr<uchar>(i);
		const uchar *p4 = img4.ptr<uchar>(i);
		float  *pp = phase.ptr<float>(i);
		uchar  *pm = mask.ptr<uchar>(i);
		for (int j = 0; j < m_width; j++)
		{
			if (p1[j] < m_maxFringeIntensity && p2[j] < m_maxFringeIntensity &&
				p3[j] < m_maxFringeIntensity && p4[j] < m_maxFringeIntensity)
			{
				float a = (float)(p4[j] - p2[j]);
				float b = (float)(p1[j] - p3[j]);
				float B = a * a + b * b;
				if (B > modThresh)
				{
					pm[j] = 255;
					pp[j] = atan2f(a, b);
					if (m_debugLog)
					{
						nValid++;
						double bb = std::sqrt((double)B);
						sumMod += bb; if (bb > maxMod) maxMod = bb;
						if (pp[j] < phMin) phMin = pp[j];
						if (pp[j] > phMax) phMax = pp[j];
					}
				}
				else if (m_debugLog) nLowMod++;
			}
			else if (m_debugLog) nSat++;
		}
	}

	if (m_debugLog)
	{
		double total = (double)m_width * m_height;
		std::ostringstream os;
		os << "包裹相位 [" << label << "]: 有效=" << nValid
		   << " (" << std::fixed << std::setprecision(1) << (100.0 * nValid / total) << "%)"
		   << " 过曝剔除=" << nSat
		   << " 低调制剔除=" << nLowMod
		   << " | 调制幅值 均值=" << std::setprecision(1) << (nValid ? sumMod / nValid : 0.0)
		   << " 最大=" << maxMod
		   << " | 包裹相位范围=[" << std::setprecision(3)
		   << (nValid ? phMin : 0.0f) << "," << (nValid ? phMax : 0.0f) << "] (理论应≈[-π,π])";
		if (nValid == 0)
			os << "  [错误] 无有效像素! 检查图像顺序/过曝阈值/调制阈值";
		dbg(os.str());
	}
	return true;
}

// 三频四步多频外差法相位解包裹
// 参考 Stereo/PhaseUnwrap.cpp PhaseUnwrap()
//   合成频率周期:
//     P12 = P1*P2/|P2-P1|
//     P23 = P2*P3/|P3-P2|
//     P123= P12*P23/|P23-P12|
//   由合成相位反推各频率展开相位
bool CPhaseUnwrap::unwrapPhase(std::vector<cv::Mat> &wrappedPhases,
	const std::vector<cv::Mat> &masks, cv::Mat &outMask,
	const std::vector<float> &pitches,
	const std::string &label)
{
	outMask = cv::Mat::zeros(m_height, m_width, CV_8UC1);

	// 合成相位缓冲
	cv::Mat phase12  = cv::Mat::zeros(m_height, m_width, CV_32FC1);
	cv::Mat phase23  = cv::Mat::zeros(m_height, m_width, CV_32FC1);
	cv::Mat phase123 = cv::Mat::zeros(m_height, m_width, CV_32FC1);

	// 条纹周期(按方向传入: V用纵向, H用横向)
	float P1 = pitches[0], P2 = pitches[1], P3 = pitches[2];
	float P12  = P1 * P2 / std::fabs(P2 - P1);
	float P23  = P2 * P3 / std::fabs(P3 - P2);
	float P123 = P12 * P23 / std::fabs(P23 - P12);

	const float TWO_PI = (float)(2.0 * CV_PI);

	// 诊断: 展开相位统计
	long long nUnwrap = 0;
	float aMin = 1e9f, aMax = -1e9f;      // 最细频率(p1)绝对相位范围
	double aSum = 0.0;
	int nSampleLogged = 0;
	const int kMaxSample = 5;
	std::ostringstream sampleOs;

	for (int i = 0; i < m_height; i++)
	{
		float *p1  = wrappedPhases[0].ptr<float>(i);
		float *p2  = wrappedPhases[1].ptr<float>(i);
		float *p3  = wrappedPhases[2].ptr<float>(i);
		float *p12 = phase12.ptr<float>(i);
		float *p23 = phase23.ptr<float>(i);
		float *p123e = phase123.ptr<float>(i);
		const uchar *m1 = masks[0].ptr<uchar>(i);
		const uchar *m2 = masks[1].ptr<uchar>(i);
		const uchar *m3 = masks[2].ptr<uchar>(i);
		uchar *pm = outMask.ptr<uchar>(i);
		for (int j = 0; j < m_width; j++)
		{
			if (m1[j] && m2[j] && m3[j])
			{
				pm[j] = 255;
				float w1 = *p1, w2 = *p2, w3 = *p3;   // 保存原始包裹相位用于采样日志
				*p12  = heterodyne(*p1, *p2);
				*p23  = heterodyne(*p2, *p3);
				*p123e = heterodyne(*p12, *p23);
				float phi123 = *p123e;

				// 由合成相位展开中间频率
				*p12 += TWO_PI * floorf((*p123e * P123 / P12 - *p12) / TWO_PI + 0.5f);
				*p23 += TWO_PI * floorf((*p123e * P123 / P23 - *p23) / TWO_PI + 0.5f);

				// 由中间频率展开原始频率
				*p1 += TWO_PI * floorf((*p12 * P12 / P1 - *p1) / TWO_PI + 0.5f);
				*p2 += TWO_PI * floorf((*p12 * P12 / P2 - *p2) / TWO_PI + 0.5f);
				*p3 += TWO_PI * floorf((*p23 * P23 / P3 - *p3) / TWO_PI + 0.5f);

				if (m_debugLog)
				{
					nUnwrap++;
					aSum += *p1;
					if (*p1 < aMin) aMin = *p1;
					if (*p1 > aMax) aMax = *p1;
					// 沿中间行采样几个点, 便于人工核对展开链
					if (i == m_height / 2 && nSampleLogged < kMaxSample &&
						(j % (m_width / (kMaxSample + 1) + 1) == 0))
					{
						sampleOs << "\n    (x=" << j << ") wrap[" << std::fixed << std::setprecision(2)
						   << w1 << "," << w2 << "," << w3 << "]"
						   << " phi123=" << phi123
						   << " -> abs[p1=" << *p1 << " p2=" << *p2 << " p3=" << *p3 << "]";
						nSampleLogged++;
					}
				}
			}
			else
			{
				*p1 = -10000; *p2 = -10000; *p3 = -10000;
				pm[j] = 0;
			}
			p1++; p2++; p3++; p12++; p23++; p123e++;
		}
	}

	if (m_debugLog)
	{
		double total = (double)m_width * m_height;
		// 期望的最细频率绝对相位跨度 ≈ 2π * 视场/P1
		int span = (label.size() && label.back() == 'H') ? m_height : m_width;
		float expMax = TWO_PI * span / P1;
		std::ostringstream os;
		os << "展开相位 [" << label << "]: 3频均有效像素=" << nUnwrap
		   << " (" << std::fixed << std::setprecision(1) << (100.0 * nUnwrap / total) << "%)"
		   << "\n  P1=" << std::setprecision(2) << P1 << " P2=" << P2 << " P3=" << P3
		   << " | P12=" << P12 << " P23=" << P23 << " P123=" << P123
		   << "\n  最细频率绝对相位范围=[" << std::setprecision(2) << (nUnwrap ? aMin : 0.0f)
		   << "," << (nUnwrap ? aMax : 0.0f) << "]"
		   << " 均值=" << (nUnwrap ? aSum / nUnwrap : 0.0)
		   << " | 理论跨度≈[0," << expMax << "] (2π·视场" << span << "/P1)";
		if (nUnwrap && (aMax - aMin) > expMax * 1.5f)
			os << "\n  [警告] 实际相位跨度远大于理论值, 说明展开出现整周期跳变错误(多频外差噪声放大)";
		if (nSampleLogged > 0)
			os << "\n  采样点(中间行):" << sampleOs.str();
		dbg(os.str());
	}
	return true;
}