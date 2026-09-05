#include "stdafx.h"
#include "Match.h"
#include <cstdio>
#include <ctime>
#include <sstream>
#include <iomanip>
#include <mutex>
#include <algorithm>
#include <cmath>
#include <windows.h>

CMatch::CMatch()
{
}

CMatch::~CMatch()
{
}

static std::mutex g_reconLogMutex;

void CMatch::setDebugLog(bool on, const std::string &logFile)
{
	m_debugLog = on;
	m_logFile = logFile.empty() ? std::string("recon_debug.log") : logFile;
}

void CMatch::dbg(const std::string &msg)
{
	if (!m_debugLog) return;
	std::lock_guard<std::mutex> lk(g_reconLogMutex);
	std::time_t t = std::time(nullptr);
	std::tm tmv; localtime_s(&tmv, &t);
	char ts[32]; std::strftime(ts, sizeof(ts), "%H:%M:%S", &tmv);
	std::string line = std::string("[RECON ") + ts + "] " + msg + "\n";
	FILE *fp = nullptr;
	if (fopen_s(&fp, m_logFile.c_str(), "a") == 0 && fp) { fputs(line.c_str(), fp); fclose(fp); }
	OutputDebugStringA(line.c_str());
}

// 将标定矩阵预展开为普通数组, 消除热循环内的 cv::Mat 分配与动态矩阵乘法
void CMatch::precomputeCalibCache()
{
	cv::Mat Rr = m_calib->rmatRight();     // 3x3 double
	cv::Mat Tr = m_calib->tvecRight();     // 3x1 double
	for (int r = 0; r < 3; r++)
		for (int c = 0; c < 3; c++)
			m_RrT[r * 3 + c] = Rr.at<double>(c, r);   // 转置(存 R^T 行主序)
	for (int k = 0; k < 3; k++)
		m_Tr3[k] = Tr.at<double>(k);

	cv::Mat Rp = m_calib->rmatPrj();
	cv::Mat Tp = m_calib->tvecPrj();
	for (int r = 0; r < 3; r++)
		for (int c = 0; c < 3; c++)
			m_Rp[r * 3 + c] = Rp.at<double>(r, c);
	for (int k = 0; k < 3; k++)
		m_Tp3[k] = Tp.at<double>(k);

	m_psW = m_calib->pixelSizeW();
	m_psH = m_calib->pixelSizeH();

	glm::vec2 fc = m_calib->fcRight();
	glm::vec2 cc = m_calib->ccRight();
	m_fcRx = fc.x; m_fcRy = fc.y; m_ccRx = cc.x; m_ccRy = cc.y;

	const cv::Mat &kc = m_calib->distortRight();
	for (int k = 0; k < 5; k++) m_kcR[k] = kc.at<double>(k);
}

// 显示屏像素 → 左相机3D (无 cv::Mat 分配; 屏面 z=0 故略去 Rp 第3列)
cv::Vec3f CMatch::displayTo3Dfast(float xS, float yS) const
{
	double xL = (double)xS * m_psW;
	double yL = (double)yS * m_psH;
	double X = m_Rp[0] * xL + m_Rp[1] * yL + m_Tp3[0];
	double Y = m_Rp[3] * xL + m_Rp[4] * yL + m_Tp3[1];
	double Z = m_Rp[6] * xL + m_Rp[7] * yL + m_Tp3[2];
	return cv::Vec3f((float)X, (float)Y, (float)Z);
}

// 左相机3D点 → 右相机像素 (无 cv::Mat 分配)
bool CMatch::projectToRightFast(const cv::Vec3f &P, float &uR, float &vR) const
{
	double d0 = (double)P[0] - m_Tr3[0];
	double d1 = (double)P[1] - m_Tr3[1];
	double d2 = (double)P[2] - m_Tr3[2];
	double x = m_RrT[0] * d0 + m_RrT[1] * d1 + m_RrT[2] * d2;
	double y = m_RrT[3] * d0 + m_RrT[4] * d1 + m_RrT[5] * d2;
	double z = m_RrT[6] * d0 + m_RrT[7] * d1 + m_RrT[8] * d2;
	if (z <= 0) return false;
	double xn = x / z, yn = y / z;
	double k1 = m_kcR[0], k2 = m_kcR[1], p1 = m_kcR[2], p2 = m_kcR[3], k3 = m_kcR[4];
	double r2 = xn * xn + yn * yn;
	double radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
	double dx = 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn);
	double dy = p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn;
	double xd = xn * radial + dx;
	double yd = yn * radial + dy;
	uR = (float)(m_fcRx * xd + m_ccRx);
	vR = (float)(m_fcRy * yd + m_ccRy);
	return uR >= 0 && uR < m_imgW && vR >= 0 && vR < m_imgH;
}

void CMatch::setParas(CCalibParas *calib,
	int imgW, int imgH,
	int prjW, int prjH,
	const std::vector<float> &pitchesV,
	const std::vector<float> &pitchesH,
	float offsetFringe, float offsetFringeH,
	float zMin, float zMax)
{
	m_calib = calib;
	m_imgW = imgW;
	m_imgH = imgH;
	m_prjW = prjW;
	m_prjH = prjH;
	m_pitches = pitchesV;
	m_pitchesH = pitchesH.size() >= 3 ? pitchesH : pitchesV;
	m_offsetFringe = offsetFringe;
	m_offsetFringeH = offsetFringeH;
	m_zMin = zMin;
	m_zMax = zMax;
}

// 预计算左相机每个像素的射线方向 (左相机坐标系, 光心在原点)
// 右相机不需要预计算射线: 重建用 projectToRight 将左相机3D点反投影到右相机像素
void CMatch::precomputeRays()
{
	m_rayDirLeft.create(m_imgH, m_imgW, CV_32FC3);

	// 左相机射线
	std::vector<cv::Point2f> ptsL(m_imgW * m_imgH);
	for (int i = 0; i < m_imgH; i++)
		for (int j = 0; j < m_imgW; j++)
			ptsL[i * m_imgW + j] = cv::Point2f((float)j, (float)i);

	cv::Mat ptsLMat(ptsL);
	cv::undistortPoints(ptsLMat, ptsLMat, m_calib->intrinsicLeft(), m_calib->distortLeft());
	for (int i = 0; i < m_imgH; i++)
	{
		cv::Vec3f *p = m_rayDirLeft.ptr<cv::Vec3f>(i);
		for (int j = 0; j < m_imgW; j++)
		{
			const cv::Point2f &pt = ptsL[i * m_imgW + j];
			p[j] = cv::Vec3f(pt.x, pt.y, 1.0f);
		}
	}
}

void CMatch::getDisplayCoord(const float *phV, const float *phH,
	int idx, float &xS, float &yS) const
{
	float pitchV0 = m_pitches[0];
	float pitchH0 = (m_pitchesH.size() >= 1 ? m_pitchesH[0] : m_pitches[0]);
	float TWO_PI = (float)(2.0 * CV_PI);
	xS = phV[idx] / TWO_PI * pitchV0 - m_offsetFringe;   // 纵向条纹->屏X, 用V最细周期
	yS = phH[idx] / TWO_PI * pitchH0 - m_offsetFringeH;  // 横向条纹->屏Y, 用H最细周期
}

// 显示屏像素坐标 → 左相机坐标系3D点
// 坐标约定(与Blender screen_frame_from_uv配套): 显示屏局部坐标系原点=屏左上角(图像(0,0)像素),
// +X沿像素xS+(图像向右), +Y沿像素yS+(图像向下), Z=X×Y(右手闭合); 屏幕平面z=0
// 像素(xS,yS) → 局部mm坐标(xS*psW, yS*psH, 0) → 经显示屏外参(om/T)变换到左相机坐标系
cv::Vec3f CMatch::displayTo3D(float xS, float yS) const
{
	double xL = xS * m_calib->pixelSizeW();
	double yL = yS * m_calib->pixelSizeH();
	cv::Mat p = (cv::Mat_<double>(3, 1) << xL, yL, 0.0);
	p = m_calib->rmatPrj() * p + m_calib->tvecPrj();
	return cv::Vec3f((float)p.at<double>(0), (float)p.at<double>(1), (float)p.at<double>(2));
}

bool CMatch::projectToRight(const cv::Vec3f &P, float &uR, float &vR) const
{
	cv::Mat R = m_calib->rmatRight();
	cv::Mat T = m_calib->tvecRight();
	cv::Mat p = (cv::Mat_<double>(3, 1) << P[0], P[1], P[2]);

	cv::Mat pRight = R.t() * (p - T);
	double z = pRight.at<double>(2);
	if (z <= 0) return false;

	double xn = pRight.at<double>(0) / z;
	double yn = pRight.at<double>(1) / z;

	const cv::Mat &kc = m_calib->distortRight();
	double k1 = kc.at<double>(0), k2 = kc.at<double>(1);
	double p1 = kc.at<double>(2), p2 = kc.at<double>(3);
	double k3 = kc.at<double>(4);
	double r2 = xn * xn + yn * yn;
	double radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
	double dx = 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn);
	double dy = p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn;
	double xd = xn * radial + dx;
	double yd = yn * radial + dy;

	glm::vec2 fc = m_calib->fcRight();
	glm::vec2 cc = m_calib->ccRight();
	uR = (float)(fc.x * xd + cc.x);
	vR = (float)(fc.y * yd + cc.y);
	return uR >= 0 && uR < m_imgW && vR >= 0 && vR < m_imgH;
}

float CMatch::samplePhase(const float *phase, int W, int H, float x, float y) const
{
	int x0 = (int)floor(x), x1 = x0 + 1;
	int y0 = (int)floor(y), y1 = y0 + 1;
	if (x0 < 0 || y0 < 0 || x1 >= W || y1 >= H) return -10000;
	float dx = x - x0, dy = y - y0;
	float p00 = phase[y0 * W + x0];
	float p01 = phase[y0 * W + x1];
	float p10 = phase[y1 * W + x0];
	float p11 = phase[y1 * W + x1];
	if (p00 < -1000 || p01 < -1000 || p10 < -1000 || p11 < -1000) return -10000;
	return p00 + dx * (p01 - p00) + dy * (p10 - p00) + dx * dy * (p00 + p11 - p01 - p10);
}

cv::Vec3f CMatch::reflectNormal(const cv::Vec3f &P, const cv::Vec3f &S, const cv::Vec3f &O)
{
	// 镜面法向 = 从表面点分别指向"屏(光源)"与"相机"的两单位向量之和(角平分线)
	//   a = (S - P) 指向屏,  b = (O - P) 指向相机,  N ∝ a + b
	// 注意: 入射向量必须是 S - P(指向屏), 而非 P - S, 否则法向错误、左右法向无法对齐
	cv::Vec3f i = S - P;
	cv::Vec3f r = O - P;
	float ni = (float)cv::norm(i), nr = (float)cv::norm(r);
	if (ni < 1e-6f || nr < 1e-6f) return cv::Vec3f(0, 0, 0);
	i /= ni; r /= nr;
	cv::Vec3f N = i + r;
	float nN = (float)cv::norm(N);
	if (nN < 1e-6f) return cv::Vec3f(0, 0, 0);
	return N / nN;
}

// 双目 Deflectometry 重建
bool CMatch::reconstructDeflectometry(
	const std::vector<cv::Mat> &phaseLeftV, const std::vector<cv::Mat> &phaseLeftH,
	const cv::Mat &maskLeftV, const cv::Mat &maskLeftH,
	const std::vector<cv::Mat> &phaseRightV, const std::vector<cv::Mat> &phaseRightH,
	const cv::Mat &maskRightV, const cv::Mat &maskRightH)
{
	if (!m_calib || !m_calib->isBinocLoaded() || !m_calib->isPrjLoaded())
	{
		std::cerr << "重建失败: 标定参数不完整" << std::endl;
		return false;
	}
	if (phaseLeftV.size() < 3 || phaseLeftH.size() < 3 ||
		phaseRightV.size() < 3 || phaseRightH.size() < 3)
	{
		std::cerr << "相位图数量不足" << std::endl;
		return false;
	}

	std::cout << "预计算相机射线..." << std::endl;
	precomputeRays();
	precomputeCalibCache();   // 预展开标定矩阵, 消除热循环内 cv::Mat 分配

	int W = m_imgW, H = m_imgH;
	cv::Vec3f O_L(0, 0, 0);
	cv::Vec3f O_R((float)m_Tr3[0], (float)m_Tr3[1], (float)m_Tr3[2]);

	cv::Mat maskL = maskLeftV & maskLeftH;
	cv::Mat maskR = maskRightV & maskRightH;

	// 掩膜腐蚀: 去掉物体边界处调制不完整的像素(减少边缘翘起伪影)
	if (m_maskErodePx > 0)
	{
		cv::Mat k = cv::getStructuringElement(cv::MORPH_ELLIPSE,
			cv::Size(2 * m_maskErodePx + 1, 2 * m_maskErodePx + 1));
		cv::erode(maskL, maskL, k);
		cv::erode(maskR, maskR, k);
	}

	const float *phLV = phaseLeftV[0].ptr<float>(0);
	const float *phLH = phaseLeftH[0].ptr<float>(0);
	const float *phRV = phaseRightV[0].ptr<float>(0);
	const float *phRH = phaseRightH[0].ptr<float>(0);
	const uchar *mkL = maskL.ptr<uchar>(0);
	const uchar *mkR = maskR.ptr<uchar>(0);

	m_rangeImg = cv::Mat::zeros(H, W, CV_8UC1);
	m_normalImg = cv::Mat::zeros(H, W, CV_8UC3);
	// 每像素结果缓冲: 并行阶段无锁填充, 之后串行汇总为点云
	cv::Mat posMat(H, W, CV_32FC3, cv::Scalar(0, 0, 0));
	cv::Mat nrmMat(H, W, CV_32FC3, cv::Scalar(0, 0, 0));
	cv::Mat okMat = cv::Mat::zeros(H, W, CV_8UC1);
	cv::Mat dotMat(H, W, CV_32F, cv::Scalar(-2.0f));   // 每像素最优法向一致性(诊断用)

	m_processed = 0;
	m_bCancelled = false;
	float tMin = m_zMin, tMax = m_zMax;
	float tStep = (tMax - tMin) / m_coarseSteps;

	// ---- 中心像素验证: 打印一个像素的全部重建中间结果, 便于核对正确性 ----
	if (m_debugLog)
	{
		std::ostringstream os;
		os << "==== 重建标定缓存 ====\n"
		   << "  右相机 O_R=(" << O_R[0] << "," << O_R[1] << "," << O_R[2] << ") mm\n"
		   << "  屏像素尺寸 psW=" << m_psW << " psH=" << m_psH << " mm/px"
		   << " | 屏分辨率 " << m_prjW << "x" << m_prjH << "\n"
		   << "  右相机内参 fx=" << m_fcRx << " fy=" << m_fcRy
		   << " cx=" << m_ccRx << " cy=" << m_ccRy << "\n"
		   << "  搜索: zMin=" << tMin << " zMax=" << tMax
		   << " 粗步数=" << m_coarseSteps << " 精步数=" << m_fineSteps
		   << " 法向一致阈值=" << m_normalDotThresh;
		dbg(os.str());

		// 取图像中心; 若中心不在掩膜内, 向外扩散找最近有效像素
		int ci = H / 2, cj = W / 2;
		if (!mkL[ci * W + cj])
		{
			bool found = false;
			for (int r = 1; r < std::max(H, W) / 2 && !found; r++)
			{
				for (int dy = -r; dy <= r && !found; dy++)
					for (int dx = -r; dx <= r && !found; dx++)
					{
						int yy = H / 2 + dy, xx = W / 2 + dx;
						if (yy >= 0 && yy < H && xx >= 0 && xx < W && mkL[yy * W + xx])
						{ ci = yy; cj = xx; found = true; }
					}
			}
			dbg(found ? "中心像素无效, 改用最近有效像素" : "中心附近未找到有效像素");
		}
		debugReconstructPixel(ci, cj, phLV, phLH, phRV, phRH, mkL, mkR, O_L, O_R);
	}

	std::atomic<int> matchedCnt(0);
	std::atomic<int> rowsDone(0);

	// ---- 并行重建: 各像素独立, 按行区间并行, 写入互不重叠的像素缓冲 ----
	cv::parallel_for_(cv::Range(0, H), [&](const cv::Range &rng)
	{
		for (int i = rng.start; i < rng.end; i++)
		{
			if (isCancelled()) m_bCancelled = true;
			if (m_bCancelled) continue;    // 取消后快速跳过剩余行
			for (int j = 0; j < W; j++)
			{
				int idx = i * W + j;
				if (!mkL[idx]) continue;

				float xS_L, yS_L;
				getDisplayCoord(phLV, phLH, idx, xS_L, yS_L);
				if (xS_L < 0 || xS_L >= m_prjW || yS_L < 0 || yS_L >= m_prjH) continue;
				cv::Vec3f S_L = displayTo3Dfast(xS_L, yS_L);
				cv::Vec3f dL = m_rayDirLeft.at<cv::Vec3f>(i, j);

				// 给定深度 t, 计算左右相机法向一致性(无效候选返回 -2)
				auto evalDot = [&](float t) -> float
				{
					cv::Vec3f P = t * dL;
					float uR, vR;
					if (!projectToRightFast(P, uR, vR)) return -2.0f;
					int uRi = (int)(uR + 0.5f), vRi = (int)(vR + 0.5f);
					if (uRi < 0 || uRi >= W || vRi < 0 || vRi >= H) return -2.0f;
					if (!mkR[vRi * W + uRi]) return -2.0f;
					float pv = samplePhase(phRV, W, H, uR, vR);
					float ph = samplePhase(phRH, W, H, uR, vR);
					if (pv < -1000 || ph < -1000) return -2.0f;
					float xr, yr; getDisplayCoord(&pv, &ph, 0, xr, yr);
					if (xr < 0 || xr >= m_prjW || yr < 0 || yr >= m_prjH) return -2.0f;
					cv::Vec3f S_R = displayTo3Dfast(xr, yr);
					cv::Vec3f nL = reflectNormal(P, S_L, O_L);
					cv::Vec3f nR = reflectNormal(P, S_R, O_R);
					return nL.dot(nR);
				};

				// 粗搜索
				float bestT = -1, bestDot = -2;
				for (int s = 0; s <= m_coarseSteps; s++)
				{
					float t = tMin + s * tStep;
					float d = evalDot(t);
					if (d > bestDot) { bestDot = d; bestT = t; }
				}
				if (bestT < 0) continue;

				// 精搜索(在粗最优附近细化)
				float tStepF = (2 * tStep) / m_fineSteps;
				for (int s = -m_fineSteps / 2; s <= m_fineSteps / 2; s++)
				{
					float t = bestT + s * tStepF;
					if (t < tMin || t > tMax) continue;
					float d = evalDot(t);
					if (d > bestDot) { bestDot = d; bestT = t; }
				}

				// 抛物线亚像素深度细化: 用相邻两点拟合峰值, 消除深度网格量化(点云分层)
				{
					float h = tStepF;
					float dm = evalDot(bestT - h);
					float dp = evalDot(bestT + h);
					if (dm > -1.5f && dp > -1.5f)
					{
						float denom = dm - 2.0f * bestDot + dp;
						if (denom < -1e-7f)   // 有明确峰(凹)时才细化
						{
							float delta = 0.5f * (dm - dp) / denom;   // 单位: h
							if (delta > -1.0f && delta < 1.0f)
							{
								float tr = bestT + delta * h;
								float dr = evalDot(tr);
								if (dr > bestDot) { bestDot = dr; bestT = tr; }
							}
						}
					}
				}

				dotMat.at<float>(i, j) = bestDot;   // 记录最优一致性(分布统计)

				if (bestDot > m_normalDotThresh)
				{
					cv::Vec3f P = bestT * dL;
					float uR, vR;
					projectToRightFast(P, uR, vR);
					float phRV_val = samplePhase(phRV, W, H, uR, vR);
					float phRH_val = samplePhase(phRH, W, H, uR, vR);
					float xS_R, yS_R;
					getDisplayCoord(&phRV_val, &phRH_val, 0, xS_R, yS_R);
					cv::Vec3f S_R = displayTo3Dfast(xS_R, yS_R);
					cv::Vec3f N_L = reflectNormal(P, S_L, O_L);
					cv::Vec3f N_R = reflectNormal(P, S_R, O_R);
					cv::Vec3f N = (N_L + N_R) * 0.5f;
					float nN = (float)cv::norm(N);
					if (nN > 1e-6f) N /= nN;

					posMat.at<cv::Vec3f>(i, j) = P;
					nrmMat.at<cv::Vec3f>(i, j) = N;
					okMat.at<uchar>(i, j) = 255;
					matchedCnt.fetch_add(1, std::memory_order_relaxed);

					int gray = (int)(255 * (P[2] - m_zMin) / (m_zMax - m_zMin));
					if (gray < 0) gray = 0; if (gray > 255) gray = 255;
					m_rangeImg.at<uchar>(i, j) = (uchar)gray;
					m_normalImg.at<cv::Vec3b>(i, j) = cv::Vec3b(
						(uchar)((N[2] * 0.5f + 0.5f) * 255),
						(uchar)((N[1] * 0.5f + 0.5f) * 255),
						(uchar)((N[0] * 0.5f + 0.5f) * 255));
				}
			}
			int rd = rowsDone.fetch_add(1, std::memory_order_relaxed) + 1;
			if (m_progressCb && (rd % 8 == 0 || rd == H))
				m_progressCb(rd, H, matchedCnt.load(std::memory_order_relaxed));
		}
	});

	// 法向一致性分布统计: 帮助判断"点数太少"是阈值过严还是对应关系错误
	if (m_debugLog)
	{
		const float thr[] = { 0.90f, 0.95f, 0.98f, 0.99f, 0.995f, 0.999f, 0.9999f };
		long long cand = 0, cnt[7] = {0};
		double dmax = -2.0;
		for (int i = 0; i < H; i++)
		{
			const float *dr = dotMat.ptr<float>(i);
			for (int j = 0; j < W; j++)
			{
				if (dr[j] < -1.5f) continue;    // 无有效候选
				cand++;
				if (dr[j] > dmax) dmax = dr[j];
				for (int k = 0; k < 7; k++) if (dr[j] >= thr[k]) cnt[k]++;
			}
		}
		std::ostringstream os;
		os << "==== 法向一致性(dot)分布 ====\n"
		   << "  有候选像素=" << cand << " 最大dot=" << std::fixed << std::setprecision(5) << dmax
		   << " | 当前阈值=" << m_normalDotThresh << "\n  各阈值可通过点数:";
		for (int k = 0; k < 7; k++)
			os << "\n    dot>=" << std::setprecision(4) << thr[k] << " : " << cnt[k]
			   << " (" << std::setprecision(1) << (cand ? 100.0 * cnt[k] / cand : 0.0) << "%)";
		os << "\n  提示: 若在较低阈值处点数骤增, 说明阈值过严, 可下调 m_normalDotThresh;"
		   << "\n        若各阈值点数都很少, 则多为对应/外参/深度范围问题。";
		dbg(os.str());
	}

	// ---- 深度邻域中值离群剔除: 去掉边缘翘起点与孤立飞点 ----
	// 翘边点常与左右两路"同样偏差"→dot仍高、阈值挡不住, 需用空间一致性剔除。
	if (m_depthMedRadius > 0)
	{
		int R = m_depthMedRadius;
		cv::Mat okIn = okMat.clone();
		cv::parallel_for_(cv::Range(0, H), [&](const cv::Range &rng)
		{
			std::vector<float> nb;
			nb.reserve((2 * R + 1) * (2 * R + 1));
			for (int i = rng.start; i < rng.end; i++)
			{
				const uchar *okr = okIn.ptr<uchar>(i);
				for (int j = 0; j < W; j++)
				{
					if (!okr[j]) continue;
					nb.clear();
					for (int dy = -R; dy <= R; dy++)
					{
						int yy = i + dy; if (yy < 0 || yy >= H) continue;
						const uchar *okn = okIn.ptr<uchar>(yy);
						const cv::Vec3f *pn = posMat.ptr<cv::Vec3f>(yy);
						for (int dx = -R; dx <= R; dx++)
						{
							int xx = j + dx; if (xx < 0 || xx >= W) continue;
							if (okn[xx]) nb.push_back(pn[xx][2]);
						}
					}
					bool reject = false;
					if ((int)nb.size() < m_depthMedMinCnt) reject = true;   // 孤立飞点
					else
					{
						std::nth_element(nb.begin(), nb.begin() + nb.size() / 2, nb.end());
						float med = nb[nb.size() / 2];
						if (std::fabs(posMat.at<cv::Vec3f>(i, j)[2] - med) > m_depthMedTol)
							reject = true;   // 与邻域中值深度偏差过大(翘边/离群)
					}
					if (reject)
					{
						okMat.at<uchar>(i, j) = 0;
						m_rangeImg.at<uchar>(i, j) = 0;
						m_normalImg.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
					}
				}
			}
		});
	}

	// 串行汇总: 统计处理像素数并构建点云(顺序确定, 结果可复现)
	m_surface.clear();
	m_surface.reserve(matchedCnt.load());
	int processed = 0;
	for (int i = 0; i < H; i++)
	{
		const uchar *okr = okMat.ptr<uchar>(i);
		const cv::Vec3f *pr = posMat.ptr<cv::Vec3f>(i);
		const cv::Vec3f *nr = nrmMat.ptr<cv::Vec3f>(i);
		for (int j = 0; j < W; j++)
		{
			if (mkL[i * W + j]) processed++;
			if (okr[j])
			{
				SurfacePoint sp;
				sp.pos.x = pr[j][0]; sp.pos.y = pr[j][1]; sp.pos.z = pr[j][2];
				sp.normal.x = nr[j][0]; sp.normal.y = nr[j][1]; sp.normal.z = nr[j][2];
				m_surface.push_back(sp);
			}
		}
	}
	m_processed = processed;
	int matched = matchedCnt.load();

	if (m_bCancelled)
	{
		std::cout << "重建被用户终止, 已匹配 " << matched << " 点" << std::endl;
		if (m_debugLog) dbg("重建被用户终止, 已匹配 " + std::to_string(matched) + " 点");
		return false;
	}
	if (m_debugLog)
		dbg("重建完成: 处理像素=" + std::to_string(processed) + " 重建点数=" + std::to_string(matched));
	std::cout << "处理像素: " << processed << "  重建点数: " << matched << std::endl;
	return !m_surface.empty();
}

// 对单个像素执行完整重建(串行), 逐步打印中间结果, 用于验证重建正确性
void CMatch::debugReconstructPixel(int i, int j,
	const float *phLV, const float *phLH,
	const float *phRV, const float *phRH,
	const uchar *mkL, const uchar *mkR,
	const cv::Vec3f &O_L, const cv::Vec3f &O_R)
{
	int W = m_imgW, H = m_imgH;
	int idx = i * W + j;
	std::ostringstream os;
	os << std::fixed << std::setprecision(4);
	os << "==== 中心像素重建验证 (i=" << i << ", j=" << j << ") ====";

	if (!mkL[idx]) { os << "\n  左相机掩膜: 无效(跳过)"; dbg(os.str()); return; }

	float xS_L, yS_L;
	getDisplayCoord(phLV, phLH, idx, xS_L, yS_L);
	os << "\n  左相位: phV=" << phLV[idx] << " phH=" << phLH[idx]
	   << " -> 屏坐标 S_L像素=(" << xS_L << ", " << yS_L << ")";
	if (xS_L < 0 || xS_L >= m_prjW || yS_L < 0 || yS_L >= m_prjH)
	{ os << "  [越界, 跳过]"; dbg(os.str()); return; }
	cv::Vec3f S_L = displayTo3Dfast(xS_L, yS_L);
	cv::Vec3f dL = m_rayDirLeft.at<cv::Vec3f>(i, j);
	os << "\n  S_L(3D,左相机系)=(" << S_L[0] << ", " << S_L[1] << ", " << S_L[2] << ") mm"
	   << "\n  左相机射线方向 dL=(" << dL[0] << ", " << dL[1] << ", " << dL[2] << ")";

	float tMin = m_zMin, tMax = m_zMax;
	float tStep = (tMax - tMin) / m_coarseSteps;
	float bestT = -1, bestDot = -2;
	int coarseValid = 0;
	for (int s = 0; s <= m_coarseSteps; s++)
	{
		float t = tMin + s * tStep;
		cv::Vec3f P = t * dL;
		float uR, vR;
		if (!projectToRightFast(P, uR, vR)) continue;
		int uRi = (int)(uR + 0.5f), vRi = (int)(vR + 0.5f);
		if (uRi < 0 || uRi >= W || vRi < 0 || vRi >= H) continue;
		if (!mkR[vRi * W + uRi]) continue;
		float phRV_val = samplePhase(phRV, W, H, uR, vR);
		float phRH_val = samplePhase(phRH, W, H, uR, vR);
		if (phRV_val < -1000 || phRH_val < -1000) continue;
		float xS_R, yS_R;
		getDisplayCoord(&phRV_val, &phRH_val, 0, xS_R, yS_R);
		if (xS_R < 0 || xS_R >= m_prjW || yS_R < 0 || yS_R >= m_prjH) continue;
		cv::Vec3f S_R = displayTo3Dfast(xS_R, yS_R);
		cv::Vec3f N_L = reflectNormal(P, S_L, O_L);
		cv::Vec3f N_R = reflectNormal(P, S_R, O_R);
		float dot = N_L.dot(N_R);
		coarseValid++;
		if (dot > bestDot) { bestDot = dot; bestT = t; }
	}
	os << "\n  粗搜索: 有效采样=" << coarseValid << "/" << (m_coarseSteps + 1)
	   << " 最优 t=" << bestT << " 法向一致性 dot=" << bestDot;

	if (bestT > 0)
	{
		float tStepF = (2 * tStep) / m_fineSteps;
		for (int s = -m_fineSteps / 2; s <= m_fineSteps / 2; s++)
		{
			float t = bestT + s * tStepF;
			if (t < tMin || t > tMax) continue;
			cv::Vec3f P = t * dL;
			float uR, vR;
			if (!projectToRightFast(P, uR, vR)) continue;
			int uRi = (int)(uR + 0.5f), vRi = (int)(vR + 0.5f);
			if (uRi < 0 || uRi >= W || vRi < 0 || vRi >= H) continue;
			if (!mkR[vRi * W + uRi]) continue;
			float phRV_val = samplePhase(phRV, W, H, uR, vR);
			float phRH_val = samplePhase(phRH, W, H, uR, vR);
			if (phRV_val < -1000 || phRH_val < -1000) continue;
			float xS_R, yS_R;
			getDisplayCoord(&phRV_val, &phRH_val, 0, xS_R, yS_R);
			if (xS_R < 0 || xS_R >= m_prjW || yS_R < 0 || yS_R >= m_prjH) continue;
			cv::Vec3f S_R = displayTo3Dfast(xS_R, yS_R);
			cv::Vec3f N_L = reflectNormal(P, S_L, O_L);
			cv::Vec3f N_R = reflectNormal(P, S_R, O_R);
			float dot = N_L.dot(N_R);
			if (dot > bestDot) { bestDot = dot; bestT = t; }
		}
		os << "\n  精搜索后: 最优 t=" << bestT << " dot=" << bestDot;

		// 最终结果
		cv::Vec3f P = bestT * dL;
		float uR, vR; projectToRightFast(P, uR, vR);
		float phRV_val = samplePhase(phRV, W, H, uR, vR);
		float phRH_val = samplePhase(phRH, W, H, uR, vR);
		float xS_R, yS_R; getDisplayCoord(&phRV_val, &phRH_val, 0, xS_R, yS_R);
		cv::Vec3f S_R = displayTo3Dfast(xS_R, yS_R);
		cv::Vec3f N_L = reflectNormal(P, S_L, O_L);
		cv::Vec3f N_R = reflectNormal(P, S_R, O_R);
		os << "\n  表面点 P=(" << P[0] << ", " << P[1] << ", " << P[2] << ") mm (深度z=" << P[2] << ")"
		   << "\n  右相机投影像素=(" << uR << ", " << vR << ")"
		   << "\n  S_R(3D)=(" << S_R[0] << ", " << S_R[1] << ", " << S_R[2] << ") mm"
		   << "\n  法向 N_L=(" << N_L[0] << ", " << N_L[1] << ", " << N_L[2] << ")"
		   << "\n  法向 N_R=(" << N_R[0] << ", " << N_R[1] << ", " << N_R[2] << ")"
		   << "\n  最终 dot=" << bestDot << " vs 阈值 " << m_normalDotThresh
		   << " -> " << (bestDot > m_normalDotThresh ? "接受(重建成功)" : "拒绝(法向不一致)");
	}
	else
	{
		os << "\n  粗搜索未找到任何有效候选(右相机投影均越界或无相位)";
	}
	dbg(os.str());
}
