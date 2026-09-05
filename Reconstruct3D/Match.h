#pragma once

#include <atomic>
#include <functional>
#include "CalibParas.h"
#include "PhaseUnwrap.h"
#include "PointCloudView.h"

struct SurfacePoint
{
	Point3D pos;
	Point3D normal;
};

class CMatch
{
public:
	CMatch();
	~CMatch();

	// 设置标定参数
	void setParas(CCalibParas *calib,
		int imgW, int imgH,
		int prjW, int prjH,
		const std::vector<float> &pitchesV,
		const std::vector<float> &pitchesH,
		float offsetFringe, float offsetFringeH,
		float zMin, float zMax);

	// 双目 Deflectometry 重建
	// 需要纵向(→显示屏x)和横向(→显示屏y)两组相位
	bool reconstructDeflectometry(
		const std::vector<cv::Mat> &phaseLeftV,   // 左相机纵向条纹展开相位(3频)
		const std::vector<cv::Mat> &phaseLeftH,   // 左相机横向条纹展开相位(3频)
		const cv::Mat &maskLeftV, const cv::Mat &maskLeftH,
		const std::vector<cv::Mat> &phaseRightV,  // 右相机纵向条纹展开相位(3频)
		const std::vector<cv::Mat> &phaseRightH,  // 右相机横向条纹展开相位(3频)
		const cv::Mat &maskRightV, const cv::Mat &maskRightH);

	// 获取重建结果
	const std::vector<SurfacePoint> &surfacePoints() const { return m_surface; }
	const cv::Mat &rangeImage() const { return m_rangeImg; }
	const cv::Mat &normalImage() const { return m_normalImg; }
	int matchedCount() const { return (int)m_surface.size(); }
	int processedPixels() const { return m_processed; }
	bool wasCancelled() const { return m_bCancelled; }

	// 取消标志(由UI设置, 重建循环中检查)
	void setCancelFlag(std::atomic<bool> *p) { m_pCancel = p; }

	// 进度回调(行号, 总行数, 已匹配点数)
	void setProgressCb(std::function<void(int, int, int)> cb) { m_progressCb = cb; }

	// 重建诊断日志: 开启后将标定缓存与中心像素重建的全部中间结果写入日志文件+调试器输出
	//   logFile 为空时默认写到 工作目录\recon_debug.log
	void setDebugLog(bool on, const std::string &logFile = "");

private:
	CCalibParas *m_calib = nullptr;
	int m_imgW = 1440, m_imgH = 1080;
	int m_prjW = 1920, m_prjH = 1080;
	std::vector<float> m_pitches;    // 纵向(V)条纹周期
	std::vector<float> m_pitchesH;   // 横向(H)条纹周期
	float m_offsetFringe = 20.0f;
	float m_offsetFringeH = 20.0f;
	float m_zMin = 180.0f, m_zMax = 800.0f;

	// 深度搜索参数
	float m_normalDotThresh = 0.9999f;    // 法向量一致性阈值(dot); 越接近1越严格
	int   m_coarseSteps = 240;          // 粗搜索步数(更细的初始网格, 避免漏峰)
	int   m_fineSteps = 100;            // 精搜索步数(之后再做抛物线亚像素细化)

	// 边界/离群抑制(修复边缘翘起、飞点)
	int   m_maskErodePx = 2;            // 掩膜腐蚀像素(去掉调制不完整的边界)
	int   m_depthMedRadius = 2;         // 深度中值滤波邻域半径(像素)
	float m_depthMedTol = 3.0f;         // 与邻域中值深度偏差阈值(mm), 超出剔除
	int   m_depthMedMinCnt = 6;         // 邻域有效点太少则视为孤立飞点剔除

	std::vector<SurfacePoint> m_surface;
	cv::Mat m_rangeImg;    // 深度图
	cv::Mat m_normalImg;   // 法向量可视化图

	int  m_processed = 0;              // 本次重建处理的有效像素数
	bool m_bCancelled = false;         // 是否被终止
	std::atomic<bool> *m_pCancel = nullptr;
	std::function<void(int, int, int)> m_progressCb;
	bool isCancelled() const { return m_pCancel && m_pCancel->load(); }

	// 预计算的查找表
	cv::Mat m_rayDirLeft;   // 左相机每个像素的射线方向 (3通道float)

	// 预计算左相机射线方向
	void precomputeRays();

	// ---- 热循环加速: 将标定矩阵预展开为普通数组, 避免循环内 cv::Mat 分配 ----
	double m_RrT[9];          // 右相机旋转矩阵的转置 (行主序)
	double m_Tr3[3];          // 右相机平移 (mm)
	double m_Rp[9];           // 显示屏外参旋转 (行主序)
	double m_Tp3[3];          // 显示屏外参平移 (mm)
	double m_psW = 1.0, m_psH = 1.0;      // 显示屏像素物理尺寸 mm/px
	double m_fcRx = 0, m_fcRy = 0, m_ccRx = 0, m_ccRy = 0;  // 右相机内参
	double m_kcR[5] = {0,0,0,0,0};        // 右相机畸变
	void precomputeCalibCache();          // 由 m_calib 填充上述缓存

	// 无 cv::Mat 分配的快速版本 (供热循环调用)
	cv::Vec3f displayTo3Dfast(float xS, float yS) const;
	bool projectToRightFast(const cv::Vec3f &P, float &uR, float &vR) const;

	// 诊断日志
	bool m_debugLog = false;
	std::string m_logFile;
	void dbg(const std::string &msg);
	// 对单个像素执行完整重建并把中间结果写入日志(串行, 仅用于验证正确性)
	void debugReconstructPixel(int i, int j,
		const float *phLV, const float *phLH,
		const float *phRV, const float *phRH,
		const uchar *mkL, const uchar *mkR,
		const cv::Vec3f &O_L, const cv::Vec3f &O_R);

	// 由相位计算显示屏像素坐标 (亚像素)
	// phaseV: 纵向条纹展开相位 → x坐标
	// phaseH: 横向条纹展开相位 → y坐标
	void getDisplayCoord(const float *phV, const float *phH,
		int idx, float &xS, float &yS) const;

	// 显示屏像素坐标 → 左相机坐标系3D点
	cv::Vec3f displayTo3D(float xS, float yS) const;

	// 左相机3D点 → 右相机像素坐标
	bool projectToRight(const cv::Vec3f &P, float &uR, float &vR) const;

	// 双线性插值获取右相机相位
	float samplePhase(const float *phase, int W, int H, float x, float y) const;

	// 计算反射法向量
	// P: 表面点, S: 显示屏点, O: 相机光心
	static cv::Vec3f reflectNormal(const cv::Vec3f &P, const cv::Vec3f &S, const cv::Vec3f &O);
};
