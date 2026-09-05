#pragma once

#include <atomic>
#include <functional>
#include <string>
#include <mutex>

// 三频四步多频外差法相位展开
// 输入: 12幅或24幅条纹图像
//   12幅: 仅纵向(0~11) 或 仅横向(0~11, 视为横向)
//   24幅: 纵向(0~11) + 横向(12~23)
// 输出: 左右相机各自的展开相位(纵向+横向) 和 有效掩膜
// 图像顺序约定(与 binocular_deflectometry.py 一致):
//   0~11  : 纵向条纹  [频0:0-3 频1:4-7 频2:8-11]
//   12~23 : 横向条纹  [频0:12-15 频1:16-19 频2:20-23]
// 当仅12幅时, 根据hasH参数决定是纵向还是横向

class CPhaseUnwrap
{
public:
	CPhaseUnwrap();
	~CPhaseUnwrap();

	void init(int imgWidth, int imgHeight,
		int prjWidth, int prjHeight,
		int numFreq, int numStep,
		const std::vector<float> &pitchesV,
		const std::vector<float> &pitchesH,
		float offsetFringe, float offsetFringeH,
		float maxFringeIntensity = 250.0f);

	bool calculatePhasesFromFile(const std::string &folder, int poseIndex);

	bool calculatePhases(const std::vector<cv::Mat> &leftImgs, const std::vector<cv::Mat> &rightImgs,
		bool hasV = true, bool hasH = true);

	// 单相机模式: 仅处理左相机图像
	bool calculatePhasesSingle(const std::vector<cv::Mat> &imgs,
		bool hasV = true, bool hasH = true);

	bool getWrappedPhase(const cv::Mat &img1, const cv::Mat &img2,
		const cv::Mat &img3, const cv::Mat &img4,
		cv::Mat &phase, cv::Mat &mask, int modThresh = 100,
		const std::string &label = "");

	bool unwrapPhase(std::vector<cv::Mat> &wrappedPhases,
		const std::vector<cv::Mat> &masks, cv::Mat &outMask,
		const std::vector<float> &pitches,
		const std::string &label = "");

	const std::vector<cv::Mat> &phaseLeftV()  const { return m_phaseLeftV; }
	const std::vector<cv::Mat> &phaseRightV() const { return m_phaseRightV; }
	const std::vector<cv::Mat> &phaseLeftH()  const { return m_phaseLeftH; }
	const std::vector<cv::Mat> &phaseRightH() const { return m_phaseRightH; }
	const cv::Mat &maskLeftV()  const { return m_maskLeftV; }
	const cv::Mat &maskRightV() const { return m_maskRightV; }
	const cv::Mat &maskLeftH()  const { return m_maskLeftH; }
	const cv::Mat &maskRightH() const { return m_maskRightH; }

	bool hasVertical()   const { return m_hasV; }
	bool hasHorizontal() const { return m_hasH; }

	// 取消标志(由UI设置, 计算循环中检查)
	void setCancelFlag(std::atomic<bool> *p) { m_pCancel = p; }
	bool wasCancelled() const { return m_bCancelled; }

	// 进度回调(0~100)
	void setProgressCb(std::function<void(int)> cb) { m_progressCb = cb; }

	// 解相诊断日志: 开启后将解相中间信息写入日志文件 + 调试器输出(OutputDebugString)
	//   logFile 为空时默认写到 工作目录\phase_unwrap_debug.log
	void setDebugLog(bool on, const std::string &logFile = "");

private:
	int m_width = 1440, m_height = 1080;
	int m_prjWidth = 1920, m_prjHeight = 1080;
	int m_numFreq = 3;
	int m_numStep = 4;
	float m_offsetFringe = 20.0f;
	float m_offsetFringeH = 20.0f;
	float m_maxFringeIntensity = 250.0f;
	std::vector<float> m_pitches;    // 纵向条纹周期(V)
	std::vector<float> m_pitchesH;   // 横向条纹周期(H)

	std::vector<cv::Mat> m_phaseLeftV, m_phaseRightV;
	std::vector<cv::Mat> m_phaseLeftH, m_phaseRightH;
	cv::Mat m_maskLeftV, m_maskRightV;
	cv::Mat m_maskLeftH, m_maskRightH;

	bool m_hasV = true;
	bool m_hasH = true;

	std::atomic<bool> *m_pCancel = nullptr;
	bool m_bCancelled = false;
	std::function<void(int)> m_progressCb;
	bool isCancelled() { return m_pCancel && m_pCancel->load(); }

	// 诊断日志
	bool m_debugLog = false;
	std::string m_logFile;
	std::mutex m_logMutex;
	void dbg(const std::string &msg);   // 线程安全: 写文件 + OutputDebugString

	inline float heterodyne(float phs1, float phs2)
	{
		float p = phs1 - phs2;
		if (p < 0) p += (float)(2.0 * CV_PI);
		return p;
	}

	bool loadImages(const std::string &folder, int poseIndex,
		const std::string &side, std::vector<cv::Mat> &imgs);

	bool processOneCamera(const std::vector<cv::Mat> &imgs,
		std::vector<cv::Mat> &phaseV, std::vector<cv::Mat> &phaseH,
		cv::Mat &maskV, cv::Mat &maskH,
		bool hasV, bool hasH, const std::string &side = "");
};