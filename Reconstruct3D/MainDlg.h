#pragma once

#include <atomic>

#include "CalibParas.h"
#include "PhaseUnwrap.h"
#include "Match.h"
#include "PointCloudView.h"

// 自定义线程通信消息 (重建在后台线程执行, 通过PostMessage与UI线程通信)
#define WM_APP_LOG      (WM_APP + 1)   // lParam = wchar_t*(堆分配, 接收方释放)
#define WM_APP_PROGRESS (WM_APP + 2)   // wParam = 进度百分比0~100
#define WM_APP_DONE     (WM_APP + 3)   // wParam=1成功/0失败/2终止, lParam = std::string*(堆, UTF-8)


class CMainDlg : public CDialogEx
{
	DECLARE_DYNAMIC(CMainDlg)
public:
	CMainDlg(CWnd *pParent = nullptr);
	virtual ~CMainDlg();

	enum { IDD = IDD_RECONSTRUCT3D_DIALOG };

protected:
	virtual void DoDataExchange(CDataExchange *pDX);
	virtual BOOL OnInitDialog();
	afx_msg void OnDestroy();

	DECLARE_MESSAGE_MAP()

public:
	afx_msg void OnBnClickedLoadImages();
	afx_msg void OnBnClickedLoadCalib();
	afx_msg void OnBnClickedPhaseUnwrap();
	afx_msg void OnBnClickedReconstruct();
	afx_msg void OnBnClickedSavePointcloud();
	afx_msg void OnBnClickedCancelRecon();

	// 点云显示样式控制
	afx_msg void OnHScroll(UINT nSBCode, UINT nPos, CScrollBar *pScrollBar);
	afx_msg void OnBnClickedDepthColor();
	afx_msg void OnBnClickedPickColor();
	afx_msg void OnBnClickedResetView();
	afx_msg void OnBnClickedLighting();

	// 后台线程 → UI线程 消息处理
	afx_msg LRESULT OnLogMsg(WPARAM wParam, LPARAM lParam);
	afx_msg LRESULT OnProgressMsg(WPARAM wParam, LPARAM lParam);
	afx_msg LRESULT OnReconDoneMsg(WPARAM wParam, LPARAM lParam);

private:
	CCalibParas     m_calib;
	CPhaseUnwrap    m_phase;
	CMatch          m_match;
	CPointCloudView m_view;

	CString m_imagePath;
	CString m_calibPath;
	CString m_fringePitches = _T("13,14,15 ; 20,22,24");   // "纵向;横向"
	CString m_zMin = _T("180");
	CString m_zMax = _T("800");
	CString m_status = _T("就绪");
	CProgressCtrl m_progress;
	CEdit m_editLog;

	// 点云显示样式控件(在OnInitDialog中动态创建, 覆盖在3D视图左上角)
	CStatic  m_lblPtSize;
	CSliderCtrl m_sliderPtSize;
	CButton  m_chkDepthColor;
	CButton  m_chkLighting;
	CButton  m_btnPtColor;
	CButton  m_btnResetView;
	COLORREF m_ptColor = RGB(180, 200, 255);

	bool m_bImagesLoaded = false;
	bool m_bCalibLoaded = false;
	bool m_bPhaseDone = false;
	bool m_hasV = false;
	bool m_hasH = false;
	bool m_autoLoaded = false;
	std::vector<cv::Mat> m_imgsLeft, m_imgsRight;

	// 重建线程控制
	std::atomic<bool> m_bCancel{ false };
	std::atomic<bool> m_bReconRunning{ false };

	// 系统参数(由pattern_info.txt自动更新, 不再硬编码)
	int   m_imgW = 1440, m_imgH = 1080;
	int   m_prjW = 1280, m_prjH = 800;
	float m_offsetFringe = 20.0f, m_offsetFringeH = 20.0f;

	// pattern_info.txt 解析结果
	std::string m_blenderImageDir;
	std::string m_blenderCalibDir;
	std::vector<float> m_blenderPitches;    // 纵向(V)
	std::vector<float> m_blenderPitchesH;   // 横向(H)

	// 配置记忆 (exe目录下 Reconstruct3D.ini)
	bool m_cfgHasPitches = false;

	void initSystemParas();
	void setStatus(const CString &s);
	void setProgress(int percent);
	void appendLog(const CString &s);
	// 线程安全日志: 复制字符串后PostMessage到UI线程
	void logThreadSafe(const std::string &utf8);

	// 解析条纹周期编辑框: 格式 "V1,V2,V3 ; H1,H2,H3" (无分号则H=V)
	void parsePitches(std::vector<float> &pitchesV, std::vector<float> &pitchesH);
	void setReconUiRunning(bool running);
	void logCalibSummary();
	std::string buildFailureReport();

	static UINT reconThreadProc(LPVOID pParam);
	void reconWorker();

	bool exportPhaseImages(const std::string &folder);
	bool exportReconMiddleData(const std::string &imageDir);
	bool scanAndLoadImages(const std::string &folder);
	bool readPatternInfo(const std::string &file);
	bool tryAutoLoadBlenderOutput();
	bool tryAutoLoadSavedConfig();
	// 用 pattern_info.txt 中的纵/横条纹周期覆盖编辑框(周期须与渲染图像一致)
	void applyPatternInfoPitches(const std::string &imageDir);

	// 配置记忆 (保存/加载上次使用的参数与路径)
	std::string configFilePath();
	bool saveConfig();
	bool loadConfig();


	// 保存点云(含法向量)为PLY
	bool savePLY(const std::string &path, const std::vector<SurfacePoint> &pts);

	CString selectFolder(LPCTSTR title);
};
