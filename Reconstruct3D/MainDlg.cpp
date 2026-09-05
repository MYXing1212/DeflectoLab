#include "stdafx.h"
#include "Reconstruct3D.h"
#include "MainDlg.h"
#include <afxdialogex.h>
#include <fstream>
#include <sstream>
#include <future>

IMPLEMENT_DYNAMIC(CMainDlg, CDialogEx)

// Blender 渲染脚本 (binocular_deflectometry.py) 的默认输出目录
// 启动时优先读取其中的 pattern_info.txt 实现全自动导入
static const char *kDefaultBlenderOut = "D:\\BlenderSave\\blenderTwo\\save2";

// 重建中间数据输出子目录名 (位于条纹图原图所在路径下, 子目录避免被图像扫描误加载)
static const char *kMiddleDataDir = "recon_data";

static std::string trimStr(const std::string &s)
{
	size_t b = s.find_first_not_of(" \t\r\n");
	if (b == std::string::npos) return "";
	size_t e = s.find_last_not_of(" \t\r\n");
	return s.substr(b, e - b + 1);
}

// UTF-8 → CString (源码窄字面量经/utf-8编译为UTF-8字节, 必须用CP_UTF8转换;
// CString直接从char*构造会按系统ACP(GBK)转换导致乱码)
static CString Utf8ToCString(const std::string &utf8)
{
	if (utf8.empty()) return _T("");
	int n = MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), (int)utf8.size(), nullptr, 0);
	CStringW w;
	if (n > 0)
	{
		MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), (int)utf8.size(), w.GetBuffer(n), n);
		w.ReleaseBuffer(n);
	}
	return w;
}

// 相位浮点图 → 8bit 可视化 BMP (按掩膜内有效值归一化, 无效像素置0)
static void savePhaseMapBmp(const cv::Mat &phase, const cv::Mat &mask, const std::string &filename)
{
	cv::Mat vis(phase.rows, phase.cols, CV_8UC1);
	double vmin = 1e9, vmax = -1e9;
	for (int i = 0; i < phase.rows; i++)
	{
		const float *pr = phase.ptr<float>(i);
		const uchar *mr = mask.ptr<uchar>(i);
		uchar *vr = vis.ptr<uchar>(i);
		for (int j = 0; j < phase.cols; j++)
		{
			if (mr[j] && pr[j] > -1000)
			{
				if (pr[j] < vmin) vmin = pr[j];
				if (pr[j] > vmax) vmax = pr[j];
			}
		}
	}
	float range = (float)(vmax - vmin);
	if (range < 1e-6f) range = 1.0f;
	for (int i = 0; i < phase.rows; i++)
	{
		const float *pr = phase.ptr<float>(i);
		const uchar *mr = mask.ptr<uchar>(i);
		uchar *vr = vis.ptr<uchar>(i);
		for (int j = 0; j < phase.cols; j++)
		{
			if (mr[j] && pr[j] > -1000)
				vr[j] = (uchar)((pr[j] - (float)vmin) / range * 255.0f);
			else
				vr[j] = 0;
		}
	}
	cv::imwrite(filename, vis);
}

// 相位浮点图 → 原始二进制 (头: width/height/nChannels 各4字节int, 后接逐行float)
static void savePhaseRawBin(const cv::Mat &phase, const std::string &filename)
{
	FILE *saveF = fopen(filename.c_str(), "wb");
	if (!saveF) return;
	int width = phase.cols;
	int height = phase.rows;
	int nChannels = 1;
	fwrite(&width, sizeof(int), 1, saveF);
	fwrite(&height, sizeof(int), 1, saveF);
	fwrite(&nChannels, sizeof(int), 1, saveF);
	for (int i = 0; i < height; i++)
	{
		const float *pr = phase.ptr<float>(i);
		for (int j = 0; j < width; j++)
			fwrite(&pr[j], sizeof(float), 1, saveF);
	}
	fclose(saveF);
}

CMainDlg::CMainDlg(CWnd *pParent /*= nullptr*/)
	: CDialogEx(IDD_RECONSTRUCT3D_DIALOG, pParent)
{
}

CMainDlg::~CMainDlg()
{
}

void CMainDlg::DoDataExchange(CDataExchange *pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Text(pDX, IDC_EDIT_IMAGE_PATH, m_imagePath);
	DDX_Text(pDX, IDC_EDIT_CALIB_PATH, m_calibPath);
	DDX_Text(pDX, IDC_EDIT_FRINGE_PITCHES, m_fringePitches);
	DDX_Text(pDX, IDC_EDIT_ZMIN, m_zMin);
	DDX_Text(pDX, IDC_EDIT_ZMAX, m_zMax);
	DDX_Text(pDX, IDC_STATIC_STATUS, m_status);
	DDX_Control(pDX, IDC_PROGRESS_RECON, m_progress);
	DDX_Control(pDX, IDC_EDIT_LOG, m_editLog);
}

BEGIN_MESSAGE_MAP(CMainDlg, CDialogEx)
	ON_BN_CLICKED(IDC_BTN_LOAD_IMAGES, &CMainDlg::OnBnClickedLoadImages)
	ON_BN_CLICKED(IDC_BTN_LOAD_CALIB, &CMainDlg::OnBnClickedLoadCalib)
	ON_BN_CLICKED(IDC_BTN_PHASE_UNWRAP, &CMainDlg::OnBnClickedPhaseUnwrap)
	ON_BN_CLICKED(IDC_BTN_RECONSTRUCT, &CMainDlg::OnBnClickedReconstruct)
	ON_BN_CLICKED(IDC_BTN_SAVE_POINTCLOUD, &CMainDlg::OnBnClickedSavePointcloud)
	ON_BN_CLICKED(IDC_BTN_CANCEL_RECON, &CMainDlg::OnBnClickedCancelRecon)
	ON_MESSAGE(WM_APP_LOG, &CMainDlg::OnLogMsg)
	ON_MESSAGE(WM_APP_PROGRESS, &CMainDlg::OnProgressMsg)
	ON_MESSAGE(WM_APP_DONE, &CMainDlg::OnReconDoneMsg)
	ON_WM_HSCROLL()
	ON_BN_CLICKED(IDC_CHK_DEPTHCOLOR, &CMainDlg::OnBnClickedDepthColor)
	ON_BN_CLICKED(IDC_CHK_LIGHTING, &CMainDlg::OnBnClickedLighting)
	ON_BN_CLICKED(IDC_BTN_PTCOLOR, &CMainDlg::OnBnClickedPickColor)
	ON_BN_CLICKED(IDC_BTN_RESETVIEW, &CMainDlg::OnBnClickedResetView)
	ON_WM_DESTROY()
END_MESSAGE_MAP()

BOOL CMainDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();
	SetIcon(AfxGetApp()->LoadStandardIcon(IDI_APPLICATION), TRUE);
	SetIcon(AfxGetApp()->LoadStandardIcon(IDI_APPLICATION), FALSE);

	CRect rc;
	GetDlgItem(IDC_STATIC_POINTCLOUD)->GetWindowRect(&rc);
	ScreenToClient(&rc);
	// 隐藏占位静态控件
	GetDlgItem(IDC_STATIC_POINTCLOUD)->ShowWindow(SW_HIDE);

	// 顶部留出一条控制栏(在对话框灰底上, 控件外观正常), 其余区域给GL视图
	const int kBarH = 30;
	CRect barRect = rc; barRect.bottom = rc.top + kBarH;
	CRect glRect = rc;  glRect.top = rc.top + kBarH + 2;

	BOOL vcreated = m_view.Create(glRect, this, IDC_POINTCLOUD_VIEW);
	m_view.SetWindowPos(&CWnd::wndTop, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE);
	m_view.ShowWindow(SW_SHOW);
	{
		CString m; m.Format(_T("[点云视图] 创建%s, 区域 %dx%d"),
			vcreated ? _T("成功") : _T("失败"), glRect.Width(), glRect.Height());
		appendLog(m);
	}

	// 点云显示样式控制栏(整齐排布在对话框灰底上)
	{
		int x = barRect.left + 6;
		int y = barRect.top + 2;
		int hh = 24;
		auto place = [&](int w) { CRect r(x, y, x + w, y + hh); x += w + 8; return r; };

		m_lblPtSize.Create(_T("点大小"), WS_CHILD | WS_VISIBLE | SS_CENTERIMAGE | SS_LEFT, place(52), this);
		m_sliderPtSize.Create(WS_CHILD | WS_VISIBLE | TBS_HORZ | TBS_AUTOTICKS, place(120), this, IDC_SLIDER_PTSIZE);
		m_sliderPtSize.SetRange(1, 12, TRUE);
		m_sliderPtSize.SetPageSize(1);
		m_sliderPtSize.SetPos((int)(m_view.pointSize() + 0.5f));
		m_chkDepthColor.Create(_T("按深度着色"), WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX, place(100), this, IDC_CHK_DEPTHCOLOR);
		m_chkDepthColor.SetCheck(m_view.colorByDepth() ? BST_CHECKED : BST_UNCHECKED);
		m_chkLighting.Create(_T("光照"), WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX, place(60), this, IDC_CHK_LIGHTING);
		m_chkLighting.SetCheck(m_view.lighting() ? BST_CHECKED : BST_UNCHECKED);
		m_btnPtColor.Create(_T("点云颜色"), WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON, place(88), this, IDC_BTN_PTCOLOR);
		m_btnResetView.Create(_T("重置视角"), WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON, place(88), this, IDC_BTN_RESETVIEW);

		// 统一字体, 外观整洁
		CFont *pf = GetFont();
		m_lblPtSize.SetFont(pf); m_sliderPtSize.SetFont(pf);
		m_chkDepthColor.SetFont(pf); m_chkLighting.SetFont(pf);
		m_btnPtColor.SetFont(pf); m_btnResetView.SetFont(pf);
	}

	initSystemParas();
	appendLog(_T("程序启动, 等待加载条纹图像与标定参数"));

	bool hasCfg = loadConfig();
	if (hasCfg)
		appendLog(_T("[配置记忆] 已加载上次保存的参数 (Reconstruct3D.ini)"));

	// 优先用上次保存的路径自动加载; 失败再回退 Blender 自动导入
	bool savedApplied = false;
	if (hasCfg)
	{
		tryAutoLoadSavedConfig();
		savedApplied = m_bImagesLoaded || m_bCalibLoaded;
	}
	if (m_bImagesLoaded && m_bCalibLoaded)
	{
		setStatus(_T("已自动加载上次配置(条纹图像+标定参数), 可直接点击\"开始三维重建\""));
		CString dir;
		dir.Format(_T("图像目录: %s"), (LPCTSTR)m_imagePath);
		appendLog(dir);
	}
	else if (tryAutoLoadBlenderOutput())
	{
		setStatus(_T("已自动加载Blender输出(条纹图像+标定参数), 可直接点击\"开始三维重建\""));
		CString dir;
		dir.Format(_T("图像目录: %s"), CString(m_blenderImageDir.c_str()));
		appendLog(dir);
	}
	else
	{
		if (savedApplied)
		{
			setStatus(_T("上次配置已部分恢复, 请手动补全剩余项"));
			appendLog(_T("[配置记忆] 仅部分内容可恢复, 请手动加载剩余项"));
		}
		else
		{
			setStatus(_T("未找到Blender自动输出, 请手动加载条纹图像与标定参数"));
			appendLog(_T("未找到Blender自动输出(pattern_info.txt), 请手动加载"));
		}
	}
	return TRUE;
}

// 系统参数默认值 (pattern_info.txt 存在时会被自动更新)
//   相机分辨率: 1440x1080, 显示屏: 1280x800, 三频四步
//   条纹周期/偏移: 与 binocular_deflectometry.py 的 data/ 图案一致
void CMainDlg::initSystemParas()
{
	std::vector<float> defV = { 13.0f, 14.0f, 15.0f };   // 纵向(沿屏宽): P123≈1365
	std::vector<float> defH = { 20.0f, 22.0f, 24.0f };   // 横向(沿屏高): P123≈1320, 较宽更稳
	m_phase.init(m_imgW, m_imgH, m_prjW, m_prjH, 3, 4,
		defV, defH, m_offsetFringe, m_offsetFringeH, 250.0f);
}

void CMainDlg::setStatus(const CString &s)
{
	m_status = s;
	UpdateData(FALSE);
}

void CMainDlg::setProgress(int percent)
{
	m_progress.SetPos(percent);
}

// 追加一行日志(仅UI线程调用)
void CMainDlg::appendLog(const CString &s)
{
	CString line;
	line.Format(_T("[%s] %s\r\n"),
		(LPCTSTR)CTime::GetCurrentTime().Format(_T("%H:%M:%S")), (LPCTSTR)s);
	int len = m_editLog.GetWindowTextLength();
	m_editLog.SetSel(len, len);
	m_editLog.ReplaceSel(line);
	// 日志过长时截断旧内容
	int lines = m_editLog.GetLineCount();
	if (lines > 2000)
	{
		int cutEnd = m_editLog.LineIndex(1000);
		m_editLog.SetSel(0, cutEnd);
		m_editLog.ReplaceSel(_T("... (旧日志已截断) ...\r\n"));
	}
}

// 线程安全日志: 复制字符串后PostMessage, 由UI线程的OnLogMsg显示
void CMainDlg::logThreadSafe(const std::string &utf8)
{
	int n = MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), (int)utf8.size(), nullptr, 0);
	if (n <= 0) return;
	wchar_t *buf = new wchar_t[n + 1];
	MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), (int)utf8.size(), buf, n);
	buf[n] = 0;
	PostMessage(WM_APP_LOG, 0, (LPARAM)buf);
}

LRESULT CMainDlg::OnLogMsg(WPARAM, LPARAM lParam)
{
	wchar_t *p = (wchar_t*)lParam;
	if (p)
	{
		appendLog(p);
		delete[] p;
	}
	return 0;
}

LRESULT CMainDlg::OnProgressMsg(WPARAM wParam, LPARAM)
{
	m_progress.SetPos((int)wParam);
	return 0;
}

// 解析条纹周期编辑框 "13,14,15"
void CMainDlg::parsePitches(std::vector<float> &pitchesV, std::vector<float> &pitchesH)
{
	pitchesV.clear();
	pitchesH.clear();
	std::string s = CW2A(m_fringePitches.GetString());
	// 以 ';' 或 '/' 分隔 V 与 H
	std::string sv = s, sh;
	size_t sep = s.find_first_of(";/");
	if (sep != std::string::npos) { sv = s.substr(0, sep); sh = s.substr(sep + 1); }

	auto parseList = [](const std::string &str, std::vector<float> &out) {
		const char *p = str.c_str();
		char *end;
		while (*p)
		{
			float v = std::strtof(p, &end);
			if (end == p) break;
			out.push_back(v);
			p = end;
			while (*p == ',' || *p == ' ' || *p == '\t') p++;
		}
	};
	parseList(sv, pitchesV);
	parseList(sh, pitchesH);
	if (pitchesH.size() < 3) pitchesH = pitchesV;   // 未指定H则与V相同
}

// 重建期间切换按钮可用状态
void CMainDlg::setReconUiRunning(bool running)
{
	m_bReconRunning = running;
	GetDlgItem(IDC_BTN_RECONSTRUCT)->EnableWindow(!running);
	GetDlgItem(IDC_BTN_CANCEL_RECON)->EnableWindow(running);
	GetDlgItem(IDC_BTN_LOAD_IMAGES)->EnableWindow(!running);
	GetDlgItem(IDC_BTN_LOAD_CALIB)->EnableWindow(!running);
	GetDlgItem(IDC_BTN_PHASE_UNWRAP)->EnableWindow(!running);
}

CString CMainDlg::selectFolder(LPCTSTR title)
{
	CFolderPickerDialog dlg(nullptr, 0, this);
	dlg.m_ofn.lpstrTitle = title;
	if (dlg.DoModal() == IDOK)
		return dlg.GetPathName();
	return _T("");
}

// 加载双目条纹图像 (镜面物体反射显示屏的条纹)
// 自动扫描目录, 支持多种文件名格式:
//   pose0_left_img0.bmp / pose0_right_img0.bmp  (Blender脚本输出)
//   L1.bmp / R1.bmp                              (简化命名)
//   left_0.bmp / right_0.bmp                     (其他格式)
//   无前缀时所有图像归入左相机
void CMainDlg::OnBnClickedLoadImages()
{
	CString folder = selectFolder(_T("选择包含双目Deflectometry条纹图像的文件夹"));
	if (folder.IsEmpty()) return;
	m_imagePath = folder;
	UpdateData(FALSE);

	std::string path = CW2A(folder.GetString());
	appendLog(_T("[图像加载] 开始扫描目录: ") + folder);
	if (scanAndLoadImages(path))
	{
		m_bImagesLoaded = true;
		appendLog(_T("[图像加载] 完成"));
		applyPatternInfoPitches(path);   // 按图像目录的 pattern_info.txt 同步纵/横条纹周期
		saveConfig();
	}
	else
	{
		m_bImagesLoaded = false;
		m_hasV = m_hasH = false;
		appendLog(_T("[图像加载] 失败, 有效图像不足12幅"));
	}
}

bool CMainDlg::scanAndLoadImages(const std::string &folder)
{
	m_imgsLeft.clear();
	m_imgsRight.clear();

	std::vector<std::string> filesLeft, filesRight, filesAll;
	std::string searchPath = folder + "\\*.*";
	WIN32_FIND_DATAA fd;
	HANDLE hFind = FindFirstFileA(searchPath.c_str(), &fd);
	if (hFind == INVALID_HANDLE_VALUE)
	{
		setStatus(_T("无法访问目录"));
		return false;
	}
	do
	{
		if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
		std::string name(fd.cFileName);
		std::string ext = name.size() >= 4 ? name.substr(name.size() - 4) : "";
		for (auto &c : ext) c = (char)tolower(c);
		if (ext != ".bmp" && ext != ".png" && ext != ".jpg" && ext != ".tif") continue;

		std::string lower = name;
		for (auto &c : lower) c = (char)tolower(c);

		bool isLeft = false, isRight = false;
		if (lower.find("left") != std::string::npos || lower.find("_l_") != std::string::npos)
			isLeft = true;
		else if (lower.find("right") != std::string::npos || lower.find("_r_") != std::string::npos)
			isRight = true;
		else if (lower.size() >= 2 && (lower[0] == 'l' || lower[0] == 'L') && lower[1] >= '0' && lower[1] <= '9')
			isLeft = true;
		else if (lower.size() >= 2 && (lower[0] == 'r' || lower[0] == 'R') && lower[1] >= '0' && lower[1] <= '9')
			isRight = true;

		std::string fullPath = folder + "\\" + name;
		if (isLeft)
			filesLeft.push_back(fullPath);
		else if (isRight)
			filesRight.push_back(fullPath);
		else
			filesAll.push_back(fullPath);
	} while (FindNextFileA(hFind, &fd));
	FindClose(hFind);

	auto extractNum = [](const std::string &s) -> int {
		for (int i = (int)s.size() - 1; i >= 0; i--)
		{
			if (s[i] >= '0' && s[i] <= '9')
			{
				int start = i;
				while (start > 0 && s[start-1] >= '0' && s[start-1] <= '9') start--;
				return atoi(s.substr(start, i - start + 1).c_str());
			}
		}
		return 0;
	};
	auto sortByNumber = [&extractNum](std::vector<std::string> &v) {
		std::sort(v.begin(), v.end(), [&extractNum](const std::string &a, const std::string &b) {
			return extractNum(a) < extractNum(b);
		});
	};
	sortByNumber(filesLeft);
	sortByNumber(filesRight);
	sortByNumber(filesAll);

	auto loadImages = [](const std::vector<std::string> &paths, std::vector<cv::Mat> &imgs) {
		imgs.clear();
		// Blender渲染导出的BMP已是线性8bit灰度(手动从Render Result线性像素量化, 无sRGB编码)
		for (auto &p : paths)
		{
			cv::Mat img = cv::imread(p, cv::IMREAD_GRAYSCALE);
			if (!img.empty())
			{
				imgs.push_back(img);
			}
		}
	};

	if (!filesLeft.empty() || !filesRight.empty())
	{
		loadImages(filesLeft, m_imgsLeft);
		loadImages(filesRight, m_imgsRight);
	}
	else
	{
		loadImages(filesAll, m_imgsLeft);
	}

	int nL = (int)m_imgsLeft.size();
	int nR = (int)m_imgsRight.size();

	if (nL < 12 && nR < 12)
	{
		CString msg;
		msg.Format(_T("加载失败: 左相机%d幅 右相机%d幅, 至少需要12幅"), nL, nR);
		setStatus(msg);
		return false;
	}

	if (nL >= 24)
	{
		m_hasV = true; m_hasH = true;
		m_imgsLeft.resize(24);
		if (nR >= 24) { m_imgsRight.resize(24); }
		else if (nR >= 12) { m_hasH = false; m_imgsRight.resize(12); }
		CString msg;
		msg.Format(_T("已加载 左%d幅 右%d幅 (纵向+横向)"), std::min(nL,24), std::min(nR,24));
		setStatus(msg);
	}
	else if (nL >= 12)
	{
		m_hasV = true; m_hasH = false;
		m_imgsLeft.resize(12);
		if (nR >= 12) m_imgsRight.resize(12);
		CString msg;
		msg.Format(_T("已加载 左%d幅 右%d幅 (仅纵向)"), std::min(nL,12), std::min(nR,12));
		setStatus(msg);
	}
	else
	{
		m_hasV = true; m_hasH = false;
		m_imgsLeft.clear();
		loadImages(filesRight, m_imgsLeft);
		m_imgsLeft.resize(std::min((int)m_imgsLeft.size(), 12));
		CString msg;
		msg.Format(_T("已加载 右相机%d幅 (仅纵向, 作为左相机处理)"), (int)m_imgsLeft.size());
		setStatus(msg);
	}
	return true;
}


// 解析 Blender 输出的 pattern_info.txt (key = value 逐行)
//   imageDir/calibDir/camW/camH/prjW/prjH/numFreq/numStep/pitches/offsetFringe/offsetFringeH
bool CMainDlg::readPatternInfo(const std::string &file)
{
	std::ifstream fs(file);
	if (!fs.is_open()) return false;

	auto parseFloatList = [](const std::string &val, std::vector<float> &out) {
		out.clear();
		const char *p = val.c_str();
		char *end;
		while (*p)
		{
			float v = std::strtof(p, &end);
			if (end == p) break;
			out.push_back(v);
			p = end;
			while (*p == ',' || *p == ' ' || *p == '\t') p++;
		}
	};

	std::string line;
	std::vector<float> pitches, pitchesH;
	while (std::getline(fs, line))
	{
		size_t eq = line.find('=');
		if (eq == std::string::npos) continue;
		std::string key = trimStr(line.substr(0, eq));
		std::string val = trimStr(line.substr(eq + 1));
		if (key == "imageDir") m_blenderImageDir = val;
		else if (key == "calibDir") m_blenderCalibDir = val;
		else if (key == "camW") m_imgW = atoi(val.c_str());
		else if (key == "camH") m_imgH = atoi(val.c_str());
		else if (key == "prjW") m_prjW = atoi(val.c_str());
		else if (key == "prjH") m_prjH = atoi(val.c_str());
		else if (key == "pitches" || key == "pitchesV") parseFloatList(val, pitches);
		else if (key == "pitchesH") parseFloatList(val, pitchesH);
		else if (key == "offsetFringe") m_offsetFringe = (float)atof(val.c_str());
		else if (key == "offsetFringeH") m_offsetFringeH = (float)atof(val.c_str());
	}
	if (pitches.size() >= 3) m_blenderPitches = pitches;
	if (pitchesH.size() >= 3) m_blenderPitchesH = pitchesH;
	else m_blenderPitchesH = m_blenderPitches;   // 未提供H则同V
	return !m_blenderImageDir.empty();
}

// 启动时自动加载 Blender 输出:
//   1. 依次在 Blender默认输出目录 / exe工作目录 查找 pattern_info.txt
//   2. 找到 → 按配置加载双目条纹图像 + 标定参数 + 条纹周期等
//   3. 未找到 → 回退默认目录约定 (save2\pose_0 + cal_res.txt + 默认周期13,14,15)
bool CMainDlg::tryAutoLoadBlenderOutput()
{
	std::string infoFile;
	std::vector<std::string> candidates = {
		std::string(kDefaultBlenderOut) + "\\pattern_info.txt",
		"pattern_info.txt"
	};
	for (auto &c : candidates)
	{
		std::ifstream f(c);
		if (f.good()) { infoFile = c; break; }
	}

	std::string imageDir, calibDir;
	std::vector<float> pitches = { 13.0f, 14.0f, 15.0f };
	std::vector<float> pitchesH = { 20.0f, 22.0f, 24.0f };
	if (!infoFile.empty() && readPatternInfo(infoFile))
	{
		imageDir = m_blenderImageDir;
		calibDir = m_blenderCalibDir;
		// 用户上次保存的条纹周期优先 (配置记忆), 仅在无保存配置时用 pattern_info.txt 的
		if (!m_blenderPitches.empty() && !m_cfgHasPitches)
		{
			pitches = m_blenderPitches;
			if (!m_blenderPitchesH.empty()) pitchesH = m_blenderPitchesH;
		}
	}
	else
	{
		imageDir = std::string(kDefaultBlenderOut) + "\\pose_0";
		calibDir = kDefaultBlenderOut;
	}

	// 仅补充加载尚未就绪的部分 (不覆盖 tryAutoLoadSavedConfig 已成功加载的内容)
	if (!m_bImagesLoaded)
	{
		DWORD attr = GetFileAttributesA(imageDir.c_str());
		if (attr != INVALID_FILE_ATTRIBUTES && (attr & FILE_ATTRIBUTE_DIRECTORY))
		{
			if (scanAndLoadImages(imageDir))
			{
				m_bImagesLoaded = true;
				m_imagePath = CString(imageDir.c_str());
			}
		}
	}

	if (!m_bCalibLoaded && !calibDir.empty())
	{
		bool ok1 = m_calib.readBinocularCalib(calibDir + "\\cal_res.txt");
		bool ok2 = m_calib.readProjectorCalib(calibDir + "\\cal_Projector.txt");
		if (ok1 && ok2)
		{
			m_bCalibLoaded = true;
			m_calibPath = CString(calibDir.c_str());
		}
	}

	CString ps;
	for (size_t i = 0; i < pitches.size(); i++)
	{
		if (i) ps += _T(",");
		ps.AppendFormat(_T("%.1f"), pitches[i]);
	}
	if (pitchesH.size() >= 3)
	{
		ps += _T(" ; ");
		for (size_t i = 0; i < pitchesH.size(); i++)
		{
			if (i) ps += _T(",");
			ps.AppendFormat(_T("%.1f"), pitchesH[i]);
		}
	}
	m_fringePitches = ps;
	UpdateData(FALSE);

	m_autoLoaded = m_bImagesLoaded && m_bCalibLoaded;
	return m_autoLoaded;
}


// ===== 配置记忆 (exe目录下 Reconstruct3D.ini) =====
// 保存: 图像路径/标定路径/条纹周期/深度范围; 启动时加载并自动恢复上次参数
std::string CMainDlg::configFilePath()
{
	char exe[MAX_PATH] = { 0 };
	GetModuleFileNameA(nullptr, exe, MAX_PATH);
	std::string p = exe;
	size_t pos = p.find_last_of("\\/");
	if (pos == std::string::npos) return "Reconstruct3D.ini";
	return p.substr(0, pos + 1) + "Reconstruct3D.ini";
}

bool CMainDlg::loadConfig()
{
	std::string ini = configFilePath();
	if (GetFileAttributesA(ini.c_str()) == INVALID_FILE_ATTRIBUTES)
		return false;

	auto getStr = [&](const char *key, const char *def) -> std::string {
		char buf[MAX_PATH * 4] = { 0 };
		GetPrivateProfileStringA("Config", key, def, buf, sizeof(buf), ini.c_str());
		return std::string(buf);
	};
	std::string img = getStr("imagePath", "");
	std::string cal = getStr("calibPath", "");
	std::string pit = getStr("fringePitches", "");
	std::string zmin = getStr("zMin", "");
	std::string zmax = getStr("zMax", "");

	if (img.empty() && cal.empty() && pit.empty())
		return false;

	if (!img.empty()) m_imagePath = CA2W(img.c_str());
	if (!cal.empty()) m_calibPath = CA2W(cal.c_str());
	if (!pit.empty())
	{
		m_fringePitches = CA2W(pit.c_str());
		m_cfgHasPitches = true;
	}
	if (!zmin.empty()) m_zMin = CA2W(zmin.c_str());
	if (!zmax.empty()) m_zMax = CA2W(zmax.c_str());
	UpdateData(FALSE);
	return true;
}

bool CMainDlg::saveConfig()
{
	UpdateData(TRUE);
	std::string ini = configFilePath();
	auto setStr = [&](const char *key, const CString &v) {
		CStringA a(v.GetString());
		WritePrivateProfileStringA("Config", key, a.GetString(), ini.c_str());
	};
	setStr("imagePath", m_imagePath);
	setStr("calibPath", m_calibPath);
	setStr("fringePitches", m_fringePitches);
	setStr("zMin", m_zMin);
	setStr("zMax", m_zMax);
	return true;
}

// 用上次保存的路径自动加载条纹图像与标定参数 (尽力而为, 任一成功即算部分恢复)
bool CMainDlg::tryAutoLoadSavedConfig()
{
	bool loadedAny = false;

	if (!m_imagePath.IsEmpty())
	{
		std::string dir = CW2A(m_imagePath.GetString());
		DWORD attr = GetFileAttributesA(dir.c_str());
		if (attr != INVALID_FILE_ATTRIBUTES && (attr & FILE_ATTRIBUTE_DIRECTORY))
		{
			if (scanAndLoadImages(dir))
			{
				m_bImagesLoaded = true;
				loadedAny = true;
				appendLog(_T("[配置记忆] 条纹图像已从上次路径自动加载"));
			}
			else
			{
				appendLog(_T("[配置记忆] 上次图像路径加载失败, 将尝试其他来源"));
			}
		}
		else
		{
			appendLog(_T("[配置记忆] 上次图像路径不存在, 将尝试其他来源"));
			m_imagePath.Empty();
			UpdateData(FALSE);
		}
	}

	if (!m_calibPath.IsEmpty())
	{
		std::string dir = CW2A(m_calibPath.GetString());
		bool ok1 = m_calib.readBinocularCalib(dir + "\\cal_res.txt");
		bool ok2 = m_calib.readProjectorCalib(dir + "\\cal_Projector.txt");
		if (ok1 && ok2)
		{
			m_bCalibLoaded = true;
			loadedAny = true;
			appendLog(_T("[配置记忆] 标定参数已从上次路径自动加载"));
			logCalibSummary();
		}
		else
		{
			appendLog(_T("[配置记忆] 上次标定路径读取失败, 将尝试其他来源"));
			m_calibPath.Empty();
			UpdateData(FALSE);
		}
	}

	// 条纹周期必须与渲染图像一致: pattern_info.txt 与图像绑定, 优先用它覆盖陈旧的配置记忆周期
	if (m_bImagesLoaded && !m_imagePath.IsEmpty())
		applyPatternInfoPitches(std::string((const char*)CW2A(m_imagePath.GetString())));

	return loadedAny;
}

// 在图像目录(或其上级)查找 pattern_info.txt, 用其中的纵/横条纹周期覆盖编辑框,
// 保证解相所用周期与实际渲染图像一致(避免旧配置记忆的周期与图像不符导致解相错误/形状变形)
void CMainDlg::applyPatternInfoPitches(const std::string &imageDir)
{
	std::vector<std::string> cands = {
		imageDir + "\\pattern_info.txt",
		imageDir + "\\..\\pattern_info.txt"
	};
	for (auto &f : cands)
	{
		std::ifstream t(f);
		if (!t.good()) continue;
		t.close();
		if (readPatternInfo(f) && m_blenderPitches.size() >= 3)
		{
			const std::vector<float> &V = m_blenderPitches;
			const std::vector<float> &H = (m_blenderPitchesH.size() >= 3) ? m_blenderPitchesH : m_blenderPitches;
			CString ps;
			for (size_t i = 0; i < V.size(); i++) { if (i) ps += _T(","); ps.AppendFormat(_T("%g"), V[i]); }
			ps += _T(" ; ");
			for (size_t i = 0; i < H.size(); i++) { if (i) ps += _T(","); ps.AppendFormat(_T("%g"), H[i]); }
			m_fringePitches = ps;
			m_cfgHasPitches = false;
			UpdateData(FALSE);
			appendLog(_T("[条纹周期] 已按 pattern_info.txt 覆盖为(与渲染图像一致): ") + ps);
			return;
		}
	}
}

// 对话框销毁时保存当前配置参数, 下次启动自动恢复
void CMainDlg::OnDestroy()
{
	saveConfig();
	CDialogEx::OnDestroy();
}


// 加载标定参数
// 需要 cal_res.txt (双目相机) 和 cal_Projector.txt (显示屏/投影仪)
// 注意: 投影仪标定参数在Deflectometry中用于建立显示屏坐标与世界坐标的映射
void CMainDlg::OnBnClickedLoadCalib()
{
	CString folder = selectFolder(_T("选择包含标定参数的文件夹 (cal_res.txt, cal_Projector.txt)"));
	if (folder.IsEmpty()) return;
	m_calibPath = folder;
	UpdateData(FALSE);

	std::string path = CW2A(folder.GetString());
	appendLog(_T("[标定加载] 目录: ") + folder);

	bool ok1 = m_calib.readBinocularCalib(path + "\\cal_res.txt");
	if (!ok1)
	{
		appendLog(_T("[标定加载] cal_res.txt 读取失败:"));
		for (auto &e : m_calib.lastReadErrors())
			appendLog(_T("    - ") + Utf8ToCString(e));
	}
	bool ok2 = m_calib.readProjectorCalib(path + "\\cal_Projector.txt");
	if (!ok2)
	{
		appendLog(_T("[标定加载] cal_Projector.txt 读取失败:"));
		for (auto &e : m_calib.lastReadErrors())
			appendLog(_T("    - ") + Utf8ToCString(e));
	}
	if (!ok1 || !ok2)
	{
		setStatus(_T("标定参数加载不完整 (需要 cal_res.txt 和 cal_Projector.txt)"));
		m_bCalibLoaded = false;
		return;
	}
	m_bCalibLoaded = true;
	setStatus(_T("双目标定+显示屏标定已就绪"));
	appendLog(_T("[标定加载] 双目标定+显示屏标定已就绪"));
	saveConfig();
	logCalibSummary();

	// 加载后立即校验, 有问题提前警告
	auto issues = m_calib.validate(m_imgW, m_imgH);
	for (auto &s : issues)
		appendLog(_T("[标定警告] ") + Utf8ToCString(s));
	if (issues.empty())
		appendLog(_T("[标定加载] 参数校验通过"));
}

// 三频四步解相: 导入条纹图像 → 计算包裹相位和展开相位 → 导出
void CMainDlg::OnBnClickedPhaseUnwrap()
{
	if (!m_bImagesLoaded) { setStatus(_T("请先加载条纹图像")); return; }

	UpdateData(TRUE);
	std::vector<float> pitchesV, pitchesH;
	parsePitches(pitchesV, pitchesH);
	if (pitchesV.size() < 3)
	{
		setStatus(_T("条纹周期参数不足, 需要3个频率 (如 13,14,15 ; 11,12,13)"));
		return;
	}

	m_phase.init(m_imgW, m_imgH, m_prjW, m_prjH, 3, 4, pitchesV, pitchesH, m_offsetFringe, m_offsetFringeH, 250.0f);

	// 开启解相诊断日志, 写入图像目录下 phase_unwrap_debug.log
	{
		std::string logPath = m_imagePath.IsEmpty()
			? std::string("phase_unwrap_debug.log")
			: (std::string((const char*)CW2A(m_imagePath.GetString())) + "\\phase_unwrap_debug.log");
		m_phase.setDebugLog(true, logPath);
		appendLog(_T("[解相诊断] 详细日志将写入: ") + m_imagePath + _T("\\phase_unwrap_debug.log"));
	}

	setStatus(_T("正在计算相位 (三频四步多频外差法)..."));
	setProgress(10);

	bool ok;
	if (m_imgsRight.empty())
		ok = m_phase.calculatePhasesSingle(m_imgsLeft, m_hasV, m_hasH);
	else
		ok = m_phase.calculatePhases(m_imgsLeft, m_imgsRight, m_hasV, m_hasH);
	setProgress(70);

	if (!ok)
	{
		setStatus(_T("相位计算失败"));
		m_bPhaseDone = false;
		return;
	}

	// 相位有效率统计
	{
		cv::Mat maskL = m_phase.maskLeftV() & m_phase.maskLeftH();
		double ratio = 100.0 * cv::countNonZero(maskL) / (double)(m_imgW * m_imgH);
		CString msg;
		msg.Format(_T("左相机有效相位像素占比: %.1f%%"), ratio);
		appendLog(_T("[相位计算] ") + msg);
	}

	CString folder = selectFolder(_T("选择相位导出文件夹"));
	if (!folder.IsEmpty())
	{
		std::string path = CW2A(folder.GetString());
		if (exportPhaseImages(path))
			setStatus(_T("相位已导出"));
		else
			setStatus(_T("相位导出失败"));
	}
	else
	{
		setStatus(_T("相位计算完成 (未导出)"));
	}

	m_bPhaseDone = true;
	setProgress(100);
}

// 导出相位图像: 包裹相位(3频x2方向x2相机=12幅) + 展开相位(3频x2方向x2相机=12幅)
// 文件命名: wrapped_V_left_f0.bmp ... unwrapped_V_left_f0.bmp ...
bool CMainDlg::exportPhaseImages(const std::string &folder)
{
	if (!CreateDirectoryA(folder.c_str(), nullptr) && GetLastError() != ERROR_ALREADY_EXISTS)
		return false;

	const char *dirs[2] = { "V", "H" };
	const char *sides[2] = { "left", "right" };

	for (int s = 0; s < 2; s++)
	{
		if (s == 1 && m_imgsRight.empty()) continue;

		const std::vector<cv::Mat> *phV, *phH;
		const cv::Mat *maskV, *maskH;
		if (s == 0) { phV = &m_phase.phaseLeftV(); phH = &m_phase.phaseLeftH(); maskV = &m_phase.maskLeftV(); maskH = &m_phase.maskLeftH(); }
		else        { phV = &m_phase.phaseRightV(); phH = &m_phase.phaseRightH(); maskV = &m_phase.maskRightV(); maskH = &m_phase.maskRightH(); }

		bool dirAvail[2] = { m_phase.hasVertical(), m_phase.hasHorizontal() };

		for (int d = 0; d < 2; d++)
		{
			if (!dirAvail[d]) continue;

			const std::vector<cv::Mat> &phList = (d == 0) ? *phV : *phH;
			const cv::Mat &msk = (d == 0) ? *maskV : *maskH;

			for (int f = 0; f < 3; f++)
			{
				char buf[512];
				sprintf_s(buf, "%s\\wrapped_%s_%s_f%d.bmp", folder.c_str(), dirs[d], sides[s], f);
				cv::Mat wrapped = cv::Mat::zeros(phList[f].rows, phList[f].cols, CV_32FC1);
				for (int i = 0; i < phList[f].rows; i++)
				{
					const float *src = phList[f].ptr<float>(i);
					float *dst = wrapped.ptr<float>(i);
					for (int j = 0; j < phList[f].cols; j++)
					{
						if (src[j] > -1000)
							dst[j] = atan2f(sinf(src[j]), cosf(src[j]));
					}
				}
				cv::Mat wrappedMask = (phList[f] > -1000);
				savePhaseMapBmp(wrapped, wrappedMask, std::string(buf));

				sprintf_s(buf, "%s\\unwrapped_%s_%s_f%d.bmp", folder.c_str(), dirs[d], sides[s], f);
				savePhaseMapBmp(phList[f], msk, std::string(buf));

				sprintf_s(buf, "%s\\unwrapped_%s_%s_f%d.bin", folder.c_str(), dirs[d], sides[s], f);
				savePhaseRawBin(phList[f], std::string(buf));
			}
		}
	}
	return true;
}

// 三维重建过程中输出中间数据到条纹图原图所在路径下 (recon_data 子目录):
//   mask_V/H_{left,right}.bmp              调制度掩膜
//   wrapped_{V,H}_{left,right}_f{0~2}.bmp  包裹相位 (可视化)
//   absolute_{V,H}_{left,right}_f{0~2}.bmp 绝对相位 (可视化)
//   absolute_..._f{0~2}.bin                绝对相位 (原始float, 供后续分析)
// 多张图像采用 std::async 并行保存, 减少磁盘写入耗时
bool CMainDlg::exportReconMiddleData(const std::string &imageDir)
{
	if (imageDir.empty()) return false;
	std::string outDir = imageDir + "\\" + kMiddleDataDir;
	if (!CreateDirectoryA(outDir.c_str(), nullptr) && GetLastError() != ERROR_ALREADY_EXISTS)
		return false;

	std::vector<std::future<void>> jobs;


	const char *dirs[2] = { "V", "H" };
	const char *sides[2] = { "left", "right" };

	for (int s = 0; s < 2; s++)
	{
		if (s == 1 && m_imgsRight.empty()) continue;

		const std::vector<cv::Mat> *phV, *phH;
		const cv::Mat *maskV, *maskH;
		if (s == 0) { phV = &m_phase.phaseLeftV(); phH = &m_phase.phaseLeftH(); maskV = &m_phase.maskLeftV(); maskH = &m_phase.maskLeftH(); }
		else        { phV = &m_phase.phaseRightV(); phH = &m_phase.phaseRightH(); maskV = &m_phase.maskRightV(); maskH = &m_phase.maskRightH(); }

		bool dirAvail[2] = { m_phase.hasVertical(), m_phase.hasHorizontal() };

		// 调制度掩膜图像 (各方向独立)
		const char *sideName = sides[s];
		jobs.push_back(std::async(std::launch::async, [maskV, &outDir, sideName] {
			cv::imwrite(outDir + "\\mask_V_" + sideName + ".bmp", *maskV);
		}));
		jobs.push_back(std::async(std::launch::async, [maskH, &outDir, sideName] {
			cv::imwrite(outDir + "\\mask_H_" + sideName + ".bmp", *maskH);
		}));

		for (int d = 0; d < 2; d++)
		{
			if (!dirAvail[d]) continue;

			const std::vector<cv::Mat> &phList = (d == 0) ? *phV : *phH;
			const cv::Mat &msk = (d == 0) ? *maskV : *maskH;

			for (int f = 0; f < 3; f++)
			{

				std::string namePrefix = dirs[d] + std::string("_") + sides[s] + "_f" + std::to_string(f);

				// 包裹相位 (可视化)
				jobs.push_back(std::async(std::launch::async, [&phList, f, &msk, &outDir, namePrefix] {
					cv::Mat wrapped = cv::Mat::zeros(phList[f].rows, phList[f].cols, CV_32FC1);
					for (int i = 0; i < phList[f].rows; i++)
					{
						const float *src = phList[f].ptr<float>(i);
						float *dst = wrapped.ptr<float>(i);
						for (int j = 0; j < phList[f].cols; j++)
						{
							if (src[j] > -1000)
								dst[j] = atan2f(sinf(src[j]), cosf(src[j]));
						}
					}
					cv::Mat wrappedMask = (phList[f] > -1000);
					savePhaseMapBmp(wrapped, wrappedMask, outDir + "\\wrapped_" + namePrefix + ".bmp");
				}));

				// 绝对相位 (可视化 + 原始float)
				jobs.push_back(std::async(std::launch::async, [&phList, f, &msk, &outDir, namePrefix] {
					savePhaseMapBmp(phList[f], msk, outDir + "\\absolute_" + namePrefix + ".bmp");
					savePhaseRawBin(phList[f], outDir + "\\absolute_" + namePrefix + ".bin");
				}));
			}
		}
	}

	for (auto &j : jobs) j.get();
	return true;
}

// 双目 Deflectometry 三维重建主流程 (后台线程执行, 不阻塞UI)
void CMainDlg::OnBnClickedReconstruct()
{
	if (m_bReconRunning) return;
	if (!m_bImagesLoaded) { setStatus(_T("请先加载条纹图像")); return; }
	if (!m_bCalibLoaded)  { setStatus(_T("请先加载标定参数")); return; }

	// 双目Deflectometry重建需要纵向(→屏x)与横向(→屏y)两组条纹, 缺一不可
	if (!m_hasV || !m_hasH)
	{
		setStatus(_T("重建需要纵向+横向条纹(每相机24幅), 当前缺少横向条纹, 无法重建"));
		appendLog(_T("[重建] 已加载图像仅含单方向条纹(纵向或横向), 双目Deflectometry重建需要V+H两组"));
		return;
	}
	if (m_imgsRight.empty())
	{
		setStatus(_T("双目重建需要右相机图像, 当前仅加载了单相机"));
		appendLog(_T("[重建] 缺少右相机图像, 无法进行双目Deflectometry重建"));
		return;
	}

	UpdateData(TRUE);
	std::vector<float> pitchesV, pitchesH;
	parsePitches(pitchesV, pitchesH);
	if (pitchesV.size() < 3)
	{
		setStatus(_T("条纹周期参数不足, 需要3个频率 (如 13,14,15 ; 11,12,13)"));
		return;
	}
	float zMin = (float)_ttof(m_zMin);
	float zMax = (float)_ttof(m_zMax);

	m_phase.init(m_imgW, m_imgH, m_prjW, m_prjH, 3, 4, pitchesV, pitchesH, m_offsetFringe, m_offsetFringeH, 250.0f);
	m_match.setParas(&m_calib, m_imgW, m_imgH, m_prjW, m_prjH, pitchesV, pitchesH, m_offsetFringe, m_offsetFringeH, zMin, zMax);

	// 开启解相诊断日志, 写入图像目录下 phase_unwrap_debug.log
	{
		std::string logPath = m_imagePath.IsEmpty()
			? std::string("phase_unwrap_debug.log")
			: (std::string((const char*)CW2A(m_imagePath.GetString())) + "\\phase_unwrap_debug.log");
		m_phase.setDebugLog(true, logPath);
		appendLog(_T("[解相诊断] 详细日志将写入: ") + m_imagePath + _T("\\phase_unwrap_debug.log"));

		std::string reconLogPath = m_imagePath.IsEmpty()
			? std::string("recon_debug.log")
			: (std::string((const char*)CW2A(m_imagePath.GetString())) + "\\recon_debug.log");
		m_match.setDebugLog(true, reconLogPath);
		appendLog(_T("[重建诊断] 中心像素中间结果将写入: ") + m_imagePath + _T("\\recon_debug.log"));
	}

	// 立即记忆当前参数 (条纹周期/深度范围等), 下次启动自动恢复
	saveConfig();

	m_bCancel = false;
	m_phase.setCancelFlag(&m_bCancel);
	m_match.setCancelFlag(&m_bCancel);
	setReconUiRunning(true);
	setStatus(_T("重建进行中... (可点击Stop终止)"));
	setProgress(0);
	appendLog(_T("启动后台重建线程"));
	AfxBeginThread(reconThreadProc, this);
}

// 终止重建
void CMainDlg::OnBnClickedCancelRecon()
{
	if (!m_bReconRunning) return;
	m_bCancel = true;
	setStatus(_T("正在终止重建, 请稍候..."));
	appendLog(_T("用户请求终止重建, 等待计算循环退出..."));
}

// 标定参数摘要 → Log
void CMainDlg::logCalibSummary()
{
	char buf[512];
	sprintf_s(buf, "  左相机 fc=(%.2f, %.2f) cc=(%.2f, %.2f)",
		m_calib.fcLeft().x, m_calib.fcLeft().y, m_calib.ccLeft().x, m_calib.ccLeft().y);
	logThreadSafe(buf);
	sprintf_s(buf, "  右相机 fc=(%.2f, %.2f) cc=(%.2f, %.2f)",
		m_calib.fcRight().x, m_calib.fcRight().y, m_calib.ccRight().x, m_calib.ccRight().y);
	logThreadSafe(buf);
	sprintf_s(buf, "  显示屏: 像素尺寸=(%.4f, %.4f)mm/px, 分辨率=%dx%d",
		m_calib.pixelSizeW(), m_calib.pixelSizeH(), m_calib.prjResW(), m_calib.prjResH());
	logThreadSafe(buf);
	const cv::Mat &tR = m_calib.tvecRight();
	const cv::Mat &tP = m_calib.tvecPrj();
	sprintf_s(buf, "  双目外参: 基线=%.2fmm, T_R=(%.2f, %.2f, %.2f)mm",
		m_calib.baseline(), tR.at<double>(0), tR.at<double>(1), tR.at<double>(2));
	logThreadSafe(buf);
	sprintf_s(buf, "  显示屏外参: T_P模长=%.2fmm, T_P=(%.2f, %.2f, %.2f)mm",
		cv::norm(tP), tP.at<double>(0), tP.at<double>(1), tP.at<double>(2));
	logThreadSafe(buf);
}

// 重建失败时的详细错误分析报告
std::string CMainDlg::buildFailureReport()
{
	std::ostringstream oss;
	oss << "===== 重建失败, 详细错误分析 =====";

	// A. 标定参数逐项校验
	oss << "\n[A. 标定参数检查]";
	auto issues = m_calib.validate(m_imgW, m_imgH);
	if (issues.empty())
	{
		oss << "\n  数值校验通过 (未发现明显异常参数)";
	}
	else
	{
		for (auto &s : issues)
			oss << "\n  - " << s;
	}

	// B. 相位质量
	char buf[512];
	oss << "\n[B. 相位质量检查]";
	{
		cv::Mat maskL = m_phase.maskLeftV() & m_phase.maskLeftH();
		int total = m_imgW * m_imgH;
		int nL = cv::countNonZero(maskL);
		sprintf_s(buf, "\n  左相机有效相位像素: %d/%d (%.1f%%)", nL, total, 100.0 * nL / total);
		oss << buf;
		if (nL * 100.0 / total < 5.0)
			oss << "\n  [问题] 左相机有效相位比例过低: 检查条纹图是否正确/曝光调制度是否足够";
		if (!m_imgsRight.empty())
		{
			cv::Mat maskR = m_phase.maskRightV() & m_phase.maskRightH();
			int nR = cv::countNonZero(maskR);
			sprintf_s(buf, "\n  右相机有效相位像素: %d/%d (%.1f%%)", nR, total, 100.0 * nR / total);
			oss << buf;
			if (nR * 100.0 / total < 5.0)
				oss << "\n  [问题] 右相机有效相位比例过低";
		}
	}

	// C. 匹配统计
	oss << "\n[C. 匹配结果统计]";
	{
		sprintf_s(buf, "\n  处理有效像素: %d, 重建出点数: %d, 深度搜索范围: [%.1f, %.1f]mm, 法向一致性阈值: %.3f",
			m_match.processedPixels(), m_match.matchedCount(),
			(float)_ttof(m_zMin), (float)_ttof(m_zMax), 0.999f);
		oss << buf;
		if (m_match.matchedCount() == 0 && m_match.processedPixels() > 0)
			oss << "\n  [问题] 所有像素均未通过法向量一致性检验, 多为外参/相位方向/深度范围错误";
	}

	// D. 排查建议
	oss << "\n[D. 排查建议]";
	oss << "\n  1. 条纹周期 pitches 是否与渲染一致(见pattern_info.txt)";
	oss << "\n  2. Z范围(zMin/zMax)是否覆盖被测表面到左相机的实际深度(mm)";
	oss << "\n  3. 外参方向: om/T 应为右相机(或显示屏)相对左相机, 平移单位mm";
	oss << "\n  4. 若上方标定检查提示[单位错误], 请用Blender重新导出标定参数";
	oss << "\n  5. 标定文件应含: cal_res.txt 33个数, cal_Projector.txt 15个数";
	return oss.str();
}

// 后台重建线程入口
UINT CMainDlg::reconThreadProc(LPVOID pParam)
{
	CMainDlg *self = (CMainDlg*)pParam;
	if (self) self->reconWorker();
	return 0;
}

// 后台重建线程主体 (仅访问重建期间不变的数据; 结果通过WM_APP_DONE交回UI线程)
void CMainDlg::reconWorker()
{
	logThreadSafe("========== 开始三维重建 ==========");

	// 1. 标定参数摘要 + 校验
	logThreadSafe("[1/3] 标定参数检查");
	logCalibSummary();
	auto issues = m_calib.validate(m_imgW, m_imgH);
	if (!issues.empty())
	{
		logThreadSafe("  [警告] 标定参数校验发现以下问题:");
		for (auto &s : issues)
			logThreadSafe("    - " + s);
		logThreadSafe("  继续尝试重建, 若失败请先修正上述参数");
	}
	else
	{
		logThreadSafe("  标定参数校验通过");
	}

	// 2. 相位计算
	logThreadSafe("[2/3] 相位计算 (三频四步多频外差法)");
	PostMessage(WM_APP_PROGRESS, 5);
	if (!m_bPhaseDone)
	{
		m_phase.setProgressCb([this](int p) { PostMessage(WM_APP_PROGRESS, 5 + p / 5); });
		bool okPhase;
		if (m_imgsRight.empty())
		{
			logThreadSafe("  单相机模式 (仅左相机图像)");
			okPhase = m_phase.calculatePhasesSingle(m_imgsLeft, m_hasV, m_hasH);
		}
		else
		{
			okPhase = m_phase.calculatePhases(m_imgsLeft, m_imgsRight, m_hasV, m_hasH);
		}
		if (!okPhase)
		{
			std::string detail = m_phase.wasCancelled() ?
				std::string("===== 重建已被用户终止 (相位计算阶段) =====") :
				std::string("===== 重建失败: 相位计算失败 =====\n检查图像数量(每相机至少12幅)与图像完整性");
			PostMessage(WM_APP_DONE, m_phase.wasCancelled() ? 2 : 0, (LPARAM)new std::string(detail));
			return;
		}
	}
	else
	{
		logThreadSafe("  相位已预先计算, 跳过");
	}
	PostMessage(WM_APP_PROGRESS, 25);

	// 输出中间数据 (调制度掩膜/包裹相位/绝对相位) 到条纹图原图所在路径
	{
		std::string imgDir = m_imagePath.IsEmpty() ? std::string() : std::string(CW2A(m_imagePath.GetString()));
		if (!imgDir.empty() && exportReconMiddleData(imgDir))
		{
			logThreadSafe("  中间数据已输出到: " + imgDir + "\\" + kMiddleDataDir);
			logThreadSafe("    (调制度掩膜 mask_*.bmp, 包裹相位 wrapped_*.bmp, 绝对相位 absolute_*.bmp/bin)");
		}
		else
		{
			logThreadSafe("  [警告] 中间数据输出失败 (图像目录无效或磁盘写入失败)");
		}
	}

	// 3. 双目 Deflectometry 匹配重建
	logThreadSafe("[3/3] 双目Deflectometry重建 (深度搜索+法向量一致性)");
	m_match.setProgressCb([this](int row, int total, int matched) {
		PostMessage(WM_APP_PROGRESS, 25 + row * 75 / std::max(total, 1));
	});
	bool ok = m_match.reconstructDeflectometry(
		m_phase.phaseLeftV(), m_phase.phaseLeftH(),
		m_phase.maskLeftV(), m_phase.maskLeftH(),
		m_phase.phaseRightV(), m_phase.phaseRightH(),
		m_phase.maskRightV(), m_phase.maskRightH());

	if (m_match.wasCancelled())
	{
		PostMessage(WM_APP_DONE, 2, (LPARAM)new std::string("===== 重建已被用户终止 ====="));
		return;
	}
	if (!ok)
	{
		PostMessage(WM_APP_DONE, 0, (LPARAM)new std::string(buildFailureReport()));
		return;
	}

	char buf[256];
	sprintf_s(buf, "重建完成: 处理像素 %d, 重建点数 %d", m_match.processedPixels(), m_match.matchedCount());
	logThreadSafe(buf);
	PostMessage(WM_APP_DONE, 1, (LPARAM)new std::string());
}

// 重建完成/失败/终止 → UI线程收尾 (点云显示等OpenGL操作必须在UI线程)
LRESULT CMainDlg::OnReconDoneMsg(WPARAM wParam, LPARAM lParam)
{
	std::string *detail = (std::string*)lParam;
	int code = (int)wParam;
	setReconUiRunning(false);

	if (code == 1)
	{
		setProgress(100);
		CString msg;
		msg.Format(_T("Deflectometry重建完成, 点数: %d"), m_match.matchedCount());
		setStatus(msg);
		appendLog(msg);

		std::vector<Point3D> pts;
		pts.reserve(m_match.surfacePoints().size());
		for (auto &sp : m_match.surfacePoints())
			pts.push_back(sp.pos);
		m_view.setPointCloud(pts);
		m_view.setRangeImage(m_match.rangeImage());
		m_view.redraw();
	}
	else if (code == 2)
	{
		setStatus(_T("重建已终止"));
		if (detail && !detail->empty())
			appendLog(Utf8ToCString(*detail));
	}
	else
	{
		setStatus(_T("重建失败, 详见Log窗口错误分析"));
		if (detail && !detail->empty())
			appendLog(Utf8ToCString(*detail));
		m_bPhaseDone = false;
	}
	delete detail;
	return 0;
}

// 点大小滑块
void CMainDlg::OnHScroll(UINT nSBCode, UINT nPos, CScrollBar *pScrollBar)
{
	if (pScrollBar && pScrollBar->GetSafeHwnd() == m_sliderPtSize.GetSafeHwnd())
	{
		int pos = m_sliderPtSize.GetPos();
		m_view.setPointSize((float)pos);
	}
	CDialogEx::OnHScroll(nSBCode, nPos, pScrollBar);
}

// 切换"按深度着色"
void CMainDlg::OnBnClickedDepthColor()
{
	m_view.setColorByDepth(m_chkDepthColor.GetCheck() == BST_CHECKED);
}

// 复位3D视角
void CMainDlg::OnBnClickedResetView()
{
	m_view.resetView();
}

// 切换光照着色
void CMainDlg::OnBnClickedLighting()
{
	m_view.setLighting(m_chkLighting.GetCheck() == BST_CHECKED);
}

// 选择点云纯色
void CMainDlg::OnBnClickedPickColor()
{
	CColorDialog dlg(m_ptColor, CC_FULLOPEN, this);
	if (dlg.DoModal() != IDOK) return;
	m_ptColor = dlg.GetColor();
	m_view.setSolidColor(GetRValue(m_ptColor), GetGValue(m_ptColor), GetBValue(m_ptColor));
	// 选定纯色后自动关闭深度着色, 使颜色立即生效
	m_chkDepthColor.SetCheck(BST_UNCHECKED);
	m_view.setColorByDepth(false);
}

// 保存点云(含法向量)为PLY
void CMainDlg::OnBnClickedSavePointcloud()
{
	if (m_match.surfacePoints().empty()) { setStatus(_T("无点云可保存")); return; }
	CFileDialog dlg(FALSE, _T("ply"), _T("deflectometry.ply"), OFN_OVERWRITEPROMPT, _T("PLY Files (*.ply)|*.ply||"), this);
	if (dlg.DoModal() != IDOK) return;
	std::string path = CW2A(dlg.GetPathName().GetString());
	if (savePLY(path, m_match.surfacePoints()))
		setStatus(_T("点云已保存"));
	else
		setStatus(_T("保存失败"));
}

bool CMainDlg::savePLY(const std::string &path, const std::vector<SurfacePoint> &pts)
{
	std::ofstream fs(path);
	if (!fs.is_open()) return false;
	fs << "ply\nformat ascii 1.0\nelement vertex " << pts.size() << "\n";
	fs << "property float x\nproperty float y\nproperty float z\n";
	fs << "property float nx\nproperty float ny\nproperty float nz\n";
	fs << "end_header\n";
	for (auto &p : pts)
		fs << p.pos.x << " " << p.pos.y << " " << p.pos.z << " "
		<< p.normal.x << " " << p.normal.y << " " << p.normal.z << "\n";
	return true;
}
