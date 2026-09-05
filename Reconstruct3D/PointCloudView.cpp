#include "stdafx.h"
#include "PointCloudView.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <fstream>
#include <ctime>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <random>
#ifdef _OPENMP
#include <omp.h>
#endif

IMPLEMENT_DYNAMIC(CPointCloudView, CWnd)

// 内嵌着色器源码(找不到 .vert/.frag 文件时的回退, 与文件内容一致)
static const char *kEmbeddedVert =
	"#version 330 core\n"
	"layout(location=0) in vec3 aPos;\n"
	"layout(location=1) in vec3 aColor;\n"
	"layout(location=2) in vec3 aNormal;\n"
	"uniform mat4 uMVP;\n"
	"uniform mat3 uNormalMat;\n"
	"uniform float uPointSize;\n"
	"out vec3 vColor;\n"
	"out vec3 vNormalView;\n"
	"void main(){ gl_Position=uMVP*vec4(aPos,1.0); gl_PointSize=uPointSize; vColor=aColor; vNormalView=uNormalMat*aNormal; }\n";
static const char *kEmbeddedFrag =
	"#version 330 core\n"
	"in vec3 vColor;\n"
	"in vec3 vNormalView;\n"
	"out vec4 FragColor;\n"
	"uniform int uUseDepthColor;\n"
	"uniform vec3 uSolidColor;\n"
	"uniform int uLighting;\n"
	"void main(){\n"
	"  vec3 base=(uUseDepthColor==1)?vColor:uSolidColor;\n"
	"  if(uLighting==1){ vec3 N=normalize(vNormalView); if(N.z<0.0)N=-N;\n"
	"    vec3 L=normalize(vec3(0.35,0.45,1.0)); float d=max(dot(N,L),0.0);\n"
	"    float amb=0.35; base*= (amb+(1.0-amb)*d); }\n"
	"  FragColor=vec4(base,1.0); }\n";

CPointCloudView::CPointCloudView()
{
	// 默认略微俯视, 便于看出立体感
	m_rotMat = glm::rotate(glm::mat4(1.0f), glm::radians(-20.0f), glm::vec3(1, 0, 0));
}

CPointCloudView::~CPointCloudView()
{
	if (m_hRC)
	{
		wglMakeCurrent(m_hDC, m_hRC);
		if (m_vbo) glDeleteBuffers(1, &m_vbo);
		if (m_vao) glDeleteVertexArrays(1, &m_vao);
		wglMakeCurrent(nullptr, nullptr);
		wglDeleteContext(m_hRC);
	}
	if (m_hDC) ::ReleaseDC(m_hWnd, m_hDC);
}

BEGIN_MESSAGE_MAP(CPointCloudView, CWnd)
	ON_WM_CREATE()
	ON_WM_SIZE()
	ON_WM_PAINT()
	ON_WM_ERASEBKGND()
	ON_WM_LBUTTONDOWN()
	ON_WM_LBUTTONUP()
	ON_WM_MOUSEMOVE()
	ON_WM_RBUTTONDOWN()
	ON_WM_RBUTTONUP()
	ON_WM_MBUTTONDOWN()
	ON_WM_MBUTTONUP()
	ON_WM_LBUTTONDBLCLK()
	ON_WM_MOUSEWHEEL()
END_MESSAGE_MAP()

std::string CPointCloudView::exeDir()
{
	char buf[MAX_PATH] = { 0 };
	GetModuleFileNameA(nullptr, buf, MAX_PATH);
	std::string p(buf);
	size_t pos = p.find_last_of("\\/");
	return pos == std::string::npos ? std::string(".") : p.substr(0, pos);
}

void CPointCloudView::logMsg(const std::string &msg)
{
	std::string path = m_logPath.empty() ? (exeDir() + "\\pointcloud_render.log") : m_logPath;
	std::time_t t = std::time(nullptr);
	std::tm tmv; localtime_s(&tmv, &t);
	char ts[32]; std::strftime(ts, sizeof(ts), "%H:%M:%S", &tmv);
	std::string line = std::string("[PCVIEW ") + ts + "] " + msg + "\n";
	FILE *fp = nullptr;
	if (fopen_s(&fp, path.c_str(), "a") == 0 && fp) { fputs(line.c_str(), fp); fclose(fp); }
	OutputDebugStringA(line.c_str());
}

void CPointCloudView::logGLError(const std::string &stage)
{
	GLenum e = glGetError();
	if (e != GL_NO_ERROR)
	{
		std::ostringstream os; os << "GL错误 @ " << stage << " : 0x" << std::hex << e;
		logMsg(os.str());
	}
}

BOOL CPointCloudView::Create(const RECT &rect, CWnd *pParent, UINT nID)
{
	LPCTSTR className = AfxRegisterWndClass(CS_HREDRAW | CS_VREDRAW | CS_OWNDC,
		::LoadCursor(nullptr, IDC_ARROW), (HBRUSH)::GetStockObject(BLACK_BRUSH), nullptr);
	BOOL ok = CWnd::Create(className, _T(""), WS_CHILD | WS_VISIBLE | WS_CLIPSIBLINGS, rect, pParent, nID);
	logMsg(std::string("CWnd::Create 返回 ") + (ok ? "TRUE" : "FALSE") +
		" 尺寸=(" + std::to_string(rect.right - rect.left) + "x" + std::to_string(rect.bottom - rect.top) + ")");
	return ok;
}

int CPointCloudView::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
	if (CWnd::OnCreate(lpCreateStruct) == -1) { logMsg("OnCreate: CWnd::OnCreate 失败"); return -1; }
	logMsg("OnCreate: 开始初始化 OpenGL");
	m_hDC = ::GetDC(m_hWnd);
	if (!m_hDC) { logMsg("OnCreate: GetDC 失败"); return 0; }
	if (!initGL())
	{
		logMsg("OnCreate: initGL 失败, 视图将保持空白(见上方日志)");
		return 0;   // 不返回 -1, 保留窗口以便看到背景/日志
	}
	if (!initShaderAndBuffers())
	{
		logMsg("OnCreate: 着色器/缓冲初始化失败");
		return 0;
	}
	m_bGLInit = true;
	logMsg("OnCreate: OpenGL + 着色器初始化完成");
	return 0;
}

bool CPointCloudView::initGL()
{
	PIXELFORMATDESCRIPTOR pfd = { 0 };
	pfd.nSize = sizeof(pfd);
	pfd.nVersion = 1;
	pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
	pfd.iPixelType = PFD_TYPE_RGBA;
	pfd.cColorBits = 32;
	pfd.cDepthBits = 24;
	pfd.iLayerType = PFD_MAIN_PLANE;

	int fmt = ::ChoosePixelFormat(m_hDC, &pfd);
	logMsg("ChoosePixelFormat 返回 " + std::to_string(fmt));
	if (!fmt) return false;
	if (!::SetPixelFormat(m_hDC, fmt, &pfd)) { logMsg("SetPixelFormat 失败"); return false; }

	m_hRC = wglCreateContext(m_hDC);
	if (!m_hRC) { logMsg("wglCreateContext 失败"); return false; }
	if (!wglMakeCurrent(m_hDC, m_hRC)) { logMsg("wglMakeCurrent 失败"); return false; }

	glewExperimental = GL_TRUE;   // 确保核心/兼容上下文下 VAO 等函数指针正确加载
	GLenum err = glewInit();
	logMsg(std::string("glewInit 返回 ") + (err == GLEW_OK ? "GLEW_OK" : (const char*)glewGetErrorString(err)));

	const GLubyte *ver = glGetString(GL_VERSION);
	const GLubyte *ren = glGetString(GL_RENDERER);
	const GLubyte *sl = glGetString(GL_SHADING_LANGUAGE_VERSION);
	logMsg(std::string("GL_VERSION=") + (ver ? (const char*)ver : "null") +
		" | GL_RENDERER=" + (ren ? (const char*)ren : "null") +
		" | GLSL=" + (sl ? (const char*)sl : "null"));

	// 关闭垂直同步: 否则每次 SwapBuffers 阻塞到 60Hz, 交互(每次移动同步渲染)会明显滞后
	if (wglSwapIntervalEXT)
	{
		wglSwapIntervalEXT(0);
		logMsg("已关闭垂直同步(wglSwapIntervalEXT(0))");
	}

	glClearColor(0.06f, 0.06f, 0.09f, 1.0f);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_PROGRAM_POINT_SIZE);   // 允许着色器设置 gl_PointSize
	return true;
}

bool CPointCloudView::initShaderAndBuffers()
{
	// 依次尝试多个路径加载着色器文件, 失败则回退内嵌源码
	std::string dir = exeDir();
	std::string cands[] = {
		dir + "\\shaders\\pointcloud",
		dir + "\\pointcloud",
		"shaders\\pointcloud",
		"pointcloud"
	};
	std::string log;
	bool loaded = false;
	for (auto &base : cands)
	{
		std::string v = base + ".vert", f = base + ".frag";
		std::ifstream test(v);
		if (!test.is_open()) continue;
		test.close();
		if (m_shader.loadFromFiles(v, f, log)) { loaded = true; break; }
	}
	if (!loaded)
	{
		log += "未能从文件加载着色器, 使用内嵌源码\n";
		if (!m_shader.loadFromSource(kEmbeddedVert, kEmbeddedFrag, log))
		{
			logMsg("着色器加载彻底失败:\n" + log);
			return false;
		}
	}
	logMsg("着色器加载日志:\n" + log);

	glGenVertexArrays(1, &m_vao);
	glGenBuffers(1, &m_vbo);
	logMsg("VAO=" + std::to_string(m_vao) + " VBO=" + std::to_string(m_vbo));
	logGLError("genVAO/VBO");
	return m_shader.valid() && m_vao != 0 && m_vbo != 0;
}

BOOL CPointCloudView::OnEraseBkgnd(CDC *)
{
	return TRUE;   // 由 GL 清屏, 禁止默认擦除
}

void CPointCloudView::OnSize(UINT nType, int cx, int cy)
{
	CWnd::OnSize(nType, cx, cy);
	if (m_bGLInit && cx > 0 && cy > 0)
	{
		wglMakeCurrent(m_hDC, m_hRC);
		setupViewport(cx, cy);
		redraw();
	}
}

void CPointCloudView::setupViewport(int w, int h)
{
	glViewport(0, 0, w, h);
}

void CPointCloudView::uploadBuffer()
{
	m_vboData.clear();
	m_vboData.reserve(m_pts.size() * 9);
	bool hasColor = (m_colors.size() == m_pts.size());
	bool hasNormal = (m_normals.size() == m_pts.size());
	for (size_t i = 0; i < m_pts.size(); i++)
	{
		m_vboData.push_back(m_pts[i].x);
		m_vboData.push_back(m_pts[i].y);
		m_vboData.push_back(m_pts[i].z);
		if (hasColor)
		{
			const cv::Vec3b &v = m_colors[i];   // [B,G,R]
			m_vboData.push_back(v[2] / 255.0f);
			m_vboData.push_back(v[1] / 255.0f);
			m_vboData.push_back(v[0] / 255.0f);
		}
		else { m_vboData.push_back(0.7f); m_vboData.push_back(0.8f); m_vboData.push_back(1.0f); }
		if (hasNormal)
		{
			m_vboData.push_back(m_normals[i].x);
			m_vboData.push_back(m_normals[i].y);
			m_vboData.push_back(m_normals[i].z);
		}
		else { m_vboData.push_back(0.0f); m_vboData.push_back(0.0f); m_vboData.push_back(1.0f); }
	}
	m_vertCount = (int)m_pts.size();

	if (!m_bGLInit) return;
	wglMakeCurrent(m_hDC, m_hRC);
	glBindVertexArray(m_vao);
	glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
	glBufferData(GL_ARRAY_BUFFER, m_vboData.size() * sizeof(float), m_vboData.data(), GL_STATIC_DRAW);
	const int stride = 9 * sizeof(float);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void*)0);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, (void*)(6 * sizeof(float)));
	glBindVertexArray(0);
	logMsg("上传顶点缓冲: 顶点数=" + std::to_string(m_vertCount));
	logGLError("uploadBuffer");
}

void CPointCloudView::OnPaint()
{
	CPaintDC dc(this);
	if (!m_bGLInit)
	{
		if (!m_loggedFrame) { logMsg("OnPaint: GL 未初始化, 无法渲染"); m_loggedFrame = true; }
		return;
	}
	wglMakeCurrent(m_hDC, m_hRC);
	CRect rc; GetClientRect(&rc);
	setupViewport(rc.Width(), rc.Height());
	renderScene();
	SwapBuffers(m_hDC);
}

void CPointCloudView::renderScene()
{
	m_frameCnt++;
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if (m_vertCount <= 0)
	{
		if (!m_loggedFrame) { logMsg("renderScene: 点数为0, 仅清屏"); m_loggedFrame = true; }
		return;
	}

	CRect rc; GetClientRect(&rc);
	float aspect = rc.Height() > 0 ? (float)rc.Width() / rc.Height() : 1.0f;
	float r = m_radius > 1e-6f ? m_radius : 1.0f;

	// 正交投影: 缩放通过改变正交半高实现(m_scale 越大越放大)。
	float halfH = r / m_scale;
	float halfW = halfH * aspect;
	float D = 100.0f * r;   // 相机沿-Z的固定距离(正交下不影响大小, 仅用于深度排序)
	glm::mat4 proj = glm::ortho(-halfW, halfW, -halfH, halfH, 0.01f * r, 2.0f * D + 10.0f * r);

	// 视图: 绕 pivot 轨道旋转; pivot 始终映射到屏幕中心(平移改变 pivot)
	glm::mat4 view = glm::translate(glm::mat4(1.0f), glm::vec3(0, 0, -D)) * m_rotMat
		* glm::translate(glm::mat4(1.0f), glm::vec3(-m_pivot.x, -m_pivot.y, -m_pivot.z));
	glm::mat4 mvp = proj * view;

	glm::mat3 normalMat = glm::mat3(view);   // 视图旋转部分, 把法向变换到视图空间
	bool hasNormal = ((int)m_normals.size() == m_vertCount) && m_vertCount > 0;

	m_shader.use();
	glUniformMatrix4fv(m_shader.uniform("uMVP"), 1, GL_FALSE, glm::value_ptr(mvp));
	glUniformMatrix3fv(m_shader.uniform("uNormalMat"), 1, GL_FALSE, glm::value_ptr(normalMat));
	// 交互降采样时适当放大点, 让稀疏预览仍显连续
	float drawPtSize = m_pointSize;
	if (m_interacting && m_vertCount > m_drawBudget) drawPtSize = m_pointSize * 1.6f;
	glUniform1f(m_shader.uniform("uPointSize"), drawPtSize);
	glUniform1i(m_shader.uniform("uUseDepthColor"), (m_colorByDepth && (int)m_colors.size() == m_vertCount) ? 1 : 0);
	glUniform3f(m_shader.uniform("uSolidColor"), m_solidR / 255.0f, m_solidG / 255.0f, m_solidB / 255.0f);
	glUniform1i(m_shader.uniform("uLighting"), (m_lighting && hasNormal) ? 1 : 0);

	// 交互(拖拽)时只画均匀子集, 保证跟手; 松手后画全量
	GLsizei drawCount = (GLsizei)m_vertCount;
	if (m_interacting && m_vertCount > m_drawBudget) drawCount = (GLsizei)m_drawBudget;

	glBindVertexArray(m_vao);
	glDrawArrays(GL_POINTS, 0, drawCount);
	glBindVertexArray(0);

	if (!m_loggedFrame)
	{
		std::ostringstream os;
		os << "首帧渲染(正交): 顶点数=" << m_vertCount << " 视口=" << rc.Width() << "x" << rc.Height()
		   << " 半径=" << r << " 半高=" << halfH
		   << " 轴心=(" << m_pivot.x << "," << m_pivot.y << "," << m_pivot.z << ")"
		   << " pointSize=" << m_pointSize << " depthColor=" << (m_colorByDepth ? 1 : 0);
		logMsg(os.str());
		logGLError("drawArrays");
		m_loggedFrame = true;
	}
}

void CPointCloudView::setPointCloud(const std::vector<Point3D> &pts)
{
	m_pts = pts;
	computeBoundingBox();
	m_colors.clear();
	m_loggedFrame = false;
	logMsg("setPointCloud: 点数=" + std::to_string(pts.size()));
	estimateNormals();      // OpenMP 并行法向估计(用于光照着色)
	// 随机打乱点序: 使"绘制前K个点"成为空间均匀子集, 供交互降采样预览用
	if (m_pts.size() == m_normals.size() && m_pts.size() > 1)
	{
		std::mt19937 rng(12345u);
		for (int i = (int)m_pts.size() - 1; i > 0; i--)
		{
			int j = (int)(rng() % (unsigned)(i + 1));
			std::swap(m_pts[i], m_pts[j]);
			std::swap(m_normals[i], m_normals[j]);
		}
	}
	uploadBuffer();
	redraw();
}

// 基于体素网格kNN + PCA(协方差最小特征向量)的法向估计, OpenMP并行。
void CPointCloudView::estimateNormals()
{
	int n = (int)m_pts.size();
	m_normals.assign(n, Point3D{ 0, 0, 1 });
	if (n < 3) return;

	// 体素尺寸: 由包围盒与点数估计平均间距(表面≈2D, 间距~diag/sqrt(N)), 邻域取 3x3x3
	double dx = 0, dy = 0, dz = 0;
	{
		float mnx = m_pts[0].x, mny = m_pts[0].y, mnz = m_pts[0].z, mxx = mnx, mxy = mny, mxz = mnz;
		for (auto &p : m_pts)
		{
			mnx = std::min(mnx, p.x); mny = std::min(mny, p.y); mnz = std::min(mnz, p.z);
			mxx = std::max(mxx, p.x); mxy = std::max(mxy, p.y); mxz = std::max(mxz, p.z);
		}
		dx = mxx - mnx; dy = mxy - mny; dz = mxz - mnz;
	}
	double diag = std::sqrt(dx * dx + dy * dy + dz * dz);
	float h = (float)std::max(1e-6, diag / std::sqrt((double)n) * 3.0);  // 体素边长≈3倍平均间距
	float invh = 1.0f / h;
	float ox = m_pts[0].x, oy = m_pts[0].y, oz = m_pts[0].z;
	for (auto &p : m_pts) { ox = std::min(ox, p.x); oy = std::min(oy, p.y); oz = std::min(oz, p.z); }

	auto keyOf = [&](int ix, int iy, int iz) -> long long {
		return ((long long)(ix & 0x1FFFFF) << 42) | ((long long)(iy & 0x1FFFFF) << 21) | (long long)(iz & 0x1FFFFF);
	};
	// 建哈希网格(串行, 快)
	std::unordered_map<long long, std::vector<int>> grid;
	grid.reserve(n * 2);
	std::vector<int> ci(n), cj(n), ck(n);
	for (int i = 0; i < n; i++)
	{
		int ix = (int)((m_pts[i].x - ox) * invh);
		int iy = (int)((m_pts[i].y - oy) * invh);
		int iz = (int)((m_pts[i].z - oz) * invh);
		ci[i] = ix; cj[i] = iy; ck[i] = iz;
		grid[keyOf(ix, iy, iz)].push_back(i);
	}

	// 每点 PCA 法向, OpenMP 并行
	#pragma omp parallel for schedule(dynamic, 1024)
	for (int i = 0; i < n; i++)
	{
		double cx = 0, cy = 0, cz = 0; int cnt = 0;
		// 收集 3x3x3 邻域点
		int nbIdx[512]; int nb = 0;
		for (int a = -1; a <= 1; a++)
			for (int b = -1; b <= 1; b++)
				for (int c = -1; c <= 1; c++)
				{
					auto it = grid.find(keyOf(ci[i] + a, cj[i] + b, ck[i] + c));
					if (it == grid.end()) continue;
					for (int idx : it->second)
					{
						if (nb < 512) nbIdx[nb++] = idx;
					}
				}
		if (nb < 4) { continue; }   // 邻域太少, 保持默认法向
		for (int k = 0; k < nb; k++) { cx += m_pts[nbIdx[k]].x; cy += m_pts[nbIdx[k]].y; cz += m_pts[nbIdx[k]].z; cnt++; }
		cx /= cnt; cy /= cnt; cz /= cnt;
		// 协方差矩阵(对称) c00,c01,c02,c11,c12,c22
		double c00 = 0, c01 = 0, c02 = 0, c11 = 0, c12 = 0, c22 = 0;
		for (int k = 0; k < nb; k++)
		{
			double ex = m_pts[nbIdx[k]].x - cx, ey = m_pts[nbIdx[k]].y - cy, ez = m_pts[nbIdx[k]].z - cz;
			c00 += ex * ex; c01 += ex * ey; c02 += ex * ez;
			c11 += ey * ey; c12 += ey * ez; c22 += ez * ez;
		}
		// Jacobi 求 3x3 对称阵特征分解, 取最小特征值对应特征向量=法向
		double A[3][3] = { {c00,c01,c02},{c01,c11,c12},{c02,c12,c22} };
		double V[3][3] = { {1,0,0},{0,1,0},{0,0,1} };
		for (int sweep = 0; sweep < 8; sweep++)
		{
			// 找最大非对角元
			int p = 0, q = 1; double mx = std::fabs(A[0][1]);
			if (std::fabs(A[0][2]) > mx) { mx = std::fabs(A[0][2]); p = 0; q = 2; }
			if (std::fabs(A[1][2]) > mx) { mx = std::fabs(A[1][2]); p = 1; q = 2; }
			if (mx < 1e-12) break;
			double app = A[p][p], aqq = A[q][q], apq = A[p][q];
			double phi = 0.5 * atan2(2 * apq, aqq - app);
			double cph = cos(phi), sph = sin(phi);
			for (int k = 0; k < 3; k++)
			{
				double akp = A[k][p], akq = A[k][q];
				A[k][p] = cph * akp - sph * akq;
				A[k][q] = sph * akp + cph * akq;
			}
			for (int k = 0; k < 3; k++)
			{
				double apk = A[p][k], aqk = A[q][k];
				A[p][k] = cph * apk - sph * aqk;
				A[q][k] = sph * apk + cph * aqk;
			}
			for (int k = 0; k < 3; k++)
			{
				double vkp = V[k][p], vkq = V[k][q];
				V[k][p] = cph * vkp - sph * vkq;
				V[k][q] = sph * vkp + cph * vkq;
			}
		}
		// 最小特征值索引
		int mi = 0; double mv = A[0][0];
		if (A[1][1] < mv) { mv = A[1][1]; mi = 1; }
		if (A[2][2] < mv) { mv = A[2][2]; mi = 2; }
		double nx = V[0][mi], ny = V[1][mi], nz = V[2][mi];
		double nl = std::sqrt(nx * nx + ny * ny + nz * nz);
		if (nl < 1e-12) continue;
		nx /= nl; ny /= nl; nz /= nl;
		// 定向: 指向相机(原点)一侧, 即使 dot(N, p) <= 0
		if (nx * m_pts[i].x + ny * m_pts[i].y + nz * m_pts[i].z > 0) { nx = -nx; ny = -ny; nz = -nz; }
		m_normals[i] = Point3D{ (float)nx, (float)ny, (float)nz };
	}
	logMsg("法向估计完成(OpenMP): 点数=" + std::to_string(n) + " 体素边长=" + std::to_string(h));
}

// 屏幕点 → 单位球向量(arcball)
glm::vec3 CPointCloudView::mapToArcball(const CPoint &pt)
{
	CRect rc; GetClientRect(&rc);
	float w = rc.Width() > 0 ? (float)rc.Width() : 1.0f;
	float hh = rc.Height() > 0 ? (float)rc.Height() : 1.0f;
	float x = (2.0f * pt.x - w) / w;      // [-1,1]
	float y = (hh - 2.0f * pt.y) / hh;    // [-1,1], 上为正
	float d = x * x + y * y;
	glm::vec3 v;
	if (d > 1.0f) { float s = 1.0f / std::sqrt(d); v = glm::vec3(x * s, y * s, 0.0f); }
	else          { v = glm::vec3(x, y, std::sqrt(1.0f - d)); }
	return v;
}

void CPointCloudView::setRangeImage(const cv::Mat &range)
{
	if (m_pts.empty()) return;
	m_colors.resize(m_pts.size());
	double minZ = 1e9, maxZ = -1e9;
	for (auto &p : m_pts)
	{
		if (p.z < minZ) minZ = p.z;
		if (p.z > maxZ) maxZ = p.z;
	}
	double rangeZ = maxZ - minZ;
	if (rangeZ < 1e-6) rangeZ = 1;
	for (size_t i = 0; i < m_pts.size(); i++)
	{
		float t = (float)((m_pts[i].z - minZ) / rangeZ);
		float r = std::min(1.0f, std::max(0.0f, 1.5f - fabsf(4 * t - 3)));
		float g = std::min(1.0f, std::max(0.0f, 1.5f - fabsf(4 * t - 2)));
		float b = std::min(1.0f, std::max(0.0f, 1.5f - fabsf(4 * t - 1)));
		m_colors[i] = cv::Vec3b((uchar)(b * 255), (uchar)(g * 255), (uchar)(r * 255));
	}
	uploadBuffer();
	redraw();
}

void CPointCloudView::computeBoundingBox()
{
	if (m_pts.empty()) { m_center = { 0,0,0 }; m_radius = 1; return; }

	// 旋转枢轴用质心(均值), 比包围盒中心更稳, 不易被离群点带偏
	double cx = 0, cy = 0, cz = 0;
	for (auto &p : m_pts) { cx += p.x; cy += p.y; cz += p.z; }
	double n = (double)m_pts.size();
	m_center.x = (float)(cx / n);
	m_center.y = (float)(cy / n);
	m_center.z = (float)(cz / n);

	// 尺寸用"到质心距离的均方根×2", 对少量离群点不敏感(避免视图被撑得过小)
	double sum2 = 0;
	for (auto &p : m_pts)
	{
		double dx = p.x - m_center.x, dy = p.y - m_center.y, dz = p.z - m_center.z;
		sum2 += dx * dx + dy * dy + dz * dz;
	}
	float rms = (float)std::sqrt(sum2 / n);
	m_radius = rms * 2.0f;
	if (m_radius < 1e-3f) m_radius = 1.0f;
	m_pivot = m_center;   // 新点云默认绕质心旋转
	m_scale = 1.0f;
}

void CPointCloudView::redraw()
{
	// 直接渲染(不走 WM_PAINT): 交互时最跟手, 避免绘制周期/DWM合成带来的延迟。
	if (!GetSafeHwnd() || !m_bGLInit) return;
	wglMakeCurrent(m_hDC, m_hRC);
	CRect rc; GetClientRect(&rc);
	if (rc.Width() > 0 && rc.Height() > 0)
	{
		setupViewport(rc.Width(), rc.Height());
		renderScene();
		SwapBuffers(m_hDC);
	}
	ValidateRect(NULL);   // 已直接绘制, 撤销无效区, 避免随后重复触发 WM_PAINT
}

void CPointCloudView::OnLButtonDown(UINT nFlags, CPoint point)
{
	m_bLDown = true; m_interacting = true; m_lastPt = point; SetCapture();
	m_dragMoves = 0;
	m_totDx = 0; m_totDy = 0;
	logMsg("[交互] 左键按下, 开始旋转");
	CWnd::OnLButtonDown(nFlags, point);
}

void CPointCloudView::OnLButtonUp(UINT nFlags, CPoint point)
{
	m_bLDown = false; m_interacting = false; ReleaseCapture();
	// 诊断: 累计像素位移 vs 累计施加角度, 用于区分"位移丢失/圆周抵消" 与 "缩放太小看着转得少"
	const float sensDbg = 1.0f;
	logMsg("[交互] 左键松开, 移动消息数=" + std::to_string(m_dragMoves) +
		" 累计位移|dx|=" + std::to_string(m_totDx) + " |dy|=" + std::to_string(m_totDy) +
		" 施加角度≈yaw " + std::to_string(m_totDx * sensDbg) + "° / pitch " + std::to_string(m_totDy * sensDbg) + "°" +
		" 当前缩放m_scale=" + std::to_string(m_scale) +
		" 渲染帧数=" + std::to_string(m_frameCnt));
	redraw();   // 松手后画全量
	CWnd::OnLButtonUp(nFlags, point);
}

void CPointCloudView::OnRButtonDown(UINT nFlags, CPoint point)
{
	m_bRDown = true; m_interacting = true; m_lastPt = point; SetCapture();
	CWnd::OnRButtonDown(nFlags, point);
}

void CPointCloudView::OnRButtonUp(UINT nFlags, CPoint point)
{
	m_bRDown = false; m_interacting = false; ReleaseCapture();
	redraw();
	CWnd::OnRButtonUp(nFlags, point);
}

void CPointCloudView::OnMButtonDown(UINT nFlags, CPoint point)
{
	m_bRDown = true; m_interacting = true; m_lastPt = point; SetCapture();   // 中键也用于平移
	CWnd::OnMButtonDown(nFlags, point);
}

void CPointCloudView::OnMButtonUp(UINT nFlags, CPoint point)
{
	m_bRDown = false; m_interacting = false; ReleaseCapture();
	redraw();
	CWnd::OnMButtonUp(nFlags, point);
}

void CPointCloudView::OnLButtonDblClk(UINT nFlags, CPoint point)
{
	resetView();   // 双击左键: 复位视角
	CWnd::OnLButtonDblClk(nFlags, point);
}

void CPointCloudView::OnMouseMove(UINT nFlags, CPoint point)
{
	int dx = point.x - m_lastPt.x;
	int dy = point.y - m_lastPt.y;

	if (m_bLDown && (dx != 0 || dy != 0))
	{
		// 左键: 增量式 turntable 旋转(响应快、可预测、无翻滚)。
		//   水平移动 dx → 绕世界竖直轴Y偏航(右乘, 世界系);
		//   垂直移动 dy → 绕屏幕水平轴X俯仰(左乘, 视图系)。
		// 每次用相对上一帧的增量累积到 m_rotMat, 拖多少转多少。
		const float sens = 1.0f;   // 度/像素 (0.5→1.0, 提高跟手感)
		glm::mat4 yaw = glm::rotate(glm::mat4(1.0f), glm::radians((float)dx * sens), glm::vec3(0, 1, 0));
		glm::mat4 pitch = glm::rotate(glm::mat4(1.0f), glm::radians((float)dy * sens), glm::vec3(1, 0, 0));
		m_rotMat = pitch * m_rotMat * yaw;
		m_dragMoves++;
		m_totDx += (dx < 0 ? -dx : dx);
		m_totDy += (dy < 0 ? -dy : dy);
		if (m_dragMoves % 25 == 0)
			logMsg("[交互] 旋转中: 累计移动=" + std::to_string(m_dragMoves) +
				" 本次dx=" + std::to_string(dx) + " dy=" + std::to_string(dy) +
				" 帧数=" + std::to_string(m_frameCnt));
		redraw();
	}
	if (m_bRDown)
	{
		// 右/中键: 平移。移动轨道中心 pivot(而非平移图像), 使抓取点跟随光标,
		// 且旋转始终绕屏幕中心的那个点进行。
		CRect rc; GetClientRect(&rc);
		int h = rc.Height() > 0 ? rc.Height() : 1;
		float r = (m_radius > 1e-6f ? m_radius : 1.0f);
		float halfH = r / m_scale;                 // 正交半高(世界)
		float worldPerPix = 2.0f * halfH / (float)h;
		// 当前视图旋转下, 屏幕右/上方向对应的世界方向
		glm::mat3 R3 = glm::mat3(m_rotMat);
		glm::vec3 camRight = glm::transpose(R3) * glm::vec3(1, 0, 0);
		glm::vec3 camUp    = glm::transpose(R3) * glm::vec3(0, 1, 0);
		glm::vec3 piv(m_pivot.x, m_pivot.y, m_pivot.z);
		piv -= camRight * (dx * worldPerPix);
		piv += camUp    * (dy * worldPerPix);
		m_pivot.x = piv.x; m_pivot.y = piv.y; m_pivot.z = piv.z;
		redraw();
	}
	m_lastPt = point;
	CWnd::OnMouseMove(nFlags, point);
}

BOOL CPointCloudView::OnMouseWheel(UINT nFlags, short zDelta, CPoint pt)
{
	// 更平滑的缩放步进; 按滚动量比例缩放
	float steps = (float)zDelta / 120.0f;
	float factor = powf(1.15f, steps);
	m_scale *= factor;
	// 正交缩放: 半高=r/scale。scale∈[0.05,50] → 半高 20r(看全貌) ~ r/50(深放大)
	if (m_scale < 0.05f) m_scale = 0.05f;
	if (m_scale > 50.0f) m_scale = 50.0f;
	redraw();
	return CWnd::OnMouseWheel(nFlags, zDelta, pt);
}

void CPointCloudView::resetView()
{
	m_rotMat = glm::rotate(glm::mat4(1.0f), glm::radians(-20.0f), glm::vec3(1, 0, 0));
	m_pivot = m_center;   // 轨道中心复位到质心
	m_scale = 1.0f;
	redraw();
}
