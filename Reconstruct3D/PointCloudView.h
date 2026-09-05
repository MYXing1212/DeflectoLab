#pragma once

#include <string>
#include <glm/glm.hpp>
#include "GLShader.h"

struct Point3D
{
	float x, y, z;
};

class CPointCloudView : public CWnd
{
	DECLARE_DYNAMIC(CPointCloudView)
public:
	CPointCloudView();
	virtual ~CPointCloudView();

	BOOL Create(const RECT &rect, CWnd *pParent, UINT nID);

	// 设置点云数据
	void setPointCloud(const std::vector<Point3D> &pts);

	// 设置深度图(用于着色)
	void setRangeImage(const cv::Mat &range);

	// 点大小 (像素)
	void setPointSize(float s) { m_pointSize = (s < 1.0f ? 1.0f : s); redraw(); }
	float pointSize() const { return m_pointSize; }

	// 着色模式: true=按深度Jet伪彩, false=纯色
	void setColorByDepth(bool on) { m_colorByDepth = on; redraw(); }
	bool colorByDepth() const { return m_colorByDepth; }

	// 纯色模式下的点云颜色 (0~255)
	void setSolidColor(unsigned char r, unsigned char g, unsigned char b)
	{ m_solidR = r; m_solidG = g; m_solidB = b; redraw(); }

	// 光照着色开关(基于法向估计, 增强立体感)
	void setLighting(bool on) { m_lighting = on; redraw(); }
	bool lighting() const { return m_lighting; }

	// 诊断日志文件路径(为空则用exe目录下 pointcloud_render.log)
	void setLogPath(const std::string &path) { m_logPath = path; }

	// 复位视角(旋转/平移/缩放归位)
	void resetView();

	// 重绘
	void redraw();

protected:
	afx_msg int  OnCreate(LPCREATESTRUCT lpCreateStruct);
	afx_msg void OnSize(UINT nType, int cx, int cy);
	afx_msg void OnPaint();
	afx_msg BOOL OnEraseBkgnd(CDC *pDC);
	afx_msg void OnLButtonDown(UINT nFlags, CPoint point);
	afx_msg void OnLButtonUp(UINT nFlags, CPoint point);
	afx_msg void OnMouseMove(UINT nFlags, CPoint point);
	afx_msg void OnRButtonDown(UINT nFlags, CPoint point);
	afx_msg void OnRButtonUp(UINT nFlags, CPoint point);
	afx_msg void OnMButtonDown(UINT nFlags, CPoint point);
	afx_msg void OnMButtonUp(UINT nFlags, CPoint point);
	afx_msg void OnLButtonDblClk(UINT nFlags, CPoint point);
	afx_msg BOOL OnMouseWheel(UINT nFlags, short zDelta, CPoint pt);
	DECLARE_MESSAGE_MAP()

private:
	HGLRC m_hRC = nullptr;
	HDC   m_hDC = nullptr;
	bool  m_bGLInit = false;

	GLShaderProgram m_shader;
	unsigned int m_vao = 0;
	unsigned int m_vbo = 0;
	int          m_vertCount = 0;

	std::vector<Point3D>  m_pts;
	std::vector<Point3D>  m_normals;      // 每点法向(OpenMP估计或外部提供)
	std::vector<cv::Vec3b> m_colors;      // [B,G,R] 深度伪彩
	std::vector<float>    m_vboData;      // 交错: x,y,z, r,g,b, nx,ny,nz

	// 视角控制: 累积旋转矩阵(arcball, 抓取跟随鼠标)
	glm::mat4 m_rotMat = glm::mat4(1.0f);
	glm::mat4 m_rotDown = glm::mat4(1.0f);   // 按下时的旋转快照
	glm::vec3 m_arcDown = glm::vec3(0, 0, 1); // 按下时的球面向量
	float m_scale = 1.0f;
	CPoint m_lastPt;
	bool   m_bLDown = false;
	bool   m_bRDown = false;
	bool   m_lighting = true;              // 默认开启光照着色
	bool   m_interacting = false;          // 拖拽/交互中(降采样预览, 保证跟手)
	int    m_drawBudget = 80000;           // 交互时最多绘制的点数(越小越跟手)
	int    m_dragMoves = 0;                // 诊断: 本次拖拽收到的移动数
	long   m_frameCnt = 0;                 // 诊断: 渲染帧计数
	int    m_totDx = 0, m_totDy = 0;       // 诊断: 本次拖拽累计像素位移

	// 点云包围盒 / 轨道中心
	Point3D m_center = { 0,0,0 };   // 质心(默认轨道中心)
	Point3D m_pivot = { 0,0,0 };    // 当前旋转轨道中心(始终映射到屏幕中心, 平移会改变它)
	float   m_radius = 1.0f;

	// 渲染样式
	float m_pointSize = 3.0f;
	bool  m_colorByDepth = true;
	unsigned char m_solidR = 180, m_solidG = 200, m_solidB = 255;

	// 诊断日志
	std::string m_logPath;
	bool m_loggedFrame = false;   // 首帧详细日志只打一次
	void logMsg(const std::string &msg);
	void logGLError(const std::string &stage);
	std::string exeDir();

	bool initGL();
	bool initShaderAndBuffers();
	void uploadBuffer();          // 由 m_pts/m_colors/m_normals 生成交错缓冲并上传 VBO
	void setupViewport(int w, int h);
	void renderScene();
	void computeBoundingBox();
	void estimateNormals();       // 基于体素网格kNN+PCA的法向估计(OpenMP并行)
	glm::vec3 mapToArcball(const CPoint &pt);  // 屏幕点→单位球向量(arcball)
};
