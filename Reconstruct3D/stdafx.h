// stdafx.h : 标准系统包含文件的包含文件，
// 或是经常使用但不常更改的
// 项目特定包含文件

#pragma once

#ifndef _SECURE_ATL
#define _SECURE_ATL 1
#endif

#ifndef VC_EXTRALEAN
#define VC_EXTRALEAN
#endif

#include "targetver.h"

#define _ATL_CSTRING_EXPLICIT_CONSTRUCTORS
#define _AFX_ALL_MFC_ENTRIES

#include <afxwin.h>
#include <afxext.h>
#include <afxdisp.h>
#include <afxdtctl.h>
#include <afxcmn.h>
#include <afxdlgs.h>
#include <afxdialogex.h>

#ifdef _UNICODE
#if defined _M_IX86
#pragma comment(linker,"/manifestdependency:\"type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='x86' publicKeyToken='6595b64144ccf1df' language='*'\"")
#elif defined _M_X64
#pragma comment(linker,"/manifestdependency:\"type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='amd64' publicKeyToken='6595b64144ccf1df' language='*'\"")
#endif
#endif

// OpenCV
#include <opencv2/opencv.hpp>

// GLM (第三方库，抑制其头文件产生的警告)
#pragma warning(push)
#pragma warning(disable: 4828) // 源文件包含无效字符 (代码页 65001)
#pragma warning(disable: 5033) // “register”不再为受支持的存储类
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#pragma warning(pop)

// OpenGL
#include <GL/glew.h>
#include <GL/wglew.h>
#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "glu32.lib")

// 标准库
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <thread>
#include <chrono>