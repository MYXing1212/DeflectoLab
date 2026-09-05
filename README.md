# DeflectoLab — 双目条纹偏折术仿真与三维重建

DeflectoLab 是一套用于**镜面/高反光物体三维面形测量**的双目条纹偏折术（Fringe Deflectometry）工具链。它由两部分组成：

1. **Blender 渲染端**（`binocular_deflectometry.py`）：在 Blender 中搭建"显示屏 + 被测镜面 + 双目相机"的虚拟场景，投射相移正弦条纹并渲染出左右相机的条纹图，同时导出相机内参与标定坐标系外参。
2. **C++ 重建端**（`Reconstruct3D/`）：读取渲染出的条纹图，完成相位解包裹、屏幕坐标映射、双目深度搜索 + 法向一致性重建，并用 OpenGL/GLSL 将点云实时渲染到交互式三维窗口中。

> 该项目主要用于算法验证与教学：在完全可控的仿真环境里研究偏折术的相位解算、标定与重建精度。

---

## 目录结构

```
DeflectoLab/
├── binocular_deflectometry.py     # Blender 端：场景搭建 / 条纹投射 / 渲染 / 标定参数导出
├── Reconstruct3D.sln              # Visual Studio 解决方案
├── Reconstruct3D/                 # C++ 重建端（MFC + OpenGL）
│   ├── MainDlg.*                  # 主界面：流程编排、参数管理、日志
│   ├── PhaseUnwrap.*              # 多频外差相位解包裹
│   ├── Match.*                    # 双目深度搜索 + 法向一致性重建
│   ├── CalibParas.*               # 标定参数（内参/外参/屏参）解析
│   ├── Rectifier.*                # 立体校正相关
│   ├── PointCloudView.*           # OpenGL 点云交互视图（旋转/平移/缩放/光照）
│   ├── GLShader.*                 # GLSL 着色器加载封装
│   ├── shaders/                   # 顶点/片元着色器（pointcloud.vert / .frag）
│   ├── 3rdparty/                  # 依赖库（OpenCV / GLEW / GLFW / freetype / freeglut / glm）
│   └── 参数说明.md                # 参数与坐标系约定说明
└── README.md
```

---

## 处理流程

```
Blender 场景
   │  投射相移条纹（纵向 V + 横向 H，多频外差）
   ▼
左/右相机条纹图（.bmp）  +  标定参数（相机内参、C_LCam/C_RCam/C_Screen 外参、屏像素尺寸/分辨率）
   │
   ▼  Reconstruct3D
相位解包裹（PhaseUnwrap） → 屏幕坐标映射 → 双目深度搜索 + 法向一致性（Match）
   │
   ▼
三维点云（含 OpenMP 法向估计） → OpenGL/GLSL 交互渲染（PointCloudView）
```

### 坐标系约定
- `C_LCam` / `C_RCam`：左右相机坐标系，遵循 OpenCV 约定——**Z+ 指向相机正前方，X+ 指向右侧，Y+ 沿画面竖直向下**。脚本会依据相机节点自动对齐（`cam.matrix_world @ diag(1,-1,-1,1)`）。
- `C_Screen`：显示屏坐标系，原点位于屏面左上角，姿态贴合屏面网格。
- 纵向（V）与横向（H）条纹使用**不同的条纹节距（pitch）组合**以提升重建精度，Blender 与 C++ 端需保持一致。

---

## 环境要求

**Blender 端**
- Blender 3.x 及以上（内置 Python，需 `numpy`）
- 运行方式：在 Blender 的脚本编辑器中打开并运行 `binocular_deflectometry.py`，脚本会构建一个操作面板（UI Panel）。

**C++ 重建端**
- Windows 10/11
- Visual Studio 2022（MSVC v143 工具集），MFC 组件
- 配置：**Release / x64**
- 已启用 OpenMP（用于点云法向并行估计）
- 依赖库已随 `Reconstruct3D/3rdparty/` 一同提供（OpenCV 3.4.11、GLEW 2.1.0、GLFW、freetype、freeglut、glm）

---

## 构建与运行

### C++ 重建端
1. 用 Visual Studio 打开 `Reconstruct3D.sln`。
2. 选择 **Release / x64** 配置。
3. 生成解决方案（`Reconstruct3D` 项目）。
4. 运行生成的可执行文件；确保 `shaders/` 目录与 `3rdparty` 运行库 DLL（如 `opencv_world3411.dll`、`freetype.dll`、`freeglut.dll`）位于可执行文件同级或系统可搜索到的路径。

### Blender 渲染端
1. 打开 Blender，在脚本编辑器中加载 `binocular_deflectometry.py`。
2. 运行脚本，使用弹出的面板设置：条纹的直流分量与幅值（避免过曝）、渲染输出目录、C_Screen 姿态微调（含"绕 Z 轴转 90°"按钮）等。
3. 渲染左右相机条纹图并导出标定参数，供 C++ 端读取。

---

## 三维视图交互（PointCloudView）
- **左键拖拽**：旋转（turntable 模型，正交投影）
- **右键 / 中键拖拽**：平移（移动轨道中心）
- **滚轮**：缩放
- **双击左键 / 重置按钮**：复位视角
- 工具栏支持：点大小、按深度着色、光照开关、点云颜色、重置视角

---

## 说明
- `3rdparty/` 中的第三方库为方便一键构建而随仓库提供；如需精简仓库体积，可改为通过包管理器或本地路径引用。
- 构建输出（`x64/`）、IDE 缓存（`.vs/`）、渲染数据图（`data/`）、Blender 工程（`*.blend`）等已在 `.gitignore` 中忽略。
