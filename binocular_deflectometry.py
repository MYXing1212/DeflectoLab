import os
import sys
import json
import bpy
import math
import mathutils
import threading
import queue
import numpy as np


# 全局变量定义
calibration_board = None
projector = None
camera_left = None
camera_right = None

pattern_path = r"D:\BlenderSave\blenderTwo\data"
base_path = r"D:\BlenderSave\blenderTwo\save2"

# 渲染输出目录的持久化配置文件(记忆用户设置的路径)
CONFIG_FILE = os.path.join(os.path.dirname(base_path), "render_config.json")


def loadRenderConfig():
    """从配置文件加载持久化的渲染输出目录"""
    global base_path
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
            saved = cfg.get('base_path', '')
            if saved and os.path.isabs(saved):
                base_path = saved
    except Exception:
        pass


def saveRenderConfig():
    """保存当前渲染输出目录到配置文件(记忆用户设置)"""
    try:
        os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump({'base_path': base_path}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存渲染配置失败: {e}")


# 启动时加载持久化配置
loadRenderConfig()

# 双目相机名称配置（根据场景实际名称修改）
LEFT_CAMERA_NAME = "Camera"
RIGHT_CAMERA_NAME = "Camera.001"
# 显示屏/投影仪对象名称
# 注意: 必须与实际显示条纹纹理的对象一致(setup_projector_texture 使用 'Light'),
#       否则导出的屏外参描述的是另一块平面, 会出现"屏在相机后方(z<0)"等外参异常。
SCREEN_OBJECT_NAME = "Light"      # 真正显示条纹、被相机(经镜面)观察到的屏面对象
PROJECTOR_NAME = "平面.001"        # 兼容旧场景命名(找不到 Light 时回退)
# 标定坐标系空对象名称(外参直接由这些坐标系的matrix_world计算)
LCAM_FRAME_NAME = "C_LCam"
RCAM_FRAME_NAME = "C_RCam"
SCREEN_FRAME_NAME = "C_Screen"

# 相机分辨率
CAM_WIDTH = 1440
CAM_HEIGHT = 1080
# 显示屏分辨率（开始渲染时按UI配置更新：文件/生成模式统一取UI的显示屏宽高，默认1280x800）
PROJ_RES = {'w': 1280, 'h': 800}
# 程序生成条纹图（生成模式）的输出目录
generated_path = r"D:\BlenderSave\blenderTwo\generated"

# 异步渲染队列相关全局变量
renderQueue = []
currentRenderTask = None
totalRenderCount = 0
completedRenderCount = 0
allRendersDoneCallback = None
renderCancelled = False
isPaused = False
currentFrameCancelled = False
pendingRenderTasks = []
pendingDoneCallback = None
isCancelledIdle = False

# 本次渲染的条纹参数记录，渲染完成后写入 pattern_info.txt 供 Reconstruct3D 自动导入
lastPatternInfo = {'pitches': [], 'pitchesH': [], 'offset': 0.0}


def get_scene_unit_scale():
    """获取Blender内部坐标值到mm的转换因子
    Blender内部所有坐标(matrix_world/location/dimensions)永远以内部单位存储，
    物理米数 = 内部值 * scale_length；unit_settings.length_unit 仅影响UI显示，不影响坐标数值
    返回：内部单位 → mm 的转换因子（默认场景 scale_length=1.0 时为 1000.0）
    """
    scene = bpy.context.scene
    scale = scene.unit_settings.scale_length
    return scale * 1000.0


def enableGpuRendering():
    prefs = bpy.context.preferences
    cyclesPrefs = prefs.addons['cycles'].preferences
    try:
        cyclesPrefs.refresh_devices()
    except Exception:
        pass
    deviceTypes = ['OPTIX', 'CUDA', 'HIP', 'METAL', 'ONEAPI']
    computeDeviceType = None
    for dt in deviceTypes:
        try:
            cyclesPrefs.compute_device_type = dt
            computeDeviceType = dt
            break
        except TypeError:
            continue
    if computeDeviceType is None:
        print("警告：未找到可用的GPU计算设备类型，回退到CPU渲染")
        bpy.context.scene.cycles.device = 'CPU'
        return False
    devices = cyclesPrefs.devices
    gpuEnabled = False
    for dev in devices:
        if dev.type == computeDeviceType:
            dev.use = True
            gpuEnabled = True
            print(f"已启用GPU渲染设备: {dev.name} ({computeDeviceType})")
        else:
            dev.use = False
    if gpuEnabled:
        bpy.context.scene.cycles.device = 'GPU'
        return True
    else:
        print("警告：未找到可用的GPU设备，回退到CPU渲染")
        bpy.context.scene.cycles.device = 'CPU'
        return False


def setup_render_settings():
    scene = bpy.context.scene
    # EXR是线性float格式, 写盘不经sRGB OETF, 作为线性像素中转(渲染后读EXR量化写8bit灰度BMP)
    scene.render.image_settings.file_format = 'OPEN_EXR'
    scene.render.image_settings.color_depth = '16'
    scene.render.image_settings.exr_codec = 'ZIP'
    scene.render.resolution_x = 1440
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False

    # 色彩管理: view_transform='Raw'使view transform恒等(线性), look=None禁用风格曲线
    # 输出: 渲染写线性EXR -> 读EXR线性像素 -> 手动量化写8bit灰度BMP(无sRGB/gamma畸变)
    scene.view_settings.view_transform = 'Raw'
    scene.view_settings.look = 'None'

    cycles = scene.cycles
    # 渲染提速：持久化场景数据，避免每幅图像重复重建BVH/场景
    scene.render.use_persistent_data = True
    # 自适应采样：噪声已收敛的区域提前停止采样，平坦区域大幅提速
    cycles.samples = 32
    cycles.use_adaptive_sampling = True
    cycles.adaptive_threshold = 0.02
    # 关闭反射/折射焦散（偏折测量为镜面反射光路，可安全关闭）
    cycles.caustics_reflective = False
    cycles.caustics_refractive = False
    # 偏折测量主要依赖镜面反射（相机->镜面->投影屏），降低多余反弹
    cycles.max_bounces = 8
    cycles.diffuse_bounces = 2
    cycles.glossy_bounces = 4
    cycles.transmission_bounces = 2
    cycles.volume_bounces = 0

    enableGpuRendering()


def get_fringe_images(pattern_path):
    valid_extensions = {'.jpg', '.png', '.bmp'}
    all_files = [f for f in os.listdir(pattern_path)
                 if os.path.splitext(f)[1].lower() in valid_extensions]
    all_files.sort(key=lambda x: int(''.join(filter(str.isdigit, os.path.splitext(x)[0])) or 0))
    return [os.path.join(pattern_path, f) for f in all_files]


# 3频4步相移法候选频率组合（周期数N1,N2,N3），最高频组合按显示屏宽度动态过滤
# 3频4步相移法候选频率组合（像素周期P1<P2<P3，等差设计），按显示屏宽度动态过滤
# 约束推导:
#   1) 相邻周期比值<=8（用户要求）
#   2) Reconstruct3D多频外差展开要求合成波长 P123 = P12*P23/|P23-P12| >= 屏宽(全屏无混叠)
#      等差周期(P,P+d,P+2d)时 P123 = P1*P2*P3/(2*d^2)
#   3) 最细周期P1>=4px(采样)，最粗P3<=屏宽(至少一个整周期)
FREQ_GROUPS = [(13, 14, 15), (18, 20, 22), (20, 22, 24), (32, 36, 40),
               (40, 46, 52), (48, 56, 64), (56, 65, 74), (64, 75, 86)]


def fringeSynthWavelength(group):
    """三频外差合成波长 P123（像素），PhaseUnwrap 展开链的有效唯一范围"""
    p1, p2, p3 = group
    p12 = p1 * p2 / (p2 - p1)
    p23 = p2 * p3 / (p3 - p2)
    return p12 * p23 / abs(p23 - p12)


# items回调返回的字符串必须保持Python侧引用，否则被GC后Blender读到悬空指针导致UI乱码
_freqGroupItemsCache = []


_freqGroupItemsCacheH = []


def _buildFreqItems(span, fallback, prefer=None):
    """构建满足 P123>=span、最细>=4px、最粗<=span 的3频组合下拉项。
    prefer: 若给定且有效, 将其移到列表首位(动态EnumProperty默认项=首项)。"""
    items = []
    for g in FREQ_GROUPS:
        p1, p2, p3 = g
        if p1 < 4 or p3 > span:
            continue
        p123 = fringeSynthWavelength(g)
        if p123 < span:
            continue
        r1, r2 = p2 / p1, p3 / p2
        items.append((f"{p1}_{p2}_{p3}",
                      f"P=({p1},{p2},{p3})px  比{r1:.2f}/{r2:.2f}  P123={p123:.0f}px",
                      f"3频组合: 像素周期({p1},{p2},{p3})，相邻比{r1:.2f}/{r2:.2f}，"
                      f"合成波长{p123:.0f}px(>={span}可全展开)"))
    if prefer is not None:
        pid = f"{prefer[0]}_{prefer[1]}_{prefer[2]}"
        idx = next((i for i, it in enumerate(items) if it[0] == pid), -1)
        if idx > 0:
            items.insert(0, items.pop(idx))
    if not items:
        items.append(fallback)
    return items


def freqGroupItems(self, context):
    """纵向(沿屏宽)3频组合: 校验 P123>=屏宽; 默认最细(13,14,15), 沿屏宽精度最高"""
    global _freqGroupItemsCache
    _freqGroupItemsCache = _buildFreqItems(context.scene.defl_proj_width,
                                           ("13_14_15", "P=(13,14,15)px", "回退组合"),
                                           prefer=(13, 14, 15))
    return _freqGroupItemsCache


def freqGroupItemsH(self, context):
    """横向(沿屏高)3频组合: 校验 P123>=屏高; 默认较宽的(20,22,24), 避免横向条纹过细"""
    global _freqGroupItemsCacheH
    _freqGroupItemsCacheH = _buildFreqItems(context.scene.defl_proj_height,
                                            ("20_22_24", "P=(20,22,24)px", "回退组合"),
                                            prefer=(20, 22, 24))
    return _freqGroupItemsCacheH


def _saveGrayPngBpy(img8, path):
    """用Blender原生API保存8bit灰度PNG（无需cv2依赖）
    注意: image.pixels是RGBA浮点(0-1)且行序为自底向上，需翻转以保持图像方向；
    colorspace设为Non-Color保证8bit值线性写入，与cv2.imwrite结果一致
    """
    h, w = img8.shape
    image = bpy.data.images.new("_fringe_tmp", width=w, height=h, alpha=False)
    try:
        try:
            image.colorspace_settings.name = 'Non-Color'
        except Exception:
            pass
        val = img8[::-1].astype(np.float32) / 255.0
        rgba = np.empty((h, w, 4), dtype=np.float32)
        rgba[..., 0] = val
        rgba[..., 1] = val
        rgba[..., 2] = val
        rgba[..., 3] = 1.0
        image.pixels.foreach_set(rgba.ravel())
        image.filepath_raw = path
        image.file_format = 'PNG'
        image.save()
    finally:
        bpy.data.images.remove(image)


def generateFringePatterns(width, height, pitchesV, pitchesH=None, bias=127.5, amplitude=100.0):
    """3频4步相移法生成条纹图：pitchesV/pitchesH为像素周期(P1<P2<P3)，每频率4步相移
    纵向12幅(用pitchesV,相位沿X变化)+横向12幅(用pitchesH,相位沿Y变化)共24幅。
    纵向沿屏宽(width)编码, 横向沿屏高(height)编码, 故两方向可用不同周期组合以提升精度。

    灰度 = bias + amplitude·cos(phase)
      bias      : 直流分量(条纹平均灰度)
      amplitude : 条纹幅值(对比度)
    为避免过曝(灰度截断到255导致解相误差), 需保证 bias+amplitude <= 255 且 bias-amplitude >= 0。
    """
    # 防止参数越界导致灰度截断: 自动收敛到不过曝的安全区间
    bias = float(np.clip(bias, 0.0, 255.0))
    maxAmp = min(bias, 255.0 - bias)
    if amplitude > maxAmp:
        print(f"[条纹生成] 幅值 {amplitude:g} 超出不过曝上限 {maxAmp:g} (直流={bias:g}), 已自动收敛以避免过曝")
        amplitude = maxAmp
    if pitchesH is None:
        pitchesH = pitchesV
    try:
        import cv2
        saveGrayPng = lambda img, path: cv2.imwrite(path, img)
    except ImportError:
        print("提示: Blender Python未安装cv2，改用Blender原生API保存条纹图")
        saveGrayPng = _saveGrayPngBpy
    os.makedirs(generated_path, exist_ok=True)
    for f in os.listdir(generated_path):
        fp = os.path.join(generated_path, f)
        try:
            os.remove(fp)
        except Exception:
            pass
    xs = np.arange(width, dtype=np.float64)
    ys = np.arange(height, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys)
    paths = []
    # 相位 = 2π·x/P + 2πk/4：与 Reconstruct3D 的 atan2(I4-I2, I1-I3)
    # 解相公式匹配，保证解出相位 = +2πx/P
    for pitch in pitchesV:   # 纵向条纹: 相位沿 X(屏宽)
        for k in range(4):
            phase = 2.0 * np.pi * X / pitch + 0.5 * np.pi * k
            img = np.clip(bias + amplitude * np.cos(phase), 0, 255).astype(np.uint8)
            path = os.path.join(generated_path, f"v_{pitch:g}_{k}.png")
            saveGrayPng(img, path)
            paths.append(path)
    for pitch in pitchesH:   # 横向条纹: 相位沿 Y(屏高)
        for k in range(4):
            phase = 2.0 * np.pi * Y / pitch + 0.5 * np.pi * k
            img = np.clip(bias + amplitude * np.cos(phase), 0, 255).astype(np.uint8)
            path = os.path.join(generated_path, f"h_{pitch:g}_{k}.png")
            saveGrayPng(img, path)
            paths.append(path)
    print(f"已生成 {len(paths)} 幅条纹图(纵向周期{tuple(pitchesV)}, 横向周期{tuple(pitchesH)}, "
          f"直流={bias:g}, 幅值={amplitude:g}) -> {generated_path}")
    return paths


def setup_projector_texture(image_path):
    light = bpy.data.objects['Light']

    if light.type == 'MESH':
        if not light.data.materials:
            print("错误：Light（投影平面）未绑定材质！")
            return False
        mat = light.data.materials[0]
        if not mat.use_nodes:
            print("错误：Light的材质未启用节点编辑！")
            return False
        nodes = mat.node_tree.nodes
    elif light.type == 'LIGHT':
        if not light.data.use_nodes:
            print("错误：Light灯光未启用节点编辑！")
            return False
        nodes = light.data.node_tree.nodes
    else:
        print(f"错误：不支持的Light类型 {light.type}，仅支持MESH/LIGHT")
        return False

    bpy.context.view_layer.objects.active = light

    emission_node = None
    texture_node = None

    for node in nodes:
        if node.type == "EMISSION":
            emission_node = node
        elif node.type == "TEX_IMAGE":
            texture_node = node
            break

    if texture_node is None:
        print("未找到图像纹理节点")
        return False

    try:
        image_name = f"fringe_pattern_{os.path.basename(image_path)}"
        # 先清空纹理引用再删除旧图像数据, 避免引用悬空
        texture_node.image = None
        if image_name in bpy.data.images:
            bpy.data.images.remove(bpy.data.images[image_name])
        image = bpy.data.images.load(image_path)
        image.name = image_name
        # 条纹图为线性灰度数据，Non-Color避免sRGB gamma失真引入相位误差
        try:
            image.colorspace_settings.name = 'Non-Color'
        except Exception:
            pass
        texture_node.image = image
        # 触发依赖图更新, 确保Cycles检测到纹理变化(use_persistent_data下尤其重要)
        bpy.context.view_layer.update()
        return True
    except Exception as e:
        print(f"加载图像失败: {image_path}, 错误: {e}")
        return False


def set_active_camera(camera_obj):
    bpy.context.scene.camera = camera_obj
    bpy.context.view_layer.update()


def buildFringeRenderTasks(fringe_images, save_path, pose_index, camera_side, camera_obj, board_pose):
    """构建渲染任务列表（不立即渲染），由异步队列消费"""
    tasks = []
    for i, image_path in enumerate(fringe_images):
        if not os.path.exists(image_path):
            print(f"图像文件不存在: {image_path}")
            continue
        tasks.append({
            'image_path': image_path,
            'filepath': os.path.join(save_path, f"pose{pose_index}_{camera_side}_img{i}.bmp"),
            'camera': camera_obj,
            'board_pose': board_pose,
            'label': f"pose{pose_index}_{camera_side}_img{i}",
        })
    return tasks


def define_board_poses():
    """标定板仅1个位姿，渲染单组条纹图"""
    return [
        {"rotation": (0, 0, 0), "location": (0, 0, 0)},
    ]


def set_board_pose(board, pose):
    board.rotation_euler = pose["rotation"]
    board.location = pose["location"]
    bpy.context.view_layer.update()


def collectRenderTasks(board_poses, fringe_images):
    """收集所有位姿下的双目渲染任务（不触发渲染）"""
    tasks = []
    for pose_idx, pose in enumerate(board_poses):
        pose_folder = os.path.join(base_path, f"pose_{pose_idx}")
        if not os.path.exists(pose_folder):
            os.makedirs(pose_folder)
        tasks.extend(buildFringeRenderTasks(fringe_images, pose_folder, pose_idx, "left", camera_left, pose))
        tasks.extend(buildFringeRenderTasks(fringe_images, pose_folder, pose_idx, "right", camera_right, pose))
    return tasks


def _printConsoleProgressBar(completed, total, label="", width=32):
    """控制台原地刷新进度条（单行，用空格清空行尾残留）"""
    if total <= 0:
        return
    ratio = completed / total
    filled = int(width * ratio)
    if filled < width:
        bar = '=' * filled + '>' + ' ' * (width - filled - 1)
    else:
        bar = '=' * width
    line = f"\r[{bar}] {completed}/{total} ({ratio * 100.0:5.1f}%)"
    if label:
        line += f"  {label}"
    sys.stdout.write(line.ljust(110))
    sys.stdout.flush()


def _finishRender(cancelled=False):
    """清理handler/进度条；中止时保存剩余任务供UI"继续渲染"恢复"""
    global allRendersDoneCallback, renderCancelled, currentFrameCancelled
    global pendingRenderTasks, pendingDoneCallback, isCancelledIdle, isPaused
    if _onRenderCompleteHandler in bpy.app.handlers.render_complete:
        bpy.app.handlers.render_complete.remove(_onRenderCompleteHandler)
    if _onRenderCancelHandler in bpy.app.handlers.render_cancel:
        bpy.app.handlers.render_cancel.remove(_onRenderCancelHandler)
    try:
        bpy.context.window_manager.progress_end()
    except Exception:
        pass
    sys.stdout.write("\n")
    if cancelled:
        pendingRenderTasks = list(renderQueue)
        if currentFrameCancelled and currentRenderTask is not None:
            pendingRenderTasks.insert(0, currentRenderTask)
            currentRenderTask = None
        pendingDoneCallback = allRendersDoneCallback
        isCancelledIdle = True
        print(f"渲染已中止: 已完成 {completedRenderCount}/{totalRenderCount}，"
              f"剩余 {len(pendingRenderTasks)} 张，可在3D视图侧边栏点击\"继续渲染\"恢复")
    else:
        pendingRenderTasks = []
        pendingDoneCallback = None
        # 等待后台后处理队列清空, 确保所有BMP写完后再触发完成回调
        _postQueue.join()
        print(f"全部渲染完成: {completedRenderCount}/{totalRenderCount}")
    sys.stdout.flush()
    cb = allRendersDoneCallback
    allRendersDoneCallback = None
    renderCancelled = False
    currentFrameCancelled = False
    isPaused = False
    if cb is not None and not cancelled:
        cb()
    _updateUI()


def _writeGrayBMP(gray, filepath):
    """写8bit灰度BMP(256色灰度调色板, bottom-up行序与image.pixels一致)"""
    h, w = gray.shape
    pad = (4 - (w % 4)) % 4
    rowSize = w + pad
    paletteSize = 256 * 4
    imageSize = rowSize * h
    pixelOffset = 14 + 40 + paletteSize
    fileSize = pixelOffset + imageSize

    hdr = bytearray()
    hdr += b'BM'
    hdr += fileSize.to_bytes(4, 'little')
    hdr += (0).to_bytes(4, 'little')
    hdr += pixelOffset.to_bytes(4, 'little')
    hdr += (40).to_bytes(4, 'little')
    hdr += w.to_bytes(4, 'little')
    hdr += h.to_bytes(4, 'little')
    hdr += (1).to_bytes(2, 'little')
    hdr += (8).to_bytes(2, 'little')
    hdr += (0).to_bytes(4, 'little')
    hdr += imageSize.to_bytes(4, 'little')
    hdr += (2835).to_bytes(4, 'little')
    hdr += (2835).to_bytes(4, 'little')
    hdr += (256).to_bytes(4, 'little')
    hdr += (0).to_bytes(4, 'little')

    palette = _getBmpPalette()

    if pad > 0:
        padded = np.empty((h, rowSize), dtype=np.uint8)
        padded[:, :w] = gray
        padded[:, w:] = 0
        pixelData = padded.tobytes()
    else:
        pixelData = gray.tobytes()

    with open(filepath, 'wb') as f:
        f.write(hdr)
        f.write(palette)
        f.write(pixelData)


def _saveRenderResultAsLinearBMP(filepath):
    """从渲染结果读取线性浮点像素，手动写8bit灰度BMP
    绕过Blender的sRGB显示编码，保证BMP像素值与线性光强成正比(无gamma畸变)
    """
    renderImg = bpy.data.images.get('Render Result')
    if renderImg is None:
        return False
    w = renderImg.size[0]
    h = renderImg.size[1]
    if w <= 0 or h <= 0:
        return False
    try:
        renderImg.update()
        pixels = np.empty(w * h * 4, dtype=np.float32)
        renderImg.pixels.foreach_get(pixels)
    except Exception as e:
        print(f"读取渲染结果像素失败: {e}")
        return False
    pixels = pixels.reshape(h, w, 4)
    _enqueuePostProcess(pixels, filepath)
    return True


# ============ 后处理并行化 ============
# 主线程只做bpy强制的操作(加载EXR/拷贝像素, bpy API必须主线程),
# 量化+写BMP交给常驻后台线程, 与下一帧渲染并行执行
_postQueue = queue.Queue()
_postWorkerThread = None
_bmpPaletteCache = None


def _getBmpPalette():
    """256级灰度BMP调色板(缓存, 只构建一次)"""
    global _bmpPaletteCache
    if _bmpPaletteCache is None:
        idx = np.arange(256, dtype=np.uint8)
        pal = np.empty((256, 4), dtype=np.uint8)
        pal[:, 0] = idx
        pal[:, 1] = idx
        pal[:, 2] = idx
        pal[:, 3] = 0
        _bmpPaletteCache = pal.tobytes()
    return _bmpPaletteCache


def _postProcessWorker():
    """常驻后台线程: 线性RGBA像素量化+写8bit灰度BMP(纯numpy/文件IO)"""
    while True:
        job = _postQueue.get()
        if job is None:
            _postQueue.task_done()
            break
        pixels, bmpPath = job
        try:
            # 渲染场景为灰度，RGB近似相等，取均值更稳健
            gray_f = (pixels[..., 0] + pixels[..., 1] + pixels[..., 2]) / 3.0
            np.clip(gray_f, 0.0, 1.0, out=gray_f)
            # 线性值直接量化到8bit，不经过sRGB OETF
            gray8 = (gray_f * 255.0 + 0.5).astype(np.uint8)
            # image.pixels是bottom-up, BMP正biHeight也是bottom-up, 直接写
            _writeGrayBMP(gray8, bmpPath)
            print(f"已保存: {bmpPath}")
        except Exception as e:
            print(f"警告: 线性BMP写入失败: {bmpPath}, 错误: {e}")
        _postQueue.task_done()


def _ensurePostWorker():
    """确保后台后处理线程已启动(daemon常驻, 只启动一次)"""
    global _postWorkerThread
    if _postWorkerThread is None or not _postWorkerThread.is_alive():
        _postWorkerThread = threading.Thread(target=_postProcessWorker, daemon=True)
        _postWorkerThread.start()


def _enqueuePostProcess(pixels, bmpPath):
    """把线性RGBA像素交给后台线程量化写BMP(与下一帧渲染并行)"""
    _ensurePostWorker()
    _postQueue.put((pixels, bmpPath))


def _loadExrPixels(exrPath):
    """主线程: 加载EXR并拷贝线性像素到numpy(bpy API必须在主线程调用)"""
    try:
        img = bpy.data.images.load(exrPath, check_existing=False)
    except Exception as e:
        print(f"读取EXR失败: {exrPath}, 错误: {e}")
        return None
    try:
        try:
            img.colorspace_settings.name = 'Linear'
        except Exception:
            pass
        w = img.size[0]
        h = img.size[1]
        if w <= 0 or h <= 0:
            print(f"EXR尺寸无效: {exrPath}")
            return None
        pixels = np.empty(w * h * 4, dtype=np.float32)
        img.pixels.foreach_get(pixels)
        return pixels.reshape(h, w, 4)
    except Exception as e:
        print(f"读取EXR像素失败: {exrPath}, 错误: {e}")
        return None
    finally:
        try:
            bpy.data.images.remove(img)
        except Exception:
            pass


def _readRenderResultPixels():
    """回退: 从Render Result读取线性像素(部分Blender版本可用)"""
    renderImg = bpy.data.images.get('Render Result')
    if renderImg is None:
        return None
    w = renderImg.size[0]
    h = renderImg.size[1]
    if w <= 0 or h <= 0:
        return None
    try:
        pixels = np.empty(w * h * 4, dtype=np.float32)
        renderImg.pixels.foreach_get(pixels)
        return pixels.reshape(h, w, 4)
    except Exception:
        return None


def _executeRenderTimer():
    """在timer上下文中执行当前帧渲染（写线性EXR，完成后由handler转8bit灰度BMP）"""
    bpy.ops.render.render(write_still=True)
    return None


def _prepareAndScheduleRender():
    """准备下一帧参数并通过timer调度渲染；暂停则挂起，队列空或已取消则收尾"""
    global renderQueue, currentRenderTask, allRendersDoneCallback, renderCancelled, isPaused
    if renderCancelled:
        _finishRender(cancelled=True)
        return
    if isPaused:
        return
    while renderQueue:
        currentRenderTask = renderQueue.pop(0)
        set_board_pose(calibration_board, currentRenderTask['board_pose'])
        set_active_camera(currentRenderTask['camera'])
        if not setup_projector_texture(currentRenderTask['image_path']):
            _printConsoleProgressBar(completedRenderCount, totalRenderCount,
                                     "跳过(纹理加载失败): " + currentRenderTask['label'])
            continue
        # 渲染写线性EXR(无扩展名基名, Blender按EXR格式自动加.exr), 完成后转8bit灰度BMP
        tmpExr = os.path.splitext(currentRenderTask['filepath'])[0] + '.exr'
        currentRenderTask['tmp_exr'] = tmpExr
        bpy.context.scene.render.filepath = os.path.splitext(currentRenderTask['filepath'])[0]
        _printConsoleProgressBar(completedRenderCount, totalRenderCount,
                                 "渲染中: " + currentRenderTask['label'])
        bpy.app.timers.register(_executeRenderTimer, first_interval=0.0)
        return
    _finishRender(cancelled=False)


def _onRenderCompleteHandler(scene):
    """render_complete 回调：主线程只拷贝线性像素并删EXR，量化/写BMP交后台线程并行，立即调度下一帧"""
    global completedRenderCount, currentRenderTask
    if currentRenderTask is not None:
        bmpPath = currentRenderTask['filepath']
        tmpExr = currentRenderTask.get('tmp_exr')
        pixels = None
        if tmpExr and os.path.exists(tmpExr):
            pixels = _loadExrPixels(tmpExr)
            if pixels is None:
                pixels = _readRenderResultPixels()
            try:
                os.remove(tmpExr)
            except Exception:
                pass
        else:
            pixels = _readRenderResultPixels()
        if pixels is not None:
            _enqueuePostProcess(pixels, bmpPath)
        else:
            print(f"警告: 线性BMP写入失败(无像素数据): {bmpPath}")
    completedRenderCount += 1
    lbl = ""
    if currentRenderTask:
        lbl = "完成: " + currentRenderTask['label']
    _printConsoleProgressBar(completedRenderCount, totalRenderCount, lbl)
    try:
        wm = bpy.context.window_manager
        wm.progress_update(completedRenderCount)
    except Exception:
        pass
    bpy.app.timers.register(_prepareAndScheduleRender, first_interval=0.0)


def _onRenderCancelHandler(scene):
    """用户在渲染窗口按Esc取消时触发：置标志并调度收尾（当前帧未保存，恢复时重做）"""
    global renderCancelled, currentFrameCancelled
    renderCancelled = True
    currentFrameCancelled = True
    sys.stdout.write("\n")
    print("检测到取消信号（Esc），停止渲染队列...")
    sys.stdout.flush()
    bpy.app.timers.register(_prepareAndScheduleRender, first_interval=0.0)


def startAsyncRender(tasks, onAllDone=None):
    """启动异步渲染队列，帧间通过timer让出主循环刷新UI；渲染窗口按Esc可取消"""
    global renderQueue, totalRenderCount, completedRenderCount, allRendersDoneCallback, renderCancelled
    global isPaused, currentFrameCancelled, isCancelledIdle, pendingRenderTasks
    renderQueue = list(tasks)
    totalRenderCount = len(renderQueue)
    completedRenderCount = 0
    allRendersDoneCallback = onAllDone
    renderCancelled = False
    isPaused = False
    currentFrameCancelled = False
    isCancelledIdle = False
    pendingRenderTasks = []
    if _onRenderCompleteHandler not in bpy.app.handlers.render_complete:
        bpy.app.handlers.render_complete.append(_onRenderCompleteHandler)
    if _onRenderCancelHandler not in bpy.app.handlers.render_cancel:
        bpy.app.handlers.render_cancel.append(_onRenderCancelHandler)
    try:
        wm = bpy.context.window_manager
        wm.progress_begin(0, totalRenderCount)
    except Exception:
        pass
    print(f"\n开始渲染：共 {totalRenderCount} 张图像。")
    print("控制：3D视图侧边栏(Deflectometry)面板可暂停/继续/中止；渲染窗口按 Esc 中止队列。")
    _printConsoleProgressBar(0, totalRenderCount)
    sys.stdout.flush()
    _updateUI()
    bpy.app.timers.register(_prepareAndScheduleRender, first_interval=0.0)


def _updateUI():
    """刷新3D视图区域，让侧边栏面板立即反映最新进度/状态"""
    try:
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
    except Exception:
        pass


def poll_board_object(self, obj):
    """被测物体选择器只显示网格物体。"""
    return obj is not None and obj.type == 'MESH'


class DeflBoardEntry(bpy.types.PropertyGroup):
    obj: bpy.props.PointerProperty(name="物体", type=bpy.types.Object, poll=poll_board_object)


def board_enum_items(self, context):
    """下拉框内容 = 用户手动维护的候选列表"""
    lst = getattr(context.scene, "defl_boards", None)
    if not lst or len(lst) == 0:
        return [('-1', "(列表为空, 请先用下方按钮添加)", "")]
    out = []
    for i in range(len(lst)):
        o = lst[i].obj
        out.append((str(i), o.name if o else "(物体已删除)", ""))
    return out


def get_selected_board(context):
    """返回下拉框当前选中的被测物体; 列表未注册或为空时返回 None"""
    lst = getattr(context.scene, "defl_boards", None)
    if not lst:
        return None
    try:
        i = int(getattr(context.scene, "defl_board_object", "-1"))
    except (ValueError, TypeError):
        return None
    if 0 <= i < len(lst):
        return lst[i].obj
    return None


def apply_board_visibility(context):
    """只隐藏候选列表里未被选中的物体; 未加入列表的物体一概不动

    作用: 切换下拉框后, 列表中的其他候选被测物在视图与渲染中隐藏,
          便于查看当前选中的那一个。
    """
    scene = context.scene
    cur = get_selected_board(context)
    lst = getattr(scene, "defl_boards", None)
    if not lst or cur is None:
        return
    for e in lst:
        if e.obj:
            hide = (e.obj is not cur)
            e.obj.hide_viewport = hide
            e.obj.hide_render = hide


def on_board_changed(self, context):
    """下拉框切换被测物体时立即刷新可见性"""
    apply_board_visibility(context)


def registerSceneProps():
    """注册UI配置属性（投图模式/条纹目录/显示屏分辨率/3频组合）"""
    bpy.types.Scene.defl_pattern_mode = bpy.props.EnumProperty(
        name="投图模式",
        description="条纹图来源：文件目录加载 或 程序生成(3频4步相移)",
        items=[('FILE', "文件目录加载", "从指定目录加载所有条纹图依次投出渲染"),
               ('GENERATE', "程序生成(3频4步)", "按显示屏分辨率与3频组合生成12幅纵+12幅横条纹")],
        default='FILE')
    bpy.types.Scene.defl_pattern_dir = bpy.props.StringProperty(
        name="条纹图目录", description="文件模式下加载条纹图的目录",
        subtype='DIR_PATH', default=pattern_path)
    bpy.types.Scene.defl_proj_width = bpy.props.IntProperty(
        name="显示屏宽(px)", description="生成模式下条纹图宽度",
        default=1280, min=64, max=8192)
    bpy.types.Scene.defl_proj_height = bpy.props.IntProperty(
        name="显示屏高(px)", description="生成模式下条纹图高度",
        default=800, min=64, max=8192)
    bpy.types.Scene.defl_freq_group = bpy.props.EnumProperty(
        name="纵向3频组合", description="纵向条纹(沿屏宽)的3频4步周期组合(N1,N2,N3), P123须>=屏宽",
        items=freqGroupItems)
    bpy.types.Scene.defl_freq_group_h = bpy.props.EnumProperty(
        name="横向3频组合", description="横向条纹(沿屏高)的3频4步周期组合(N1,N2,N3), P123须>=屏高(可比纵向更细)",
        items=freqGroupItemsH)
    bpy.types.Scene.defl_fringe_bias = bpy.props.FloatProperty(
        name="直流分量", description="条纹平均灰度(直流分量), 灰度=直流+幅值·cos(相位)",
        default=120.0, min=0.0, max=255.0)
    bpy.types.Scene.defl_fringe_amplitude = bpy.props.FloatProperty(
        name="条纹幅值", description="条纹幅值(对比度). 需满足 直流+幅值<=255 且 直流-幅值>=0, 否则会过曝截断影响解相",
        default=100.0, min=1.0, max=255.0)
    bpy.types.Scene.defl_file_pitches = bpy.props.StringProperty(
        name="纵向周期(px)", description="文件模式下纵向条纹(沿屏宽)的3个像素周期(须与图案一致)",
        default="13,14,15")
    bpy.types.Scene.defl_file_pitches_h = bpy.props.StringProperty(
        name="横向周期(px)", description="文件模式下横向条纹(沿屏高)的3个像素周期(须与图案一致)",
        default="20,22,24")
    bpy.types.Scene.defl_file_offset = bpy.props.FloatProperty(
        name="条纹偏移(px)", description="文件模式下图案的零位偏移",
        default=20.0, min=0.0, max=1000.0)
    bpy.types.Scene.defl_base_path = bpy.props.StringProperty(
        name="渲染输出目录", description="渲染条纹图的保存目录(自动记忆到配置文件)",
        subtype='DIR_PATH', default=base_path)
    bpy.types.Scene.defl_boards = bpy.props.CollectionProperty(type=DeflBoardEntry)
    bpy.types.Scene.defl_board_object = bpy.props.EnumProperty(
        name="被测物体",
        description="要重建的被测物体; 切换后场景中只显示它, 其余网格物体隐藏",
        items=board_enum_items,
        update=on_board_changed,
    )


def unregisterSceneProps():
    for name in ("defl_pattern_mode", "defl_pattern_dir", "defl_proj_width",
                 "defl_proj_height", "defl_freq_group", "defl_freq_group_h",
                 "defl_fringe_bias", "defl_fringe_amplitude",
                 "defl_file_pitches", "defl_file_pitches_h", "defl_file_offset", "defl_base_path",
                 "defl_board_object", "defl_boards"):
        try:
            delattr(bpy.types.Scene, name)
        except Exception:
            pass


class DEFLECTOMETRY_PT_render_config(bpy.types.Panel):
    bl_label = "渲染配置"
    bl_idname = "VIEW3D_PT_deflectometry_render_config"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Deflectometry"

    def draw(self, context):
        scene = context.scene
        layout = self.layout
        col = layout.column()
        col.prop(scene, "defl_board_object")
        col.operator("deflectometry.board_add", text="添加全部被测物体", icon='ADD')
        col.operator("deflectometry.board_del", text="删除", icon='REMOVE')
        col.separator()
        col.prop(scene, "defl_pattern_mode")
        if scene.defl_pattern_mode == 'FILE':
            col.prop(scene, "defl_pattern_dir")
            col.prop(scene, "defl_file_pitches")
            col.prop(scene, "defl_file_pitches_h")
            col.prop(scene, "defl_file_offset")
            # 文件模式也需显示屏分辨率(用于导出像素尺寸/分辨率, 须与data/图案尺寸一致)
            row = col.row(align=True)
            row.prop(scene, "defl_proj_width")
            row.prop(scene, "defl_proj_height")
            col.label(text=f"显示屏分辨率: {scene.defl_proj_width}x{scene.defl_proj_height} (须与投图图案尺寸一致)")
        else:
            row = col.row(align=True)
            row.prop(scene, "defl_proj_width")
            row.prop(scene, "defl_proj_height")
            col.prop(scene, "defl_freq_group")
            col.prop(scene, "defl_freq_group_h")
            row = col.row(align=True)
            row.prop(scene, "defl_fringe_bias")
            row.prop(scene, "defl_fringe_amplitude")
            # 灰度范围与过曝提示
            bias = scene.defl_fringe_bias
            amp = scene.defl_fringe_amplitude
            gMin = bias - amp
            gMax = bias + amp
            col.label(text=f"灰度范围: [{gMin:.0f}, {gMax:.0f}]")
            if gMax > 255 or gMin < 0:
                col.label(text="警告: 灰度超出[0,255]会过曝截断, 请调低幅值!", icon='ERROR')
            elif gMax > 250:
                col.label(text="提示: 峰值>250会被重建端过曝阈值剔除, 建议降低", icon='INFO')
            col.label(text=f"投图: 12纵+12横 共24幅 ({scene.defl_proj_width}x{scene.defl_proj_height})")
        col.separator()
        col.prop(scene, "defl_base_path")
        col.separator()
        col.operator("deflectometry.start_render", icon='RENDER_STILL')
        col.operator("deflectometry.export_calib", icon='FILE_TEXT')
        col.operator("deflectometry.rotate_screen_frame", icon='FILE_REFRESH')


class DEFLECTOMETRY_OT_board_add(bpy.types.Operator):
    bl_idname = "deflectometry.board_add"
    bl_label = "添加选中物体"
    bl_description = "把当前选中的网格物体加入被测物体候选列表"

    def execute(self, context):
        objs = [o for o in context.selected_objects if o.type == 'MESH']
        if not objs:
            self.report({'WARNING'}, "请先在3D视图中选中一个网格物体")
            return {'CANCELLED'}
        lst = context.scene.defl_boards
        added = 0
        for o in objs:
            if any(e.obj is o for e in lst):
                continue
            e = lst.add()
            e.obj = o
            added += 1
        context.scene.defl_board_object = str(len(lst) - 1)
        self.report({'INFO'}, "已添加 %d 个物体, 当前列表 %d 项" % (added, len(lst)))
        return {'FINISHED'}


class DEFLECTOMETRY_OT_board_del(bpy.types.Operator):
    bl_idname = "deflectometry.board_del"
    bl_label = "删除"
    bl_description = "从候选列表中移除当前项(只从列表移除, 不删除场景中的物体)"

    def execute(self, context):
        lst = context.scene.defl_boards
        try:
            i = int(context.scene.defl_board_object)
        except (ValueError, TypeError):
            return {'CANCELLED'}
        if not (0 <= i < len(lst)):
            return {'CANCELLED'}
        lst.remove(i)
        context.scene.defl_board_object = (str(max(0, min(i, len(lst) - 1)))
                                           if len(lst) > 0 else '-1')
        return {'FINISHED'}


class DEFLECTOMETRY_OT_export_calib(bpy.types.Operator):
    bl_idname = "deflectometry.export_calib"
    bl_label = "导出标定参数"
    bl_description = "从Blender场景对象读取相机内外参+显示屏外参，导出cal_res.txt与cal_Projector.txt(供Reconstruct3D直接导入)"

    def execute(self, context):
        global projector, camera_left, camera_right, base_path
        base_path = bpy.path.abspath(context.scene.defl_base_path)
        saveRenderConfig()
        projector = bpy.data.objects.get("Light") or projector
        camera_left = bpy.data.objects.get(LEFT_CAMERA_NAME) or camera_left
        camera_right = bpy.data.objects.get(RIGHT_CAMERA_NAME) or camera_right
        if not all([projector, camera_left, camera_right]):
            self.report({'ERROR'}, "场景缺少必要对象(双相机/Light)")
            return {'CANCELLED'}
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        ok = export_calibration_params(base_path)
        if ok:
            self.report({'INFO'}, f"标定参数已导出至: {base_path}")
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, "标定参数导出失败, 请查看控制台")
            return {'CANCELLED'}


class DEFLECTOMETRY_OT_rotate_screen_frame(bpy.types.Operator):
    bl_idname = "deflectometry.rotate_screen_frame"
    bl_label = "C_Screen绕Z转90°"
    bl_description = "每点一次, 让 C_Screen 坐标系绕其自身局部 Z 轴旋转 90°(用于快速试出正确的屏面X/Y朝向)"

    def execute(self, context):
        frame = bpy.data.objects.get(SCREEN_FRAME_NAME)
        if frame is None:
            self.report({'ERROR'}, f"未找到坐标系对象 '{SCREEN_FRAME_NAME}'")
            return {'CANCELLED'}
        # 绕自身局部Z轴转90°: 世界矩阵右乘局部Z旋转(保持原点不变)
        rotZ = mathutils.Matrix.Rotation(math.radians(90.0), 4, 'Z')
        frame.matrix_world = frame.matrix_world @ rotZ
        self.report({'INFO'}, f"{SCREEN_FRAME_NAME} 已绕自身Z轴旋转90°")
        return {'FINISHED'}


class DEFLECTOMETRY_OT_start_render(bpy.types.Operator):
    bl_idname = "deflectometry.start_render"
    bl_label = "开始渲染"
    bl_description = "按当前配置采集条纹图：标定板1个位姿，双目各渲染一遍"

    def execute(self, context):
        global calibration_board, projector, camera_left, camera_right, base_path
        base_path = bpy.path.abspath(context.scene.defl_base_path)
        saveRenderConfig()
        calibration_board = get_selected_board(context)
        if calibration_board is None:
            self.report({'ERROR'}, "未选择被测物体: 请先在面板的列表中添加并选择一个被测物体")
            return {'CANCELLED'}
        # 只渲染当前选中的被测物(其余网格物体保持隐藏)
        apply_board_visibility(context)
        projector = bpy.data.objects.get("Light") or projector
        camera_left = bpy.data.objects.get(LEFT_CAMERA_NAME) or camera_left
        camera_right = bpy.data.objects.get(RIGHT_CAMERA_NAME) or camera_right
        if not all([calibration_board, projector, camera_left, camera_right]):
            self.report({'ERROR'}, "场景缺少必要对象(被测物体/Light/双相机)")
            return {'CANCELLED'}
        # 渲染前按实际相机对象摆正 C_LCam/C_RCam 标定坐标系(OpenCV相机系)
        align_calib_frames()
        running = (totalRenderCount > 0 and completedRenderCount < totalRenderCount
                   and not isPaused and not isCancelledIdle)
        if running:
            self.report({'WARNING'}, "渲染进行中，请先暂停或中止")
            return {'CANCELLED'}
        scene = context.scene
        if scene.defl_pattern_mode == 'FILE':
            patternDir = bpy.path.abspath(scene.defl_pattern_dir)
            if not patternDir or not os.path.isdir(patternDir):
                self.report({'ERROR'}, f"条纹图目录无效: {patternDir}")
                return {'CANCELLED'}
            try:
                filePitchesV = [float(x) for x in scene.defl_file_pitches.replace('，', ',').split(',') if x.strip()]
                filePitchesH = [float(x) for x in scene.defl_file_pitches_h.replace('，', ',').split(',') if x.strip()]
            except Exception:
                self.report({'ERROR'}, "条纹周期解析失败，格式如: 13,14,15")
                return {'CANCELLED'}
            if len(filePitchesV) < 3 or len(filePitchesH) < 3:
                self.report({'ERROR'}, "纵向/横向条纹周期各需要3个值(逗号分隔)")
                return {'CANCELLED'}
            # 多频外差: 纵向P123须>=屏宽, 横向P123须>=屏高, 否则该方向绝对相位有整周期歧义
            p123v = fringeSynthWavelength(tuple(filePitchesV[:3]))
            p123h = fringeSynthWavelength(tuple(filePitchesH[:3]))
            if p123v < scene.defl_proj_width:
                self.report({'ERROR'},
                            f"纵向周期{tuple(filePitchesV[:3])}的P123={p123v:.0f}px < 屏宽{scene.defl_proj_width}px, 会有歧义")
                return {'CANCELLED'}
            if p123h < scene.defl_proj_height:
                self.report({'ERROR'},
                            f"横向周期{tuple(filePitchesH[:3])}的P123={p123h:.0f}px < 屏高{scene.defl_proj_height}px, 会有歧义")
                return {'CANCELLED'}
            fringeImages = get_fringe_images(patternDir)
            if not fringeImages:
                self.report({'ERROR'}, f"目录中未找到条纹图: {patternDir}")
                return {'CANCELLED'}
            PROJ_RES['w'] = scene.defl_proj_width
            PROJ_RES['h'] = scene.defl_proj_height
            lastPatternInfo['pitches'] = filePitchesV[:3]
            lastPatternInfo['pitchesH'] = filePitchesH[:3]
            lastPatternInfo['offset'] = scene.defl_file_offset
        else:
            try:
                pitchesV = tuple(float(x) for x in scene.defl_freq_group.split('_'))
                pitchesH = tuple(float(x) for x in scene.defl_freq_group_h.split('_'))
            except Exception:
                self.report({'ERROR'}, "3频组合解析失败")
                return {'CANCELLED'}
            fringeImages = generateFringePatterns(
                scene.defl_proj_width, scene.defl_proj_height, pitchesV, pitchesH,
                bias=scene.defl_fringe_bias, amplitude=scene.defl_fringe_amplitude)
            PROJ_RES['w'] = scene.defl_proj_width
            PROJ_RES['h'] = scene.defl_proj_height
            # 组合即为像素周期，写入 pattern_info 供 Reconstruct3D 使用
            lastPatternInfo['pitches'] = list(pitchesV)
            lastPatternInfo['pitchesH'] = list(pitchesH)
            lastPatternInfo['offset'] = 0.0
        setup_render_settings()
        tasks = collectRenderTasks(define_board_poses(), fringeImages)
        if not tasks:
            self.report({'ERROR'}, "没有可执行的渲染任务")
            return {'CANCELLED'}
        startAsyncRender(tasks, onAllDone=lambda: (export_calibration_params(base_path), exportPatternInfo()))
        self.report({'INFO'}, f"开始渲染: 投图{len(fringeImages)}幅 x 双目 = {len(tasks)}幅渲染")
        return {'FINISHED'}


class DEFLECTOMETRY_PT_render_progress(bpy.types.Panel):
    bl_label = "双目条纹渲染"
    bl_idname = "VIEW3D_PT_deflectometry_render_progress"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Deflectometry"

    def draw(self, context):
        layout = self.layout
        if totalRenderCount <= 0:
            layout.label(text="未开始渲染", icon='INFO')
            return
        ratio = completedRenderCount / totalRenderCount
        col = layout.column()
        col.label(text=f"进度: {completedRenderCount}/{totalRenderCount} ({ratio * 100.0:.1f}%)")
        col.progress(factor=ratio, type='BAR')
        allDone = (completedRenderCount >= totalRenderCount and not renderQueue and not isCancelledIdle)
        if isPaused:
            col.label(text="状态: 已暂停", icon='PAUSE')
        elif isCancelledIdle:
            col.label(text=f"状态: 已中止(剩余{len(pendingRenderTasks)}张)", icon='CANCEL')
        elif allDone:
            col.label(text="状态: 全部完成", icon='CHECKMARK')
        else:
            col.label(text="状态: 渲染中", icon='RENDER_ANIMATION')
        if currentRenderTask is not None and not isPaused and not isCancelledIdle and not allDone:
            col.label(text=f"当前: {currentRenderTask['label']}")
        if not allDone:
            row = col.row(align=True)
            if isPaused or isCancelledIdle:
                row.operator("deflectometry.resume_render", text="继续渲染", icon='PLAY')
            else:
                row.operator("deflectometry.pause_render", text="暂停", icon='PAUSE')
            row.operator("deflectometry.cancel_render", text="中止", icon='CANCEL')


class DEFLECTOMETRY_OT_pause_render(bpy.types.Operator):
    bl_idname = "deflectometry.pause_render"
    bl_label = "暂停渲染"
    bl_description = "当前帧渲染完成后暂停后续图像渲染（队列保留，可继续）"

    def execute(self, context):
        global isPaused
        isPaused = True
        _updateUI()
        return {'FINISHED'}


class DEFLECTOMETRY_OT_resume_render(bpy.types.Operator):
    bl_idname = "deflectometry.resume_render"
    bl_label = "继续渲染"
    bl_description = "继续执行暂停或中止后剩余的渲染队列"

    def execute(self, context):
        global isPaused, isCancelledIdle, pendingRenderTasks
        if isCancelledIdle:
            isCancelledIdle = False
            tasks = list(pendingRenderTasks)
            pendingRenderTasks = []
            if tasks:
                startAsyncRender(tasks, onAllDone=pendingDoneCallback)
            _updateUI()
            return {'FINISHED'}
        isPaused = False
        _updateUI()
        bpy.app.timers.register(_prepareAndScheduleRender, first_interval=0.0)
        return {'FINISHED'}


class DEFLECTOMETRY_OT_cancel_render(bpy.types.Operator):
    bl_idname = "deflectometry.cancel_render"
    bl_label = "中止渲染"
    bl_description = "当前帧渲染完成后中止整个队列（剩余任务可通过\"继续渲染\"恢复）"

    def execute(self, context):
        global renderCancelled
        renderCancelled = True
        bpy.app.timers.register(_prepareAndScheduleRender, first_interval=0.0)
        _updateUI()
        return {'FINISHED'}


_uiClassesRegistered = False


def registerRenderUI():
    """注册渲染进度面板与控制按钮（幂等，可重复调用）"""
    global _uiClassesRegistered
    if _uiClassesRegistered:
        return
    for cls in (DeflBoardEntry,
                DEFLECTOMETRY_PT_render_config,
                DEFLECTOMETRY_PT_render_progress,
                DEFLECTOMETRY_OT_start_render,
                DEFLECTOMETRY_OT_board_add,
                DEFLECTOMETRY_OT_board_del,
                DEFLECTOMETRY_OT_export_calib,
                DEFLECTOMETRY_OT_rotate_screen_frame,
                DEFLECTOMETRY_OT_pause_render,
                DEFLECTOMETRY_OT_resume_render,
                DEFLECTOMETRY_OT_cancel_render):
        try:
            bpy.utils.register_class(cls)
        except Exception:
            pass
    registerSceneProps()
    # 同步持久化的base_path到UI属性(重新加载脚本时UI显示记忆的路径)
    try:
        bpy.context.scene.defl_base_path = base_path
    except Exception:
        pass
    _uiClassesRegistered = True


def unregisterRenderUI():
    """注销渲染进度UI（脚本结束清理用）"""
    global _uiClassesRegistered
    if not _uiClassesRegistered:
        return
    for cls in (DEFLECTOMETRY_OT_cancel_render,
                DEFLECTOMETRY_OT_resume_render,
                DEFLECTOMETRY_OT_pause_render,
                DEFLECTOMETRY_OT_rotate_screen_frame,
                DEFLECTOMETRY_OT_export_calib,
                DEFLECTOMETRY_OT_board_del,
                DEFLECTOMETRY_OT_board_add,
                DEFLECTOMETRY_OT_start_render,
                DEFLECTOMETRY_PT_render_progress,
                DEFLECTOMETRY_PT_render_config):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
    unregisterSceneProps()
    try:
        bpy.utils.unregister_class(DeflBoardEntry)
    except Exception:
        pass
    _uiClassesRegistered = False


def blender_camera_intrinsics(cam_obj, width, height):
    """从Blender相机对象提取内参 (fx, fy, cx, cy)，单位：像素"""
    cam_data = cam_obj.data
    sensor_w = cam_data.sensor_width
    focal_length = cam_data.lens
    if cam_data.sensor_fit == 'AUTO':
        aspect = width / height
        if aspect >= 1.0:
            sensor_fit_h = sensor_w / aspect
            fx = focal_length * width / sensor_w
            fy = focal_length * height / sensor_fit_h
        else:
            sensor_fit_w = sensor_w * aspect
            fx = focal_length * width / sensor_fit_w
            fy = focal_length * height / sensor_w
    elif cam_data.sensor_fit == 'HORIZONTAL':
        fx = focal_length * width / sensor_w
        sensor_h = cam_data.sensor_height if cam_data.sensor_height > 0 else sensor_w * height / width
        fy = focal_length * height / sensor_h
    else:
        sensor_h = cam_data.sensor_height if cam_data.sensor_height > 0 else sensor_w * height / width
        fy = focal_length * height / sensor_h
        sensor_w_fit = sensor_h * width / height
        fx = focal_length * width / sensor_w_fit
    cx = width / 2.0
    cy = height / 2.0
    return fx, fy, cx, cy


def rotationVectorToMatrix(rvec):
    """旋转向量→旋转矩阵(Rodrigues公式)，纯numpy实现替代cv2.Rodrigues"""
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    k = rvec / theta
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]], dtype=np.float64)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def matrixToRotationVector(R):
    """旋转矩阵→旋转向量，纯numpy实现替代cv2.Rodrigues"""
    cosTheta = (np.trace(R) - 1.0) / 2.0
    cosTheta = max(-1.0, min(1.0, cosTheta))
    theta = np.arccos(cosTheta)
    if theta < 1e-12:
        return np.zeros(3, dtype=np.float64)
    if abs(theta - np.pi) < 1e-6:
        # 【补丁5】θ≈π 时普通公式含 1/sinθ → 数值爆炸, 必须单独处理。
        # 原实现用 k = sqrt(diag((2I-R-Rᵀ)/4)) 是错的: 该矩阵实为 c(I-kkᵀ), c=(1-cosθ)/2,
        # 于是解出的是 sqrt(1-kᵢ²) 而不是 kᵢ, 旋转轴会错 45°~90°。
        # 恒等式: (R+Rᵀ)/2 = cos²(θ/2)·I + sin²(θ/2)·kkᵀ  (k = 单位旋转轴)
        #   ⇒ 沿 k 的特征值恒为 1(最大), 其余为 cos²(θ/2)≈0, 取最大特征向量即得轴。
        # 注意必须用"对称部分"再 eigh: eigh 只读下三角, 直接喂非对称的 (R+I)/2 会得到垃圾。
        w, V = np.linalg.eigh((R + R.T) * 0.5)
        k = V[:, int(np.argmax(w))]
        # 轴的符号由反对称部分 (R−Rᵀ)/2 = sinθ·K 决定。
        # (|θ−π|<1e-6 时 sinθ≈1e-6, 远大于 float64 机器精度, 定符号仍然可靠)
        K = (R - R.T) / (2.0 * np.sin(theta))
        if float(np.dot(k, np.array([K[2, 1], K[0, 2], K[1, 0]]))) < 0.0:
            k = -k
        return theta * k
    return theta / (2.0 * np.sin(theta)) * np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ], dtype=np.float64)


def blender_object_extrinsics(obj, ref_obj=None):
    """提取obj相对于ref_obj的外参 (旋转向量om, 平移T)
    若ref_obj为None，则返回obj在世界坐标系下的位姿
    返回的外参在OpenCV坐标系下（Y朝下，Z朝前），平移单位为mm
    对于相机：Blender相机朝-Y转为OpenCV朝+Z
    对于非相机对象（如显示屏）：在ref_obj的OpenCV坐标系下表达
    Blender默认单位为m，平移分量乘以1000转为mm
    """
    # 单目
    T_b2c = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ], dtype=np.float64)
    M_obj = np.array(obj.matrix_world)
    if ref_obj is not None:
        M_ref = np.array(ref_obj.matrix_world)
        M_rel = np.linalg.inv(M_ref) @ M_obj
    else:
        M_rel = M_obj.copy()
    R_blender = M_rel[:3, :3]
    t_blender = M_rel[:3, 3]
    R_cv = T_b2c @ R_blender
    t_cv = T_b2c @ t_blender
    # 场景单位 → mm
    unit_to_mm = get_scene_unit_scale()
    t_cv = t_cv * unit_to_mm
    om = matrixToRotationVector(R_cv)
    return om, t_cv


def camera_extrinsics_cv(cam_obj, ref_cam_obj):
    """两个Blender相机对象之间的外参 (OpenCV约定, mm)

    返回满足 p_left_cv = R @ p_right_cv + T 的 (om, T):
      R,T 为"右相机相对左相机"外参, T = 左相机系中右相机光心坐标
    推导: p_w = R_l·T_b2c·p_l_cv + t_l (左), p_w = R_r·T_b2c·p_r_cv + t_r (右)
      => R = T_b2c·R_rel·T_b2c, T = T_b2c·t_rel  (R_rel,t_rel = inv(M_l)@M_r)
    渲染实际使用相机对象位姿, 故外参必须由相机对象计算(而非坐标系空对象)
    """
    # 双目
    T_b2c = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ], dtype=np.float64)
    M_l = np.array(ref_cam_obj.matrix_world)
    M_r = np.array(cam_obj.matrix_world)
    M_rel = np.linalg.inv(M_l) @ M_r
    R_rel = M_rel[:3, :3]
    t_rel = M_rel[:3, 3]
    R_cv = T_b2c @ R_rel @ T_b2c
    t_cv = T_b2c @ t_rel
    unit_to_mm = get_scene_unit_scale()
    t_cv = t_cv * unit_to_mm
    om = matrixToRotationVector(R_cv)
    return om, t_cv


def _screen_tex_nodes(mesh_obj):
    """【补丁4】取屏面材质的 (TEX_IMAGE 节点, 其 Vector 上游的 MAPPING 节点)。
    屏面实际显示的是"经材质节点变换后的贴图", 而非裸 UV 层 —— 本场景 Mapping 是
    180°绕Z + 镜像X, 等效把贴图相对 UV 层上下翻转, 必须一起考虑。"""
    if mesh_obj is None:
        return None, None
    if mesh_obj.type == 'MESH':
        if not mesh_obj.data.materials:
            return None, None
        mat = mesh_obj.data.materials[0]
        if not mat.use_nodes:
            return None, None
        nodes = mat.node_tree.nodes
    elif mesh_obj.type == 'LIGHT':
        if not mesh_obj.data.use_nodes:
            return None, None
        nodes = mesh_obj.data.node_tree.nodes
    else:
        return None, None
    tex = next((n for n in nodes if n.type == 'TEX_IMAGE'), None)
    mp = None
    if tex is not None:
        for inp in tex.inputs:
            if inp.name in ('Vector', 'UV') and inp.is_linked:
                fn = inp.links[0].from_node
                if fn.type == 'MAPPING' and getattr(fn, 'vector_type', 'POINT') == 'POINT':
                    mp = fn
                break
    return tex, mp


def _loop_image_pixels(mesh_obj, prj_w, prj_h):
    """【补丁1-3】每个 loop 的 (xS,yS) 图像像素坐标 与 世界坐标。
    已计入: 活动渲染UV层 → Mapping(POINT) 变换 → REPEAT 环绕
    贴图坐标→图像像素:  xS = u_tex*W ,  yS = (1 - v_tex)*H  (图像原点在左上角)
    返回 (X, Wm), X 的每行是 [1, xS, yS]"""
    tex, mp = _screen_tex_nodes(mesh_obj)
    ext = getattr(tex, 'extension', 'REPEAT') if tex is not None else 'REPEAT'
    mesh = mesh_obj.data
    # 注意: active_render 是"每个UV层上的布尔属性", 不是 uv_layers 集合的属性!
    #       mesh.uv_layers.active_render 会抛 AttributeError(集合只有 active / active_index)
    uvl = next((l for l in mesh.uv_layers if getattr(l, 'active_render', False)), None) \
        or mesh.uv_layers.active
    M = np.array(mesh_obj.matrix_world)
    R, t = M[:3, :3], M[:3, 3]
    rows, ws = [], []
    for poly in mesh.polygons:
        for li in poly.loop_indices:
            vi = mesh.loops[li].vertex_index
            uv = uvl.data[li].uv
            tu, tv = float(uv.x), float(uv.y)
            if mp is not None:
                loc = np.array([float(v) for v in mp.inputs['Location'].default_value])
                rot = np.array([float(v) for v in mp.inputs['Rotation'].default_value])
                scl = np.array([float(v) for v in mp.inputs['Scale'].default_value])
                v3 = np.array([tu, tv, 0.0]) * scl          # 缩放
                cz, sz = math.cos(rot[2]), math.sin(rot[2])
                v3 = np.array([cz * v3[0] - sz * v3[1],     # 绕Z旋转
                               sz * v3[0] + cz * v3[1], v3[2]]) + loc
                tu, tv = float(v3[0]), float(v3[1])
            rows.append([1.0, tu * prj_w, (1.0 - tv) * prj_h])
            ws.append(R @ np.array(mesh.vertices[vi].co) + t)
    X = np.array(rows, dtype=np.float64)
    # REPEAT 环绕等价于沿 xS/yS 平移"整数个图像宽/高", 所以不要用 frac() 取模:
    # frac() 在整数边界处不连续, 而网格顶点恰好落在边界上 —— 本场景 Mapping 是
    # 180°绕Z + 镜像X(即 tv = -v), 四个角点正好是 frac(-v) 的分支点, 会被判到贴图另一端,
    # 导致设计矩阵退化(两行重合) → 最小二乘解爆炸。
    # 改为按整数倍平移, 把拟合窗口对齐到图像 [0,W]×[0,H]。
    if ext == 'REPEAT':
        X[:, 1] += round(-X[:, 1].min() / prj_w) * prj_w
        X[:, 2] += round(-X[:, 2].min() / prj_h) * prj_h
    return X, np.array(ws)


def screen_frame_from_uv(light_obj, cam_left):
    """从屏面网格UV拟合显示屏外参 (左相机OpenCV系, mm)

    屏坐标系约定(与Reconstruct3D displayTo3D配套, 用户自定义):
      原点 = 屏左上角(相机视角下图像(0,0)像素处)
      +X = 像素xS+(图像向右), +Y = 像素yS+(图像向下), Z = X×Y(右手闭合)
    【补丁1-4】由 _loop_image_pixels 在"图像像素空间"最小二乘拟合屏面平面, 不依赖C_Screen空对象:
      每个网格 loop 先按 活动UV层 → 贴图Mapping变换 → REPEAT环绕 换算成图像像素(xS,yS),
      再由 [1,xS,yS] → 世界坐标 拟合。因此贴图若被翻转/旋转, 屏坐标系会与渲染实际显示保持一致。
    返回 (om, T); 失败返回 None
    """
    mesh = light_obj.data
    if light_obj.type != 'MESH' or not mesh.uv_layers or not mesh.uv_layers.active:
        print("错误: 屏面对象必须是带UV贴图的网格, 无法按UV定位屏坐标系")
        return None
    X, Wm = _loop_image_pixels(light_obj, PROJ_RES['w'], PROJ_RES['h'])
    if len(X) < 4:
        print("错误: 屏面网格顶点过少, 无法拟合屏坐标系")
        return None
    theta, *_ = np.linalg.lstsq(X, Wm, rcond=None)
    # theta 的"行"才是基向量: W = theta[0] + xS·theta[1] + yS·theta[2]
    # (原实现取 theta[:,0]/[:,1]/[:,2] 是错的: 那会把世界 X/Y/Z 分量混在一起)
    origin_w = theta[0]                     # 图像(0,0) = 屏左上角
    dpx, dpy = theta[1], theta[2]           # ∂W/∂xS, ∂W/∂yS (内部单位/像素)
    U_w = dpx / np.linalg.norm(dpx)         # +X: 图像右
    V_w = dpy / np.linalg.norm(dpy)         # +Y: 图像下(已含贴图翻转)
    B_w = np.cross(U_w, V_w)                # +Z: 右手闭合(X×Y)
    B_w = B_w / np.linalg.norm(B_w)
    U_o = U_w - V_w * np.dot(U_w, V_w) - B_w * np.dot(U_w, B_w)
    U_o = U_o / np.linalg.norm(U_o)         # UV斜切时正交修正
    V_o = np.cross(B_w, U_o)
    V_o = V_o / np.linalg.norm(V_o)
    R_w = np.column_stack([U_o, V_o, B_w])  # 屏局部(用户约定) → 世界
    # 左相机对象: Blender相机局部(-Y视线) → OpenCV(+Z视线), 逆相机即 T_b2c(自逆)
    T_b2c = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ], dtype=np.float64)
    M_cam = np.array(cam_left.matrix_world)
    R_c2w = M_cam[:3, :3] @ T_b2c           # 左相机OpenCV系 → 世界
    t_cam = M_cam[:3, 3]                    # 左相机光心(世界)
    R = R_c2w.T @ R_w
    t = R_c2w.T @ (origin_w - t_cam)
    unit_to_mm = get_scene_unit_scale()
    t = t * unit_to_mm
    om = matrixToRotationVector(R)
    return om, t


def screen_world_frame_from_uv(mesh_obj):
    """【补丁1-4】由屏面网格 + 实际贴图映射, 在"图像像素空间"拟合屏坐标系(几何精确)。
    返回 (origin_w, R_w, width_mm, height_mm):
      origin_w = 图像(0,0)像素处(屏左上角)的世界坐标
      R_w      = 列 [X=图像右, Y=图像下, Z=X×Y]
      width_mm/height_mm = 图像区域在屏上的物理尺寸(= |∂W/∂xS|·W)
    失败返回 None。"""
    if mesh_obj is None or getattr(mesh_obj, "type", None) != 'MESH':
        return None
    mesh = mesh_obj.data
    if not mesh.uv_layers or not mesh.uv_layers.active:
        return None
    X, Wm = _loop_image_pixels(mesh_obj, PROJ_RES['w'], PROJ_RES['h'])
    if len(X) < 4:
        return None
    theta, *_ = np.linalg.lstsq(X, Wm, rcond=None)
    origin_w = theta[0]                       # 图像(0,0)
    dpx = theta[1]                            # ∂W/∂xS (内部单位/像素)
    dpy = theta[2]                            # ∂W/∂yS (内部单位/像素)
    U_w = dpx / np.linalg.norm(dpx)           # +X 图像右
    V_w = dpy / np.linalg.norm(dpy)           # +Y 图像下(已含贴图翻转)
    B_w = np.cross(U_w, V_w)
    B_w = B_w / np.linalg.norm(B_w)           # +Z 法向
    # 正交化(应对UV轻微斜切), 保证 R_w 为正交阵
    U_o = U_w - V_w * np.dot(U_w, V_w) - B_w * np.dot(U_w, B_w)
    U_o = U_o / np.linalg.norm(U_o)
    V_o = np.cross(B_w, U_o)
    V_o = V_o / np.linalg.norm(V_o)
    R_w = np.column_stack([U_o, V_o, B_w])
    # 像元尺寸: 直接取"每像素的世界位移", 除以分辨率即得原语义的物理尺寸。
    # (原实现用"网格顶点跨度/分辨率", 隐含"UV[0,1] 必须正好铺满网格"; 这里不再依赖该前提)
    unit_to_mm = get_scene_unit_scale()
    return (origin_w, R_w,
            float(np.linalg.norm(dpx)) * unit_to_mm * PROJ_RES['w'],
            float(np.linalg.norm(dpy)) * unit_to_mm * PROJ_RES['h'])


def refine_screen_frame_to_mesh(frame_obj, mesh_obj):
    """用屏面网格(平面.001/Light)几何+UV, 把 C_Screen 精确吸附到屏左上角与正确姿态。
    C_Screen 手摆通常接近但不精确, 此处以网格几何为准做精调; 调整量写入日志。
    返回是否成功。"""
    if frame_obj is None:
        return False
    fr = screen_world_frame_from_uv(mesh_obj)
    if fr is None:
        print("[C_Screen精调] 屏面网格无有效UV/顶点, 跳过(保持手动摆放)")
        return False
    origin_w, R_w = fr[0], fr[1]

    old = np.array(frame_obj.matrix_world)
    old_o = old[:3, 3]
    old_R = old[:3, :3]
    unit_to_mm = get_scene_unit_scale()
    dpos = float(np.linalg.norm(old_o - origin_w) * unit_to_mm)

    def _ang(a, b):
        a = a / (np.linalg.norm(a) + 1e-12)
        b = b / (np.linalg.norm(b) + 1e-12)
        return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))
    dax = [_ang(old_R[:, k], R_w[:, k]) for k in range(3)]

    # 写回精确位姿(内部单位; matrix_world 即内部单位)
    M = mathutils.Matrix.Identity(4)
    for r in range(3):
        for c in range(3):
            M[r][c] = float(R_w[r, c])
        M[r][3] = float(origin_w[r])
    frame_obj.matrix_world = M

    print(f"[C_Screen精调] 已按屏面网格精确吸附: 原点修正={dpos:.3f}mm, "
          f"轴修正 X(右)={dax[0]:.3f}° Y(下)={dax[1]:.3f}° Z(法向)={dax[2]:.3f}°")
    return True


def align_cam_frame_to_camera(frame_obj, cam_obj):
    """把标定坐标系空对象(C_LCam/C_RCam)对齐到相机的 OpenCV 相机坐标系:
      原点 = 相机光心; X+ = 相机右; Y+ = 画面竖直向下; Z+ = 相机正前方(视线方向)。
    Blender 相机本地: X右、Y上、-Z视线。OpenCV: X右、Y下、Z前。
    故 OpenCV 相机系 = 相机世界位姿 右乘 diag(1,-1,-1):
      col0=+X_cam, col1=-Y_cam(向下), col2=-Z_cam(视线前方)。
    返回是否成功。"""
    if frame_obj is None or cam_obj is None:
        return False
    flip = mathutils.Matrix.Diagonal((1.0, -1.0, -1.0, 1.0))  # 4x4, det(旋转)=+1
    frame_obj.matrix_world = cam_obj.matrix_world @ flip
    return True


def align_calib_frames():
    """渲染/导出前, 依据实际相机对象自动摆正 C_LCam / C_RCam 坐标系。
    保证屏幕外参(相对 C_LCam)与相机射线(OpenCV 内参模型)在同一坐标系, 避免系统性变形。"""
    camL = bpy.data.objects.get(LEFT_CAMERA_NAME)
    camR = bpy.data.objects.get(RIGHT_CAMERA_NAME)
    fL = bpy.data.objects.get(LCAM_FRAME_NAME)
    fR = bpy.data.objects.get(RCAM_FRAME_NAME)
    okL = align_cam_frame_to_camera(fL, camL)
    okR = align_cam_frame_to_camera(fR, camR)
    if okL:
        print(f"[标定坐标系] 已将 {LCAM_FRAME_NAME} 对齐到 {LEFT_CAMERA_NAME} 的OpenCV相机系")
    else:
        print(f"[标定坐标系] 警告: 未找到 {LCAM_FRAME_NAME} 或 {LEFT_CAMERA_NAME}, 未对齐")
    if okR:
        print(f"[标定坐标系] 已将 {RCAM_FRAME_NAME} 对齐到 {RIGHT_CAMERA_NAME} 的OpenCV相机系")
    else:
        print(f"[标定坐标系] 警告: 未找到 {RCAM_FRAME_NAME} 或 {RIGHT_CAMERA_NAME}, 未对齐")
    return okL and okR


def screen_extrinsics_from_frames(screen_frame_obj, lcam_frame_obj):
    """由手动创建的坐标系空对象 C_Screen / C_LCam 直接计算显示屏外参

    直接取两坐标系之间的相对位姿(不做相机-OpenCV轴变换, 约定完全由用户在
    Blender 中摆放 C_Screen / C_LCam 时确定):
        [p_lcam; 1] = inv(M_LCam) @ M_Screen @ [p_screen; 1]
      => R = (inv(M_LCam)@M_Screen)[:3,:3],  T = (...)[:3,3] * (内部单位->mm)
    其中 R 把 C_Screen 局部坐标旋转到 C_LCam 坐标系, T 为 C_Screen 原点在
    C_LCam 坐标系中的位置(mm)。与 Reconstruct3D displayTo3D 的
    P = R·[u·psW, v·psH, 0] + T 配套(屏局部原点=屏左上角, +X图像右, +Y图像下)。
    返回 (om, T); 失败返回 None
    """
    if screen_frame_obj is None or lcam_frame_obj is None:
        return None
    M_lcam = np.array(lcam_frame_obj.matrix_world)
    M_screen = np.array(screen_frame_obj.matrix_world)
    M_rel = np.linalg.inv(M_lcam) @ M_screen
    R = M_rel[:3, :3].copy()
    # 去除空对象缩放的影响, 保证 R 为纯旋转(列向量单位化)
    for c in range(3):
        n = np.linalg.norm(R[:, c])
        if n > 1e-12:
            R[:, c] /= n
    t = M_rel[:3, 3] * get_scene_unit_scale()
    om = matrixToRotationVector(R)
    return om, t


def compute_fundamental_matrix(R, t, K1, K2):
    """从外参R,t和内参K1,K2计算基本矩阵F
    F = K2^{-T} [t]_x R K1^{-1}
    """
    t_skew = np.array([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0]
    ], dtype=np.float64)
    F = np.linalg.inv(K2).T @ t_skew @ R @ np.linalg.inv(K1)
    F = F / F[2, 2] if abs(F[2, 2]) > 1e-12 else F
    return F


def export_calibration_params(save_dir):
    """从Blender场景导出标定参数文件 cal_res.txt 和 cal_Projector.txt

    双目外参: 由左/右相机对象位姿直接计算 (渲染使用相机对象, 外参必须与之一致)
    显示屏外参: 由屏面网格('Light')UV拟合用户自定义屏坐标系:
      原点=屏左上角(图像(0,0)处), +X=像素xS+(图像右), +Y=像素yS+(图像下), Z=X×Y右手闭合
      不依赖C_LCam/C_RCam/C_Screen坐标系空对象
    相机内参从相机对象读取(sensor_width/lens)，显示屏物理尺寸从投影仪对象读取。
    """

    # 从UI同步显示屏分辨率到全局PROJ_RES, 确保导出的分辨率/像素尺寸与当前配置一致
    # (文件/生成模式都取 defl_proj_width/height, 默认1280x800; 与start_render中PROJ_RES设置一致)
    scene = bpy.context.scene
    if hasattr(scene, 'defl_proj_width') and scene.defl_proj_width > 0:
        PROJ_RES['w'] = scene.defl_proj_width
        PROJ_RES['h'] = scene.defl_proj_height
    cam_left = bpy.data.objects.get(LEFT_CAMERA_NAME)
    cam_right = bpy.data.objects.get(RIGHT_CAMERA_NAME)
    # 屏面对象必须与渲染时显示条纹的对象一致(setup_projector_texture 用 'Light'),
    # 否则导出的屏外参是另一块平面, 会导致屏在相机后方(z<0)等外参异常。
    projector_obj = bpy.data.objects.get(SCREEN_OBJECT_NAME)
    screen_src = SCREEN_OBJECT_NAME
    if not projector_obj:
        projector_obj = bpy.data.objects.get(PROJECTOR_NAME)
        screen_src = PROJECTOR_NAME
    if not cam_left or not cam_right:
        print("错误：找不到相机对象，无法导出标定参数")
        return False
    if not projector_obj:
        print(f"错误：找不到显示屏对象 '{SCREEN_OBJECT_NAME}'/'{PROJECTOR_NAME}'，无法导出投影仪标定参数")
        return False
    print(f"[标定导出] 显示屏外参使用对象: '{screen_src}' (须与渲染时显示条纹的屏面一致)")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 左相机内参
    fx_l, fy_l, cx_l, cy_l = blender_camera_intrinsics(cam_left, CAM_WIDTH, CAM_HEIGHT)
    # 右相机内参
    fx_r, fy_r, cx_r, cy_r = blender_camera_intrinsics(cam_right, CAM_WIDTH, CAM_HEIGHT)
    # 右相机相对左相机的外参（由真实相机对象位姿计算）
    om_right, t_right = camera_extrinsics_cv(cam_right, cam_left)
    R_right = rotationVectorToMatrix(om_right)
    # 畸变系数（Blender相机无畸变，全部为0）
    dist_left = [0.0] * 5
    dist_right = [0.0] * 5
    # 基本矩阵F
    K_left = np.array([[fx_l, 0, cx_l], [0, fy_l, cy_l], [0, 0, 1]], dtype=np.float64)
    K_right = np.array([[fx_r, 0, cx_r], [0, fy_r, cy_r], [0, 0, 1]], dtype=np.float64)
    F = compute_fundamental_matrix(R_right, t_right, K_left, K_right)
    # 写入 cal_res.txt
    cal_res_path = os.path.join(save_dir, "cal_res.txt")
    with open(cal_res_path, 'w') as f:
        for i in range(3):
            f.write(f"{om_right[i]:.15g}\n")
        for i in range(3):
            f.write(f"{t_right[i]:.15g}\n")
        f.write(f"{fx_l:.15g}\n")
        f.write(f"{fy_l:.15g}\n")
        f.write(f"{cx_l:.15g}\n")
        f.write(f"{cy_l:.15g}\n")
        for i in range(5):
            f.write(f"{dist_left[i]:.15g}\n")
        f.write(f"{fx_r:.15g}\n")
        f.write(f"{fy_r:.15g}\n")
        f.write(f"{cx_r:.15g}\n")
        f.write(f"{cy_r:.15g}\n")
        for i in range(5):
            f.write(f"{dist_right[i]:.15g}\n")
        for i in range(3):
            for j in range(3):
                f.write(f"{F[i, j]:.15g}\n")
    print(f"已导出双目标定参数: {cal_res_path}")
    # 显示屏外参 + 物理参数 (cal_Projector.txt 新格式, 10个数)
    # 屏坐标系约定(与Reconstruct3D displayTo3D配套):
    #   原点 = 屏左上角(相机视角下图像(0,0)像素处) —— 即手动创建的 C_Screen 空对象原点
    #   +X = 像素xS+(图像右), +Y = 像素yS+(图像下), Z = X×Y(右手闭合, 垂直屏面)
    #   像素(xS,yS) -> 局部mm坐标 [xS*psW, yS*psH, 0]
    #   -> 左相机坐标系: P = R @ local + T
    # 优先由手动坐标系 C_Screen / C_LCam 直接取相对位姿计算(RT), 缺失时回退到屏面UV拟合。
    # 先按实际相机对象摆正 C_LCam/C_RCam(OpenCV相机系), 保证屏外参与相机射线同一坐标系。
    align_calib_frames()
    screen_frame = bpy.data.objects.get(SCREEN_FRAME_NAME)
    lcam_frame = bpy.data.objects.get(LCAM_FRAME_NAME)
    # 以屏面网格几何为准, 精调 C_Screen 的位置与三轴姿态(手摆近似 → 精确吸附)
    # refine_screen_frame_to_mesh(screen_frame, projector_obj)
    result = screen_extrinsics_from_frames(screen_frame, lcam_frame)
    if result is not None:
        print(f"[标定导出] 显示屏外参由坐标系 '{SCREEN_FRAME_NAME}' 相对 '{LCAM_FRAME_NAME}' 直接计算")
    else:
        print(f"[标定导出] 未找到 '{SCREEN_FRAME_NAME}'/'{LCAM_FRAME_NAME}' 坐标系, 回退到屏面UV拟合")
        result = screen_frame_from_uv(projector_obj, cam_left)
    if result is None:
        print("错误: 显示屏外参计算失败")
        return False
    om_prj, t_prj = result

    # ---- 对照诊断: 用屏面网格UV独立再算一次屏外参, 与 C_Screen 方法对比 ----
    # dot一致性无法发现屏标定错误(左右共用同一套屏), 故此处独立校核 C_Screen 是否摆放准确。
    try:
        uv_res = screen_frame_from_uv(projector_obj, cam_left)
        if uv_res is not None:
            om_uv, t_uv = uv_res
            R_fr = rotationVectorToMatrix(om_prj)
            R_uv = rotationVectorToMatrix(om_uv)
            dT = float(np.linalg.norm(np.array(t_prj, float) - np.array(t_uv, float)))
            print("[屏外参对照] C_Screen方法 vs 屏面UV拟合(独立校核):")
            print(f"  原点T(左相机系,mm): C_Screen=({t_prj[0]:.2f},{t_prj[1]:.2f},{t_prj[2]:.2f})  "
                  f"UV=({t_uv[0]:.2f},{t_uv[1]:.2f},{t_uv[2]:.2f})  相差={dT:.2f}mm")
            for k, nm in enumerate(("X(图像右)", "Y(图像下)", "Z(屏法向)")):
                a = R_fr[:, k]; b = R_uv[:, k]
                ang = float(np.degrees(np.arccos(np.clip(float(np.dot(a, b)), -1.0, 1.0))))
                print(f"    轴{nm}夹角={ang:.2f}°")
            if dT > 5.0:
                print("  [警告] 两法原点相差>5mm: C_Screen 很可能没建在正确的屏左上角!")
            print("  (若原点相差大或某轴夹角明显>0, 说明 C_Screen 位置/朝向不准, 建议改用UV法或修正C_Screen)")
    except Exception as e:
        print(f"[屏外参对照] UV独立校核失败: {e}")

    # 显示屏物理尺寸 → 像素尺寸 (mm/px)
    # 首选: 由屏面UV几何求宽高(|du|,|dv|), 与图像X/Y严格对应, 方向不会反;
    # 回退: obj.dimensions(可能按局部轴取包围盒, X/Y可能与图像不符)。
    uvfit = screen_world_frame_from_uv(projector_obj)
    if uvfit is not None:
        proj_phys_w = uvfit[2]
        proj_phys_h = uvfit[3]
        src_size = "屏面UV几何"
    else:
        pd = get_projector_dimensions(projector_obj)
        proj_phys_w, proj_phys_h = pd[0], pd[1]
        src_size = "obj.dimensions(回退)"
    pixel_size_w = proj_phys_w / PROJ_RES['w']
    pixel_size_h = proj_phys_h / PROJ_RES['h']
    # 诊断: 打印物理尺寸/像素尺寸/像素长宽比, 并与图像长宽比对比(方形像素时二者应接近)
    try:
        pd2 = get_projector_dimensions(projector_obj)
        pxAspect = (pixel_size_w / pixel_size_h) if pixel_size_h > 1e-9 else 0.0
        imgAspect = float(PROJ_RES['w']) / float(PROJ_RES['h']) if PROJ_RES['h'] else 0.0
        physAspect = (proj_phys_w / proj_phys_h) if proj_phys_h > 1e-9 else 0.0
        print(f"[像素尺寸] 屏对象='{screen_src}' 来源={src_size}: 屏物理={proj_phys_w:.2f}x{proj_phys_h:.2f}mm "
              f"-> psW={pixel_size_w:.5f} psH={pixel_size_h:.5f} mm/px "
              f"(校验: psW*{PROJ_RES['w']}={pixel_size_w*PROJ_RES['w']:.2f}mm, psH*{PROJ_RES['h']}={pixel_size_h*PROJ_RES['h']:.2f}mm)")
        print(f"           分辨率={PROJ_RES['w']}x{PROJ_RES['h']}  物理长宽比={physAspect:.3f} 图像长宽比={imgAspect:.3f} "
              f"像素长宽比psW/psH={pxAspect:.3f}")
        print(f"           对比 obj.dimensions={pd2[0]:.2f}x{pd2[1]:.2f}mm")
        if imgAspect > 1e-6 and abs(physAspect - imgAspect) / imgAspect > 0.02:
            print("  [提示] 屏物理长宽比与图像长宽比相差>2%: 像素非方形。若显示器应为方形像素, "
                  "请核对屏面网格尺寸/UV是否与所投图像一致。")
    except Exception as e:
        print(f"[像素尺寸] 诊断输出失败: {e}")
    # 屏外参物理合理性自检: 屏四角(自定义屏系)变换到左相机系后 z 必须>0(相机前方)
    R_prj = rotationVectorToMatrix(om_prj)
    corners = [(0, 0), (PROJ_RES['w'], 0), (0, PROJ_RES['h']), (PROJ_RES['w'], PROJ_RES['h'])]
    zs = []
    for xS, yS in corners:
        P = R_prj @ np.array([xS * pixel_size_w, yS * pixel_size_h, 0.0]) + t_prj
        zs.append(P[2])
    print("显示屏外参自检 (左相机系, mm):")
    for (xS, yS), z in zip(corners, zs):
        print(f"  屏角({xS},{yS}) -> z = {z:.2f}")
    if min(zs) <= 0:
        print("错误: 屏面变换后z<=0(位于相机后方), 屏外参无效, 请检查屏面/相机位姿与UV方向")
        return False
    # 写入 cal_Projector.txt: om(3) T(3) psW psH prjW prjH
    cal_prj_path = os.path.join(save_dir, "cal_Projector.txt")
    with open(cal_prj_path, 'w') as f:
        for i in range(3):
            f.write(f"{om_prj[i]:.15g}\n")
        for i in range(3):
            f.write(f"{t_prj[i]:.15g}\n")
        f.write(f"{pixel_size_w:.15g}\n")
        f.write(f"{pixel_size_h:.15g}\n")
        f.write(f"{PROJ_RES['w']}\n")
        f.write(f"{PROJ_RES['h']}\n")
    print(f"已导出显示屏标定参数: {cal_prj_path}")
    print(f"  显示屏物理尺寸: {proj_phys_w:.2f} x {proj_phys_h:.2f} mm (像素尺寸 {pixel_size_w:.4f} x {pixel_size_h:.4f} mm/px)")
    print(f"  显示屏分辨率: {PROJ_RES['w']} x {PROJ_RES['h']}")
    print(f"  屏坐标系: 原点=屏左上角, +X=图像右, +Y=图像下 (T=屏左上角位置)")
    print(f"  显示屏外参: T=({t_prj[0]:.2f}, {t_prj[1]:.2f}, {t_prj[2]:.2f}) mm")
    print(f"  左相机内参: fx={fx_l:.4f} fy={fy_l:.4f} cx={cx_l:.1f} cy={cy_l:.1f}")
    print(f"  右相机内参: fx={fx_r:.4f} fy={fy_r:.4f} cx={cx_r:.1f} cy={cy_r:.1f}")
    print(f"  基线长度: {np.linalg.norm(t_right):.4f} mm")
    return True


def get_projector_dimensions(proj_obj):
    """获取显示屏(平面)的物理尺寸，单位mm
    使用Blender对象的dimensions属性（已应用scale），乘以场景单位→mm的转换因子
    """
    unit_to_mm = get_scene_unit_scale()
    scale = proj_obj.scale
    bbox = [proj_obj.dimensions[0], proj_obj.dimensions[1], proj_obj.dimensions[2]]
    w_mm = bbox[0] * unit_to_mm
    h_mm = bbox[1] * unit_to_mm
    # 如果dimensions为0（可能未应用scale），则用scale * 默认平面尺寸(2场景单位)
    if w_mm < 0.01 and h_mm < 0.01:
        w_mm = scale[0] * 2.0 * unit_to_mm
        h_mm = scale[1] * 2.0 * unit_to_mm
    return (w_mm, h_mm)


def exportPatternInfo():
    """渲染完成后输出条纹配置到 base_path/pattern_info.txt
    供 Reconstruct3D 启动时自动导入: 图像目录/标定目录/分辨率/像素周期/偏移
    """
    info = lastPatternInfo
    pose0 = os.path.join(base_path, "pose_0")
    pitchesV = info.get('pitches', [])
    pitchesH = info.get('pitchesH', []) or pitchesV
    pitchesStr = ",".join(f"{p:.3f}" for p in pitchesV)
    pitchesHStr = ",".join(f"{p:.3f}" for p in pitchesH)
    lines = [
        f"imageDir = {pose0}",
        f"calibDir = {base_path}",
        f"camW = {CAM_WIDTH}",
        f"camH = {CAM_HEIGHT}",
        f"prjW = {PROJ_RES['w']}",
        f"prjH = {PROJ_RES['h']}",
        "numFreq = 3",
        "numStep = 4",
        f"pitches = {pitchesStr}",
        f"pitchesV = {pitchesStr}",
        f"pitchesH = {pitchesHStr}",
        f"offsetFringe = {info.get('offset', 0.0):.3f}",
        f"offsetFringeH = {info.get('offset', 0.0):.3f}",
    ]
    path = os.path.join(base_path, "pattern_info.txt")
    try:
        with open(path, 'w') as f:
            f.write("\n".join(lines) + "\n")
        print(f"已导出条纹配置: {path}")
        return True
    except Exception as e:
        print(f"导出pattern_info.txt失败: {e}")
        return False


def main():
    global calibration_board, projector, camera_left, camera_right

    if not os.path.exists(base_path):
        os.makedirs(base_path)

    calibration_board = get_selected_board(bpy.context) or bpy.data.objects.get("CalibrationBoard")
    projector = bpy.data.objects.get("Light")
    camera_left = bpy.data.objects.get(LEFT_CAMERA_NAME)
    camera_right = bpy.data.objects.get(RIGHT_CAMERA_NAME)

    missing = []
    if not calibration_board: missing.append("被测物体(CalibrationBoard或面板所选)")
    if not projector: missing.append("Light")
    if not camera_left: missing.append(LEFT_CAMERA_NAME)
    if not camera_right: missing.append(RIGHT_CAMERA_NAME)
    if missing:
        print(f"警告：场景中缺少必要对象: {', '.join(missing)}（点击开始渲染时同样会校验）")
    else:
        print("场景对象检查通过。")

    registerRenderUI()
    print("\n请在 3D 视图侧边栏(按N键)的 Deflectometry 面板中配置投图模式，然后点击\"开始渲染\"。")


if __name__ == "__main__":
    main()
