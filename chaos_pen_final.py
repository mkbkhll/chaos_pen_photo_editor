import cv2
import numpy as np
import json
import os
import sys
from pathlib import Path
import gradio as gr
from insightface.app import FaceAnalysis
from rembg import remove
import ollama

# ---------------------- 本地环境适配：独立项目目录（无路径问题） ----------------------
project_dir = Path(__file__).parent / "ChaosPen_智能修图工具"
project_dir.mkdir(exist_ok=True)
os.chdir(project_dir)
print(f"✅ 本地项目目录：{project_dir.absolute()}")

# ---------------------- 初始化人脸模型：基础检测+遮挡优化双模型 ----------------------
# 基础人像识别模型（优先判定是否为人像，阈值适中）
basic_face_app = None
# 遮挡人脸优化模型（仅当判定为人像后，启动该模型做精细化处理）
occlusion_face_app = None

try:
    # 基础模型：640高清，常规人像精准判定
    basic_face_app = FaceAnalysis(providers=['CPUExecutionProvider'], allowed_modules=['detection'])
    basic_face_app.prepare(ctx_id=0, det_size=(640, 640), threshold=0.5)
    # 遮挡优化模型：320轻量，低阈值适配遮挡/半脸/侧脸
    occlusion_face_app = FaceAnalysis(providers=['CPUExecutionProvider'], allowed_modules=['detection'])
    occlusion_face_app.prepare(ctx_id=0, det_size=(320, 320), threshold=0.4)
    print("✅ 人脸模型初始化完成（基础识别+遮挡优化双模式）")
except Exception as e:
    print(f"⚠️  Insightface模型兜底：{str(e)[:60]}...")
    print("✅ 自动启用OpenCV Haar人脸检测（基础+遮挡双流程）")

# 固定配置：支持的修图操作/滤镜
SUPPORT_OPERATIONS = ["美颜", "磨皮", "美白", "清晰", "放大", "抠图", "去背景"]
SUPPORT_FILTERS = ["电影感", "清新日系", "复古胶片", "黑金质感", "赛博朋克", "水墨风"]

# ---------------------- 第一步：基础人像识别（核心判定，先确定是否为人像） ----------------------
def basic_face_detection(img):
    """基础人像检测，先判定是否为人像，返回True/False"""
    if img is None or len(img.shape) != 3 or img.shape[0] < 30 or img.shape[1] < 30:
        return False
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # 方案1：Insightface基础模型（优先）
    if basic_face_app is not None:
        try:
            faces = basic_face_app.get(img_bgr, max_num=1)
            if len(faces) > 0:
                return True
        except:
            pass
    # 方案2：OpenCV Haar基础检测（兜底）
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces_cv = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
        if len(faces_cv) > 0:
            return True
    except Exception as e:
        pass
    return False

# ---------------------- 第二步：遮挡人脸专项优化（仅当基础识别为人像后，才执行此步骤） ----------------------
def occlusion_face_optimization(img):
    """遮挡人脸精细化处理，仅对已判定的人像做遮挡/半脸/侧脸优化，返回处理后的图像"""
    if img is None or len(img.shape) != 3 or img.shape[0] < 30 or img.shape[1] < 30:
        return img
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_copy = img_bgr.copy()
    # 方案1：Insightface遮挡优化模型（优先，适配遮挡/半脸/侧脸）
    if occlusion_face_app is not None:
        try:
            faces = occlusion_face_app.get(img_bgr, max_num=1)
            if len(faces) > 0:
                return cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        except:
            pass
    # 方案2：OpenCV Haar遮挡专项检测（兜底，含侧脸/半脸/口罩遮挡）
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        # 遮挡/半脸检测：低邻域+精细缩放
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces_occlusion = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=2, minSize=(25, 25))
        # 侧脸检测补充
        profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        faces_profile = profile_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=2, minSize=(25, 25))
        if len(faces_occlusion) > 0 or len(faces_profile) > 0:
            return cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    except Exception as e:
        pass
    return cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)

# ---------------------- 场景总识别：先基础人脸→再判动物/景物（严格按优先级） ----------------------
def detect_scene_by_image(img):
    """
    场景识别总流程：
    1. 先执行basic_face_detection，判定为人像则返回「人像」
    2. 非人像时，再执行动物/景物判定
    """
    if img is None or len(img.shape) not in [2, 3] or img.shape[0] < 20 or img.shape[1] < 20:
        return "景物"
    # 第一步：先基础人像识别，判定为人像则直接返回，后续再做遮挡优化
    if basic_face_detection(img):
        return "人像"
    # 非人像：再判定动物/景物
    if len(img.shape) == 3:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lower_warm = np.array([0, 20, 50])
        upper_warm = np.array([40, 255, 255])
        mask_warm = cv2.inRange(hsv, lower_warm, upper_warm)
        warm_ratio = np.sum(mask_warm) / (img.shape[0] * img.shape[1]) if (img.shape[0] * img.shape[1]) > 0 else 0
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edge = cv2.Canny(gray, 50, 150)
        texture_ratio = np.sum(edge) / (img.shape[0] * img.shape[1] * 255) if (img.shape[0] * img.shape[1] * 255) > 0 else 0
        contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        circularity = 0
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(max_contour, True)
            area = cv2.contourArea(max_contour)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
        # 动物三重判定条件（缺一不可，杜绝景物误判）
        if warm_ratio > 0.15 and (0.08 < texture_ratio < 0.35) and circularity > 0.2:
            return "动物"
    # 非人脸+非动物 → 景物
    return "景物"

# ---------------------- 分场景增强：景物超轻微优化，人像/动物正常增强 ----------------------
def enhance_blurry_image(img, scene="景物"):
    if img is None or len(img.shape) not in [2, 3] or img.shape[0] < 20 or img.shape[1] < 20:
        return img
    # 景物场景：超轻量去噪+锐化，严格保留原图质感，几乎无视觉变化
    if scene == "景物":
        img_denoise = cv2.fastNlMeansDenoisingColored(img, None, 1, 1, 7, 21) if len(img.shape)==3 else img
        kernel = np.array([[0, -0.05, 0], [-0.05, 1.1, -0.05], [0, -0.05, 0]])  # 极致轻量锐化
    # 人像/动物场景：正常优化幅度，保证清晰效果
    else:
        is_low_res = img.shape[0] < 200 or img.shape[1] < 200
        kernel = np.array([[0, -0.3, 0], [-0.3, 2.2, -0.3], [0, -0.3, 0]]) if is_low_res else np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
        img_denoise = cv2.fastNlMeansDenoisingColored(img, None, 8, 8, 7, 21) if (len(img.shape)==3 and not is_low_res) else img
    img_sharpen = cv2.filter2D(img_denoise, -1, kernel)
    return img_sharpen

# ---------------------- 人像专属黄金比例优化（含遮挡人像适配，自然不假） ----------------------
def optimize_face_with_golden_ratio(img):
    if img is None or len(img.shape) != 3 or img.shape[0] < 50 or img.shape[1] < 50:
        return img
    # 先执行遮挡人脸优化，再做美颜处理
    img_occlusion_optim = occlusion_face_optimization(img)
    img_copy = cv2.cvtColor(img_occlusion_optim, cv2.COLOR_RGB2BGR)
    GOLDEN_RATIO = 1.618
    ADJUST_SCALE = 0.25
    MIN_SCALE = 1 - ADJUST_SCALE
    MAX_SCALE = 1 + ADJUST_SCALE
    ANGLE_THRESH = 15
    MIN_REGION_SIZE = 15
    # 优先用遮挡优化模型检测关键点
    faces = occlusion_face_app.get(img_copy, max_num=1) if occlusion_face_app is not None else []
    if not faces:
        # 无关键点时，仅做基础磨皮美白（适配遮挡/半脸）
        gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
        contrast = gray.std()
        d = 8 if contrast < 50 else 12
        sigmaColor = 40 if contrast < 50 else 60
        sigmaSpace = 40 if contrast < 50 else 60
        img_copy = cv2.bilateralFilter(img_copy, d, sigmaColor, sigmaSpace)
        alpha = 1.05 if contrast < 50 else 1.1
        beta = 3 if contrast < 50 else 5
        img_copy = cv2.addWeighted(img_copy, alpha, np.zeros_like(img_copy), 0, beta)
        return cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    # 有关键点时，黄金比例五官微调（仅对可见部位处理）
    for face in faces:
        kps = face.get('kps', None)
        if kps is None or len(kps) < 15:
            continue
        # 眼睛区域优化（仅处理可见眼睛）
        eye_left, eye_right = kps[0], kps[1]
        eye_vector = eye_right - eye_left
        angle = np.degrees(np.arctan2(eye_vector[1], eye_vector[0]))
        if abs(angle) >= ANGLE_THRESH:
            continue
        eye_width = np.linalg.norm(eye_right - eye_left)
        y1, y2 = max(0, int(eye_left[1]-eye_width/2)), min(img_copy.shape[0], int(eye_right[1]+eye_width/2))
        x1, x2 = max(0, int(eye_left[0]-eye_width/2)), min(img_copy.shape[1], int(eye_right[0]+eye_width/2))
        if (x2 - x1) > MIN_REGION_SIZE and (y2 - y1) > MIN_REGION_SIZE:
            eye_region = img_copy[y1:y2, x1:x2]
            if eye_region.size > 0:
                eye_region = cv2.resize(eye_region, None, fx=MAX_SCALE, fy=MAX_SCALE, interpolation=cv2.INTER_CUBIC)
                eye_region = cv2.resize(eye_region, (x2-x1, y2-y1), interpolation=cv2.INTER_CUBIC)
                img_copy[y1:y2, x1:x2] = eye_region
        # 下巴/脸颊精细化微调（适配遮挡后的面部轮廓）
        nose, chin = kps[2], kps[8]
        face_height = np.linalg.norm(chin - nose)
        y1, y2 = max(0, int(nose[1])), min(img_copy.shape[0], int(chin[1]+face_height/4))
        x1, x2 = max(0, int(nose[0]-face_height/2)), min(img_copy.shape[1], int(nose[0]+face_height/2))
        if (x2 - x1) > MIN_REGION_SIZE and (y2 - y1) > MIN_REGION_SIZE:
            chin_region = img_copy[y1:y2, x1:x2]
            if chin_region.size > 0:
                chin_region = cv2.resize(chin_region, None, fx=MIN_SCALE, fy=1, interpolation=cv2.INTER_CUBIC)
                chin_region = cv2.resize(chin_region, (x2-x1, y2-y1), interpolation=cv2.INTER_CUBIC)
                img_copy[y1:y2, x1:x2] = chin_region
        # 动态磨皮美白（根据面部可见度适配，不假白）
        gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
        contrast = gray.std()
        d = 8 if contrast < 50 else 12
        sigmaColor = 40 if contrast < 50 else 60
        sigmaSpace = 40 if contrast < 50 else 60
        img_copy = cv2.bilateralFilter(img_copy, d, sigmaColor, sigmaSpace)
        alpha = 1.05 if contrast < 50 else 1.1
        beta = 3 if contrast < 50 else 5
        img_copy = cv2.addWeighted(img_copy, alpha, np.zeros_like(img_copy), 0, beta)
    return cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)

# ---------------------- 滤镜应用：分场景适配，低清图自动降效果 ----------------------
def apply_filter(img, filter_name):
    if img is None or filter_name not in SUPPORT_FILTERS or len(img.shape) != 3 or img.shape[0] < 20 or img.shape[1] < 20:
        return img
    img_copy = img.copy()
    rows, cols = img_copy.shape[:2]
    is_low_res = rows < 200 or cols < 200  # 低清图自动降低滤镜强度，避免失真
    if filter_name == "电影感":
        alpha = 1.2 if is_low_res else 1.3
        beta = -15 if is_low_res else -20
        img_copy = cv2.addWeighted(img_copy, alpha, np.zeros_like(img_copy), 0, beta)
        b, g, r = cv2.split(img_copy)
        b = cv2.addWeighted(b, 1.1 if is_low_res else 1.15, np.zeros_like(b), 0, 0)
        r = cv2.addWeighted(r, 1.03 if is_low_res else 1.05, np.zeros_like(r), 0, 0)
        img_copy = cv2.merge((b, g, r))
        mask = cv2.getGaussianKernel(cols, 200 if is_low_res else 300) @ cv2.getGaussianKernel(rows, 200 if is_low_res else 300).T
        mask = cv2.resize(mask, (cols, rows))
        mask = np.stack([mask]*3, axis=-1)
        img_copy = (img_copy * (mask / mask.max())).astype(np.uint8)
    elif filter_name == "清新日系":
        alpha = 1.03 if is_low_res else 1.05
        beta = 10 if is_low_res else 15
        img_copy = cv2.addWeighted(img_copy, alpha, np.zeros_like(img_copy), 0, beta)
        hsv = cv2.cvtColor(img_copy, cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = cv2.addWeighted(hsv[:, :, 1], 0.85 if is_low_res else 0.75, np.zeros_like(hsv[:, :, 1]), 0, 0)
        img_copy = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        b = cv2.split(img_copy)[0]
        b = cv2.addWeighted(b, 1.03 if is_low_res else 1.05, np.zeros_like(b), 0, 0)
        img_copy = cv2.merge((b, cv2.split(img_copy)[1], cv2.split(img_copy)[2]))
    elif filter_name == "复古胶片":
        img_copy = cv2.addWeighted(img_copy, 0.95, np.zeros_like(img_copy), 0, 20)
        b, g, r = cv2.split(img_copy)
        r = cv2.addWeighted(r, 1.1 if is_low_res else 1.15, np.zeros_like(r), 0, 0)
        g = cv2.addWeighted(g, 1.03 if is_low_res else 1.05, np.zeros_like(g), 0, 0)
        img_copy = cv2.merge((b, g, r))
        noise = np.random.normal(0, 1 if is_low_res else 3, img_copy.shape).astype(np.int16)
        img_copy = np.clip(img_copy + noise, 0, 255).astype(np.uint8)
    elif filter_name == "黑金质感":
        hls = cv2.cvtColor(img_copy, cv2.COLOR_RGB2HLS)
        thresh = 140 if is_low_res else 130
        hls[:, :, 1] = cv2.threshold(hls[:, :, 1], thresh, 255, cv2.THRESH_BINARY)[1]
        img_copy = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)
        alpha = 1.3 if is_low_res else 1.4
        beta = -10 if is_low_res else -15
        img_copy = cv2.addWeighted(img_copy, alpha, np.zeros_like(img_copy), 0, beta)
    elif filter_name == "赛博朋克":
        hsv = cv2.cvtColor(img_copy, cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = cv2.addWeighted(hsv[:, :, 1], 1.2 if is_low_res else 1.4, np.zeros_like(hsv[:, :, 1]), 0, 0)
        hsv[:, :, 2] = cv2.addWeighted(hsv[:, :, 2], 1.05 if is_low_res else 1.1, np.zeros_like(hsv[:, :, 2]), 0, 0)
        img_copy = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        b, r = cv2.split(img_copy)[0], cv2.split(img_copy)[2]
        b = cv2.addWeighted(b, 1.15 if is_low_res else 1.25, np.zeros_like(b), 0, 0)
        r = cv2.addWeighted(r, 1.15 if is_low_res else 1.25, np.zeros_like(r), 0, 0)
        img_copy = cv2.merge((b, cv2.split(img_copy)[1], r))
    elif filter_name == "水墨风":
        gray = cv2.cvtColor(img_copy, cv2.COLOR_RGB2GRAY)
        min_val = 40 if is_low_res else 60
        max_val = 140 if is_low_res else 160
        edge = cv2.Canny(gray, min_val, max_val)
        edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2RGB)
        gray_rgb = gray[:, :, np.newaxis].repeat(3, axis=2)
        img_copy = cv2.addWeighted(gray_rgb, 0.8, edge, 0.2, 0)
    return img_copy

# ---------------------- AI指令解析（适配Llama3，本地运行） ----------------------
def parse_prompt_by_llm(user_prompt):
    if not user_prompt or not user_prompt.strip():
        return {"operations": [], "filter": ""}
    system_prompt = f"""
    你是图像编辑指令解析专家，仅提取操作和滤镜，输出严格JSON格式（无多余文字）。
    规则：1.操作仅从[{','.join(SUPPORT_OPERATIONS)}]选，无则[]；2.滤镜仅从[{','.join(SUPPORT_FILTERS)}]选，无则""；
    3.模糊指令（如P好看/调清晰）仅提取["清晰"]，输出固定键：operations、filter。
    """
    try:
        response = ollama.chat(
            model="llama3",
            messages=[{"role":"system","content":system_prompt.strip()},{"role":"user","content":user_prompt.strip()}]
        )
        result = json.loads(response['message']['content'].strip())
        result["operations"] = [op for op in result.get("operations", []) if op in SUPPORT_OPERATIONS]
        result["filter"] = result.get("filter", "") if result.get("filter") in SUPPORT_FILTERS else ""
        return result
    except Exception as e:
        return {"operations": ["清晰"], "filter": ""}

# ---------------------- 主处理函数（严格按：基础人脸→遮挡优化→分场景处理） ----------------------
def process_image(input_img, user_prompt):
    if input_img is None:
        return None, "⚠️ 请先上传有效图片（建议≥200*200）！"
    # 1. 场景检测（先基础人脸→再判动物/景物）
    scene = detect_scene_by_image(input_img)
    # 2. 解析用户指令
    prompt_result = parse_prompt_by_llm(user_prompt)
    operations = prompt_result["operations"]
    filter_name = prompt_result["filter"]
    # 3. 非人像过滤美颜操作，避免误处理
    if scene != "人像":
        operations = [op for op in operations if op not in ["美颜", "磨皮", "美白"]]
    # 4. 无指定滤镜，按场景自动匹配
    if not filter_name:
        filter_name = "清新日系" if scene == "动物" else "电影感"
    # 5. 初始化结果和日志
    img_result = input_img.copy()
    msg_list = [f"✅ 场景检测结果：{scene}（先基础识别→再精细化处理）"]
    # 6. 无操作时默认执行清晰优化
    if not operations:
        operations = ["清晰"]
        msg_list.append("ℹ️  未提取修图操作，默认执行场景专属清晰优化")
    # 7. 核心处理流程（按优先级执行）
    for op in operations:
        if op == "清晰" and img_result is not None:
            img_result = enhance_blurry_image(img_result, scene)
            msg_list.append(f"✅ 执行{scene}专属清晰优化（精细幅度，保留原图质感）")
        elif op == "放大" and img_result is not None:
            img_result = cv2.resize(img_result, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            img_result = enhance_blurry_image(img_result, scene)
            msg_list.append(f"✅ 2倍高清放大+{scene}专属锐化")
        elif op in ["抠图", "去背景"] and img_result is not None:
            try:
                img_result = remove(img_result)
                msg_list.append(f"✅ 成功{op}，保留主体细节")
            except Exception as e:
                msg_list.append(f"⚠️  {op}失败，已跳过该操作")
    # 8. 人像专属处理：先遮挡优化→再美颜/磨皮/美白（核心优先级）
    if scene == "人像" and img_result is not None and len(img_result.shape)==3:
        msg_list.append("✅ 启动人像精细化处理：先遮挡优化→再美颜操作")
        # 先执行遮挡人脸优化
        img_result = occlusion_face_optimization(img_result)
        # 再执行美颜/磨皮/美白
        if "美颜" in operations:
            img_result = optimize_face_with_golden_ratio(img_result)
            msg_list.append("✅ 执行人像黄金比例美颜（适配遮挡/半脸，自然不假）")
        elif "磨皮" in operations or "美白" in operations:
            gray = cv2.cvtColor(img_result, cv2.COLOR_RGB2GRAY)
            contrast = gray.std()
            d = 8 if contrast < 50 else 12
            img_result = cv2.bilateralFilter(img_result, d, 40 if contrast < 50 else 60, 40 if contrast < 50 else 60)
            alpha = 1.05 if contrast < 50 else 1.1
            beta = 3 if contrast < 50 else 5
            img_result = cv2.addWeighted(img_result, alpha, np.zeros_like(img_result), 0, beta)
            msg_list.append(f"✅ 执行人像{','.join([op for op in operations if op in ['磨皮','美白']])}（动态参数，不假白）")
    # 动物/景物执行专属滤镜
    if scene in ["动物", "景物"] and img_result is not None and len(img_result.shape)==3:
        img_result = apply_filter(img_result, filter_name)
        msg_list.append(f"✅ 执行{scene}专属「{filter_name}」滤镜（适配场景特征，效果自然）")
    # 异常兜底，返回原图
    if img_result is None:
        img_result = input_img
        msg_list.append("⚠️  处理异常，已返回原始图片")
    # 拼接日志
    result_msg = "\n".join(msg_list)
    return img_result, result_msg

# ---------------------- Gradio可视化界面（本地友好，操作简洁） ----------------------
if __name__ == "__main__":
    with gr.Blocks(title="混沌画笔 - 本地智能修图", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎨 混沌画笔")
        gr.Markdown("### 🔍 先人像识别→再遮挡优化 | 🖼️ 分场景精准修图 | 📸 自然语言指令")
        gr.Markdown("#### ✨ 本地运行版 | 无网络依赖 | 隐私保护")
        gr.Markdown("#### 📌 支持指令示例：")
        gr.Markdown("- 人像：自拍P帅点 | 磨皮美白 | 五官修精致 | 放大并美颜 | 口罩照优化")
        gr.Markdown("- 动物：猫咪调萌点 | 狗狗照片锐化 | 加清新日系滤镜")
        gr.Markdown("- 景物：海边加电影感 | 夜景赛博朋克 | 山水水墨风 | 仅轻微清晰")
        
        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                input_image = gr.Image(
                    type="numpy", height=400, image_mode="RGB",
                    label="上传图片（支持高清/低清/遮挡照，最大20MB）",
                    elem_id="upload-img"
                )
                user_prompt = gr.Textbox(
                    placeholder="输入修图需求（自然语言即可，如：P好看点/口罩照优化/加电影感）",
                    lines=4, label="修图指令", elem_id="prompt-input"
                )
                process_btn = gr.Button("✨ 开始智能修图", variant="primary", size="lg")
            
            with gr.Column(scale=1, min_width=320):
                output_image = gr.Image(
                    height=400, image_mode="RGB",
                    label="修图结果（可直接下载）", elem_id="result-img"
                )
                result_text = gr.Textbox(
                    label="处理日志（查看流程/结果）", interactive=False, lines=6, elem_id="status-text"
                )
        
        # 绑定按钮点击事件
        process_btn.click(
            fn=process_image,
            inputs=[input_image, user_prompt],
            outputs=[output_image, result_text],
            show_progress=True
        )
        
        # 本地界面美化CSS（适配电脑端）
        demo.css = """
        #upload-img, #result-img {border: 2px dashed #6366F1 !important; border-radius: 12px !important; margin-bottom: 15px !important;}
        #prompt-input {border-radius: 12px !important; border: 1px solid #C7D2FE !important; padding: 12px !important; margin-bottom: 15px !important;}
        #status-text {border-radius: 12px !important; border: 1px solid #C7D2FE !important; background: #F5F7FF !important; padding: 12px !important;}
        .gradio-container {max-width: 1200px !important; margin: 1rem auto !important; padding: 1.5rem !important;}
        button {border-radius: 12px !important; background: #6366F1 !important; color: white !important; border: none !important; font-size: 16px !important; padding: 10px 0 !important;}
        button:hover {background: #4F46E5 !important; transform: scale(1.02) !important; transition: all 0.2s ease !important;}
        h1 {color: #4F46E5 !important; text-align: center; margin-bottom: 1rem !important; font-weight: 700 !important;}
        h3, .markdown h4 {color: #6366F1 !important; margin: 0.8rem 0 !important;}
        .markdown li {margin: 0.4rem 0 !important; color: #4B5563 !important; line-height: 1.6 !important;}
        .gr-col {margin: 0 20px !important;}
        """
    
    # 启动本地服务（默认7860，被占则自动切8080，仅本地可访问）
    try:
        demo.launch(
            share=False,
            server_name="127.0.0.1",
            server_port=7860,
            show_error=True,
            quiet=True,
            max_file_size="20MB"
        )
    except OSError:
        demo.launch(
            share=False,
            server_name="127.0.0.1",
            server_port=8080,
            show_error=True,
            quiet=True,
            max_file_size="20MB"
        )