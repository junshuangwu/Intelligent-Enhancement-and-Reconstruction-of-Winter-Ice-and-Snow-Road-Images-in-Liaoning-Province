# -*- coding: utf-8 -*-
# ============================================================
# A题：道路图像增强与评价系统
# ============================================================
# 这份程序可以理解成一个“批量修图 + 自动打分”的系统。
#
# 它主要做三件事：
# 1. 找图片：从文件夹或 zip 压缩包中收集道路图片。
# 2. 增强图片：根据图片情况进行提亮、去雾、压高光、增强边缘、处理雪天区域等。
# 3. 评价结果：计算增强前后的亮度、对比度、熵、边缘、SSIM、综合质量分等指标。
#
# scene 表示图片场景类型：
# - clear：清晰/普通场景
# - light：轻度退化场景，例如轻雾、轻雪、轻微低照度
# - heavy：重度退化场景，例如大雾、大雪、强反光、严重模糊
#
# 注意：OpenCV 默认图片颜色顺序是 BGR，不是常见的 RGB。
# 所以代码里 img_bgr 表示“OpenCV 格式的彩色图片”。
# ============================================================

from __future__ import annotations
# argparse：读取命令行参数，比如输入数据路径、输出目录等。
import argparse
# os：系统相关操作，本程序里用得不多，保留不影响运行。
import os
# re：正则表达式，用来整理文件夹/压缩包名字。
import re
# math：数学函数，这里主要用 exp 计算平滑权重。
import math
# zipfile：用于解压 zip 格式的数据集。
import zipfile
# Path：更方便地处理文件路径，Windows 和 Linux 都适用。
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
# cv2：OpenCV 图像处理库，负责读图、滤波、去雾、锐化等操作。
import cv2
# numpy：数值计算库，图片本质上就是数组。
import numpy as np
# pandas：表格处理库，用来保存每张图片的评价指标。
import pandas as pd
# tqdm：进度条，运行时能看到处理到哪一张图。
from tqdm import tqdm
# SSIM：结构相似度，用来衡量增强图和原图结构是否接近。
from skimage.metrics import structural_similarity as ssim
# matplotlib：画图工具，用来生成增强前后对比图。
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
# 程序允许处理的图片格式。
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
# ------------------------------------------------------------
# 创建文件夹：如果输出目录不存在，就自动创建。
# parents=True 表示上级目录不存在也一起创建；exist_ok=True 表示目录已存在也不报错。
def safe_makedirs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
# ------------------------------------------------------------
# 读取图片，支持中文路径。
# 普通 cv2.imread 有时读不了中文路径，所以这里先读成字节，再解码成图片。
# 读取成功返回图片数组；失败返回 None。
def imread_unicode(path: Path) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None
# ------------------------------------------------------------
# 保存图片，支持中文路径。
# 先把图片编码成 jpg/png 等格式，再写入文件，避免中文路径报错。
# 返回 True 表示保存成功。
def imwrite_unicode(path: Path, img: np.ndarray) -> bool:
    safe_makedirs(path.parent)
    ext = path.suffix.lower() or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(str(path))
    return bool(ok)
# ------------------------------------------------------------
# 解压 zip 数据集。
# 如果传入 zip 文件，就解压到 output_dir/extracted 下面；如果之前解压过，就直接复用。
def extract_if_needed(zip_path: Optional[str], extract_root: Path) -> Optional[Path]:
    if not zip_path:
        return None
    zpath = Path(zip_path)
    if not zpath.exists():
        raise FileNotFoundError(f"找不到压缩包：{zpath}")
    out_dir = extract_root / re.sub(r"[^\w\u4e00-\u9fff]+", "_", zpath.stem)
    if out_dir.exists() and any(out_dir.rglob("*")):
        return out_dir
    safe_makedirs(out_dir)
    with zipfile.ZipFile(zpath, "r") as zf:
        zf.extractall(out_dir)
    return out_dir
# ------------------------------------------------------------
# 收集目录下的全部图片。
# root.rglob("*") 会递归扫描所有子文件夹，只保留 jpg/png/bmp/tif 等图片。
# 排序时优先处理 test/text，其次 val，最后 train，方便优先看到测试集效果。
def collect_images(root: Path, scene_name: str) -> List[Path]:
    files = [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]
    def key_func(p: Path):
        s = str(p).lower()
        split_priority = 0 if ("test" in s or "text" in s) else (1 if "val" in s else 2)
        return (split_priority, str(p))
    return sorted(files, key=key_func)
# ------------------------------------------------------------
# 判断图片属于 test、val、train 还是 unknown。
# 这个结果主要用于输出路径分类，例如 enhanced/clear/test/xxx_enhanced.jpg。
def infer_split(path: Path) -> str:
    s = str(path).lower()
    if "test" in s or "text" in s:
        return "test"
    if "val" in s:
        return "val"
    if "train" in s:
        return "train"
    return "unknown"
# ------------------------------------------------------------
# 把图片像素从 0~255 转成 0~1 的小数形式。
# 当前程序中不是核心步骤，但保留这个函数方便后续扩展。
def to_float01(img_bgr: np.ndarray) -> np.ndarray:
    return img_bgr.astype(np.float32) / 255.0
# ------------------------------------------------------------
# 计算图像熵 Entropy。
# 通俗理解：熵表示图像信息丰富程度。熵越高，灰度变化越丰富，细节通常越多。
def calc_entropy(gray_u8: np.ndarray) -> float:
    hist = cv2.calcHist([gray_u8], [0], None, [256], [0, 256]).ravel()
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())
# ------------------------------------------------------------
# 计算单张图片的基础指标。
# 包括亮度、对比度、信息熵、边缘强度、清晰度、过曝比例、暗部比例、疑似雪/白雾比例、噪声估计等。
# 这些指标后面既用来判断图片退化情况，也用来评价增强效果。
def calc_metrics(img_bgr: np.ndarray) -> Dict[str, float]:
    img = img_bgr.copy()
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    v = hsv[:, :, 2].astype(np.float32) / 255.0
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    mean_brightness = float(v.mean())        # 平均亮度，越大说明整体越亮
    contrast = float(gray.std() / 255.0)     # 对比度，越大说明明暗差异越明显
    entropy = calc_entropy(gray)             # 信息熵，越大说明图像细节/变化越丰富
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_map = np.sqrt(gx * gx + gy * gy)
    edge_strength = float(edge_map.mean() / 255.0)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    over_exposure = float(np.mean(v > 0.96))                 # 过曝比例，接近纯白的区域占比
    highlight_ratio = float(np.mean((v > 0.82) & (s < 0.28))) # 高光/反光比例
    dark_ratio = float(np.mean(v < 0.18))                     # 暗部比例
    snow_ratio = float(np.mean((v > 0.62) & (s < 0.35)))      # 疑似雪/白雾区域比例
    noise_est = float(np.mean(np.abs(cv2.Laplacian(gray, cv2.CV_32F))) / 255.0) # 粗略噪声估计
    return {
        "height": h,
        "width": w,
        "mean_brightness": mean_brightness,
        "contrast": contrast,
        "entropy": entropy,
        "edge_strength": edge_strength,
        "laplacian_var": lap_var,
        "over_exposure": over_exposure,
        "highlight_ratio": highlight_ratio,
        "dark_ratio": dark_ratio,
        "snow_ratio": snow_ratio,
        "noise_est": noise_est,
    }
# ------------------------------------------------------------
# 计算原图和增强图之间的对比指标。
# edge_keep_ratio：边缘保持比例，增强后边缘强度 / 原图边缘强度。
# ssim_to_original：结构相似度，越接近 1 表示增强图和原图结构越一致。
def calc_pair_metrics(original_bgr: np.ndarray, enhanced_bgr: np.ndarray) -> Dict[str, float]:
    if original_bgr.shape[:2] != enhanced_bgr.shape[:2]:
        enhanced_bgr = cv2.resize(enhanced_bgr, (original_bgr.shape[1], original_bgr.shape[0]))
    g0 = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
    gx0 = cv2.Sobel(g0, cv2.CV_32F, 1, 0, ksize=3)
    gy0 = cv2.Sobel(g0, cv2.CV_32F, 0, 1, ksize=3)
    gx1 = cv2.Sobel(g1, cv2.CV_32F, 1, 0, ksize=3)
    gy1 = cv2.Sobel(g1, cv2.CV_32F, 0, 1, ksize=3)
    edge0 = float(np.mean(np.sqrt(gx0 * gx0 + gy0 * gy0)) + 1e-6)
    edge1 = float(np.mean(np.sqrt(gx1 * gx1 + gy1 * gy1)))
    try:
        ssim_val = float(ssim(g0, g1, data_range=255))
    except Exception:
        ssim_val = np.nan
    return {
        "edge_keep_ratio": edge1 / edge0,
        "ssim_to_original": ssim_val,
    }
# ------------------------------------------------------------
# 判断图片退化程度。
# F_low：低照度程度，越大说明越暗。
# F_haze：雾/模糊程度，越大说明对比度和边缘越弱。
# F_ref：高光/反光程度，越大说明过亮区域越多。
# q_snow：雪天或白雾综合退化指标，越大越接近重雪/强白雾场景。
def degradation_factors(img_bgr: np.ndarray, ref_edge: float = 0.08, ref_contrast: float = 0.22) -> Dict[str, float]:
    m = calc_metrics(img_bgr)
    f_low = 0.6 * (1 - m["mean_brightness"]) + 0.4 * m["dark_ratio"]
    f_haze = 0.45 * max(0, 1 - m["contrast"] / (ref_contrast + 1e-6)) + \
             0.35 * max(0, 1 - m["edge_strength"] / (ref_edge + 1e-6)) + \
             0.20 * m["highlight_ratio"]
    f_ref = m["highlight_ratio"] + 0.5 * m["over_exposure"]
    q_snow = 0.35 * m["snow_ratio"] + 0.25 * m["over_exposure"] + \
             0.20 * max(0, 1 - m["edge_strength"] / (ref_edge + 1e-6)) + \
             0.20 * max(0, 1 - m["contrast"] / (ref_contrast + 1e-6))
    return {
        "F_low": float(np.clip(f_low, 0, 1)),
        "F_haze": float(np.clip(f_haze, 0, 1)),
        "F_ref": float(np.clip(f_ref, 0, 1)),
        "q_snow": float(np.clip(q_snow, 0, 1)),
        **m,
    }
# ------------------------------------------------------------
# 提取高光/反光区域。
# HSV 中 V 表示亮度，S 表示饱和度；高光通常是“亮度高、饱和度低”。
# 输出 mask：白色代表高光区域，黑色代表普通区域。
def highlight_mask(img_bgr: np.ndarray, v_thr: int = 215, s_thr: int = 70) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]
    mask = ((v > v_thr) & (s < s_thr)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))
    return mask
# ------------------------------------------------------------
# 提取疑似雪/白雾区域。
# 雪和白雾通常也是“亮度偏高、饱和度偏低”。
# 后面用这个 mask 保护雪区，避免增强过度导致雪地变灰或失真。
def snow_mask(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]
    mask = ((v > 160) & (s < 90)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask
# ------------------------------------------------------------
# CLAHE 局部对比度增强。
# 可以理解为“分区域拉开明暗差异”。这里只增强 LAB 颜色空间的 L 亮度通道，尽量不破坏颜色。
def clahe_lab(img_bgr: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)
# ------------------------------------------------------------
# Gamma 校正，用来调整整体明暗。
# gamma 较小时暗部会变亮；gamma 较大时画面会变暗。这里限制范围，防止调节过度。
def gamma_correction(img_bgr: np.ndarray, gamma: float) -> np.ndarray:
    gamma = max(0.25, min(2.5, gamma))
    inv = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img_bgr, table)
# ------------------------------------------------------------
# Retinex 亮度增强。
# 思想是模拟人眼视觉，同时保留整体亮度和局部细节。这里只处理 HSV 的 V 亮度通道，减少颜色偏移。
def retinex_v_channel(img_bgr: np.ndarray, sigma_list: Tuple[int, ...] = (15, 80, 250)) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    v = hsv[:, :, 2] + 1.0
    ret = np.zeros_like(v, dtype=np.float32)
    for sigma in sigma_list:
        blur = cv2.GaussianBlur(v, (0, 0), sigmaX=sigma, sigmaY=sigma) + 1.0
        ret += np.log(v) - np.log(blur)
    ret /= len(sigma_list)
    ret = cv2.normalize(ret, None, 0, 255, cv2.NORM_MINMAX)
    hsv[:, :, 2] = 0.55 * hsv[:, :, 2] + 0.45 * ret
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
# ------------------------------------------------------------
# 暗通道计算，是经典去雾算法的基础。
# 简单理解：无雾图像的小区域里通常有一个颜色通道很暗；有雾时这个规律会被破坏。
def dark_channel(img_bgr: np.ndarray, patch_size: int = 15) -> np.ndarray:
    min_channel = np.min(img_bgr, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    return cv2.erode(min_channel, kernel)
# ------------------------------------------------------------
# 估计大气光 A。
# 去雾模型需要估计雾光/环境光。这里从暗通道较亮的像素里找原图最亮点作为估计值。
def estimate_atmospheric_light(img_bgr: np.ndarray, dark: np.ndarray, top_percent: float = 0.001) -> np.ndarray:
    h, w = dark.shape
    n = max(1, int(h * w * top_percent))
    flat_dark = dark.ravel()
    flat_img = img_bgr.reshape(-1, 3)
    idx = np.argpartition(flat_dark, -n)[-n:]
    brightest = idx[np.argmax(np.sum(flat_img[idx], axis=1))]
    return flat_img[brightest].astype(np.float32)
# ------------------------------------------------------------
# 引导滤波。
# 用来平滑透射率图，同时尽量保留边缘，让去雾强度变化更自然，不在边缘处乱跳。
def guided_filter_gray(I: np.ndarray, p: np.ndarray, r: int = 40, eps: float = 1e-3) -> np.ndarray:
    mean_I = cv2.boxFilter(I, cv2.CV_32F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_32F, (r, r))
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_32F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = cv2.boxFilter(I * I, cv2.CV_32F, (r, r))
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, cv2.CV_32F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_32F, (r, r))
    return mean_a * I + mean_b
# ------------------------------------------------------------
# 暗通道先验去雾。
# omega：去雾强度，越大去雾越明显。
# t0：透射率下限，防止除数太小造成过亮或失真。
# patch_size：局部窗口大小。最后会和原图混合，避免去雾过猛。
def dehaze_dark_channel(img_bgr: np.ndarray, omega: float = 0.85, t0: float = 0.12, patch_size: int = 15) -> np.ndarray:
    I = img_bgr.astype(np.float32)
    dark = dark_channel(img_bgr, patch_size)
    A = estimate_atmospheric_light(img_bgr, dark)
    norm_img = I / (A.reshape(1, 1, 3) + 1e-6)
    dark_norm = dark_channel(np.clip(norm_img * 255, 0, 255).astype(np.uint8), patch_size) / 255.0
    t = 1 - omega * dark_norm
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    try:
        t = guided_filter_gray(gray, t.astype(np.float32), r=40, eps=1e-3)
    except Exception:
        t = cv2.GaussianBlur(t, (0, 0), 3)
    t = np.clip(t, t0, 1.0)
    J = (I - A.reshape(1, 1, 3)) / t[:, :, None] + A.reshape(1, 1, 3)
    J = np.clip(J, 0, 255).astype(np.uint8)
    return cv2.addWeighted(J, 0.75, img_bgr, 0.25, 0)
# ------------------------------------------------------------
# 抑制高光/反光。
# 先找出高光区域，再压低这些区域亮度；如果高光范围较小，就用 inpaint 修补，使画面更自然。
def suppress_highlight(img_bgr: np.ndarray) -> np.ndarray:
    mask = highlight_mask(img_bgr)
    if mask.mean() < 1:
        return img_bgr
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    m = mask > 0
    hsv[:, :, 2][m] = 0.72 * hsv[:, :, 2][m] + 45
    compressed = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    if np.mean(m) < 0.10:
        repaired = cv2.inpaint(compressed, mask, 3, cv2.INPAINT_TELEA)
        return cv2.addWeighted(repaired, 0.50, compressed, 0.50, 0)
    return compressed
# ------------------------------------------------------------
# 非锐化掩膜，用来增强清晰度。
# 原理是“原图 + 一部分原图与模糊图的差异”，从而突出边缘。
# amount 控制锐化强度，radius 控制模糊半径。
def unsharp_mask(img_bgr: np.ndarray, amount: float = 0.45, radius: float = 1.2) -> np.ndarray:
    blur = cv2.GaussianBlur(img_bgr, (0, 0), radius)
    sharp = cv2.addWeighted(img_bgr, 1 + amount, blur, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)
# ------------------------------------------------------------
# 保边平滑。
# 作用是降低噪声，同时尽量不把道路边缘、车道线等重要结构抹掉。
def edge_preserving_smooth(img_bgr: np.ndarray, sigma_s: int = 50, sigma_r: float = 0.25) -> np.ndarray:
    try:
        return cv2.edgePreservingFilter(img_bgr, flags=1, sigma_s=sigma_s, sigma_r=sigma_r)
    except Exception:
        return cv2.bilateralFilter(img_bgr, d=7, sigmaColor=50, sigmaSpace=50)
# ------------------------------------------------------------
# 清晰/普通场景增强主函数。
# 它不是只用一种方法，而是把四种结果加权融合：
# y_low 处理偏暗，y_haze 处理雾化，y_ref 处理反光，y_clahe 增强局部对比度。
# 权重由 F_low、F_haze、F_ref 自动决定，哪种问题更严重，对应方法权重就更大。
def enhance_problem1_clear(img_bgr: np.ndarray) -> np.ndarray:
    factors = degradation_factors(img_bgr)
    f_low, f_haze, f_ref = factors["F_low"], factors["F_haze"], factors["F_ref"]
    gamma = 0.75 + 0.35 * f_low
    y_low = retinex_v_channel(gamma_correction(img_bgr, gamma))
    y_haze = dehaze_dark_channel(img_bgr, omega=0.72 + 0.15 * f_haze, t0=0.18)
    y_ref = suppress_highlight(img_bgr)
    y_clahe = clahe_lab(img_bgr, clip_limit=2.0, tile_grid_size=8)
    weights = np.array([0.25 + f_low, 0.15 + f_haze, 0.10 + f_ref, 0.25], dtype=np.float32)
    weights = weights / weights.sum()
    merged = (weights[0] * y_low.astype(np.float32) +
              weights[1] * y_haze.astype(np.float32) +
              weights[2] * y_ref.astype(np.float32) +
              weights[3] * y_clahe.astype(np.float32))
    merged = np.clip(merged, 0, 255).astype(np.uint8)
    merged = unsharp_mask(merged, amount=0.25, radius=1.0)
    return merged
# ------------------------------------------------------------
# 轻度退化场景增强。
# 先做基础增强，再做弱去雾、局部对比度增强和轻微锐化，最后和原图混合，避免增强过度。
def enhance_light(img_bgr: np.ndarray) -> np.ndarray:
    base = enhance_problem1_clear(img_bgr)
    weak_dehaze = dehaze_dark_channel(base, omega=0.65, t0=0.22, patch_size=11)
    local = clahe_lab(weak_dehaze, clip_limit=1.8, tile_grid_size=8)
    sharp = unsharp_mask(local, amount=0.28, radius=1.0)
    return cv2.addWeighted(sharp, 0.72, img_bgr, 0.28, 0)
# ------------------------------------------------------------
# 重度退化场景增强。
# 流程更强：基础增强 -> 强去雾 -> 保边平滑 -> CLAHE -> 锐化 -> 高光抑制。
# 同时用 snow_mask 保护雪区，避免雪地被过度锐化或颜色变脏。
def enhance_heavy(img_bgr: np.ndarray) -> np.ndarray:
    base = enhance_problem1_clear(img_bgr)
    strong = dehaze_dark_channel(base, omega=0.82, t0=0.16, patch_size=15)
    smooth = edge_preserving_smooth(strong, sigma_s=60, sigma_r=0.25)
    local = clahe_lab(smooth, clip_limit=2.2, tile_grid_size=8)
    sharp = unsharp_mask(local, amount=0.38, radius=1.2)
    sharp = suppress_highlight(sharp)
    m = snow_mask(img_bgr).astype(np.float32) / 255.0
    m = cv2.GaussianBlur(m, (0, 0), 3)[:, :, None]
    out = (1 - m) * sharp.astype(np.float32) + m * (0.65 * local.astype(np.float32) + 0.35 * img_bgr.astype(np.float32))
    return np.clip(out, 0, 255).astype(np.uint8)
# ------------------------------------------------------------
# 渐进式增强控制函数，也就是整个增强系统的“调度器”。
# clear 直接走普通增强；light/heavy 会同时算轻度和重度增强结果，再按 q_snow 自动融合。
# g 越接近 0，越偏向轻度增强；g 越接近 1，越偏向重度增强。
def enhance_progressive(img_bgr: np.ndarray, scene: str, q_threshold: float = 0.46) -> Tuple[np.ndarray, Dict[str, float], str]:
    factors = degradation_factors(img_bgr)
    q = factors["q_snow"]
    if scene == "clear":
        return enhance_problem1_clear(img_bgr), factors, "problem1_clear"
    y_light = enhance_light(img_bgr)
    y_heavy = enhance_heavy(img_bgr)
    bias = 0.06 if scene == "heavy" else -0.04
    g = 1.0 / (1.0 + math.exp(-12.0 * (q + bias - q_threshold)))
    if scene == "light":
        g *= 0.65
    out = (1 - g) * y_light.astype(np.float32) + g * y_heavy.astype(np.float32)
    branch = "progressive_light" if g < 0.45 else "progressive_heavy"
    factors["gate_g"] = float(g)
    return np.clip(out, 0, 255).astype(np.uint8), factors, branch
# ------------------------------------------------------------
# 指标归一化。
# 不同指标范围不一样，归一化就是把它们统一变成 0~1，方便后面加权求综合分。
# negative_cols 表示“越小越好”的指标，比如过曝率、噪声估计。
def normalize_indicators(df: pd.DataFrame, indicator_cols: List[str], negative_cols: List[str]) -> pd.DataFrame:
    z = pd.DataFrame(index=df.index)
    for col in indicator_cols:
        x = df[col].astype(float)
        xmin, xmax = x.min(), x.max()
        if abs(xmax - xmin) < 1e-12:
            z[col] = 1.0
        elif col in negative_cols:
            z[col] = (xmax - x) / (xmax - xmin)
        else:
            z[col] = (x - xmin) / (xmax - xmin)
    return z
# ------------------------------------------------------------
# 熵权法计算指标权重。
# 简单理解：某个指标在不同图片之间差异越明显，说明它提供的信息越多，权重就越大。
def entropy_weights(z: pd.DataFrame) -> pd.Series:
    Z = z.clip(lower=0).astype(float)
    P = Z.div(Z.sum(axis=0) + 1e-12, axis=1)
    n = len(Z)
    if n <= 1:
        return pd.Series(1.0 / len(Z.columns), index=Z.columns)
    e = -(P * np.log(P + 1e-12)).sum(axis=0) / math.log(n)
    d = 1 - e
    if d.sum() <= 1e-12:
        return pd.Series(1.0 / len(Z.columns), index=Z.columns)
    return d / d.sum()
# ------------------------------------------------------------
# 计算综合质量分 Q_score。
# 先选择评价指标，再归一化，然后用熵权法自动算权重。
# Q_score 越高，表示综合增强效果越好，但它仍然是客观指标，不完全等同于人眼主观评价。
def add_quality_score(metrics_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    indicator_cols = [
        "entropy_enh", "contrast_enh", "edge_strength_enh", "laplacian_var_enh",
        "edge_keep_ratio", "ssim_to_original", "mean_brightness_balance",
        "over_exposure_enh", "noise_est_enh"
    ]
    negative_cols = ["over_exposure_enh", "noise_est_enh"]
    use_cols = [c for c in indicator_cols if c in metrics_df.columns]
    z = normalize_indicators(metrics_df, use_cols, negative_cols)
    weights = entropy_weights(z)
    scored = metrics_df.copy()
    scored["Q_score"] = (z * weights).sum(axis=1)
    weights_df = weights.rename("weight").reset_index().rename(columns={"index": "indicator"})
    return scored, weights_df
# ------------------------------------------------------------
# 生成增强前后对比图。
# 每个场景最多取 4 张样例，左边原图，右边增强图，保存到 output_dir/figures。
def make_comparison_figure(scene: str, pairs: List[Tuple[Path, np.ndarray, np.ndarray]], out_path: Path, max_items: int = 4) -> None:
    if not pairs:
        return
    pairs = pairs[:max_items]
    fig, axes = plt.subplots(len(pairs), 2, figsize=(8, 3 * len(pairs)))
    if len(pairs) == 1:
        axes = np.array([axes])
    for i, (path, orig, enh) in enumerate(pairs):
        orig_rgb = cv2.cvtColor(cv2.resize(orig, (360, 240)), cv2.COLOR_BGR2RGB)
        enh_rgb = cv2.cvtColor(cv2.resize(enh, (360, 240)), cv2.COLOR_BGR2RGB)
        axes[i, 0].imshow(orig_rgb)
        axes[i, 0].set_title(f"{scene} Original\n{path.name}", fontsize=9)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(enh_rgb)
        axes[i, 1].set_title(f"{scene} Enhanced", fontsize=9)
        axes[i, 1].axis("off")
    plt.tight_layout()
    safe_makedirs(out_path.parent)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
# ------------------------------------------------------------
# 处理一个场景下的所有图片。
# 主要流程：收集图片 -> 读图 -> 必要时缩放 -> 增强 -> 保存增强图 -> 计算指标 -> 记录结果。
# 返回 rows 是指标数据；examples 是前几张样例图，用来画对比图。
def process_scene(scene: str, roots: List[Path], output_dir: Path, max_images: Optional[int], resize_long_side: int) -> Tuple[List[Dict], List[Tuple[Path, np.ndarray, np.ndarray]]]:
    all_files: List[Path] = []
    for root in roots:
        if root and root.exists():
            all_files.extend(collect_images(root, scene))
    seen = set()
    files = []
    for f in all_files:
        if str(f) not in seen:
            files.append(f)
            seen.add(str(f))
    if max_images is not None and max_images > 0:
        files = files[:max_images]
    rows: List[Dict] = []
    examples: List[Tuple[Path, np.ndarray, np.ndarray]] = []
    for p in tqdm(files, desc=f"处理 {scene}"):
        img = imread_unicode(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = 1.0
        if resize_long_side and max(h, w) > resize_long_side:
            scale = resize_long_side / max(h, w)
            img_work = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            img_work = img
        enh_work, factors, branch = enhance_progressive(img_work, scene)
        if img_work.shape[:2] != img.shape[:2]:
            enh = cv2.resize(enh_work, (w, h), interpolation=cv2.INTER_CUBIC)
        else:
            enh = enh_work
        rel_name = f"{scene}/{infer_split(p)}/{p.stem}_enhanced.jpg"
        out_img_path = output_dir / "enhanced" / rel_name
        imwrite_unicode(out_img_path, enh)
        m0 = calc_metrics(img)
        m1 = calc_metrics(enh)
        pm = calc_pair_metrics(img, enh)
        brightness_balance = 1 - abs(m1["mean_brightness"] - 0.55) / 0.55
        row = {
            "scene": scene,
            "split": infer_split(p),
            "file": str(p),
            "output_file": str(out_img_path),
            "branch": branch,
            "q_snow": factors.get("q_snow", np.nan),
            "gate_g": factors.get("gate_g", 0.0 if scene == "clear" else np.nan),
            "mean_brightness_orig": m0["mean_brightness"],
            "mean_brightness_enh": m1["mean_brightness"],
            "mean_brightness_balance": float(np.clip(brightness_balance, 0, 1)),
            "contrast_orig": m0["contrast"],
            "contrast_enh": m1["contrast"],
            "entropy_orig": m0["entropy"],
            "entropy_enh": m1["entropy"],
            "edge_strength_orig": m0["edge_strength"],
            "edge_strength_enh": m1["edge_strength"],
            "laplacian_var_orig": m0["laplacian_var"],
            "laplacian_var_enh": m1["laplacian_var"],
            "over_exposure_orig": m0["over_exposure"],
            "over_exposure_enh": m1["over_exposure"],
            "snow_ratio_orig": m0["snow_ratio"],
            "snow_ratio_enh": m1["snow_ratio"],
            "noise_est_orig": m0["noise_est"],
            "noise_est_enh": m1["noise_est"],
            **pm,
        }
        rows.append(row)
        if len(examples) < 4:
            examples.append((p, img, enh))
    return rows, examples
# ------------------------------------------------------------
# 按场景汇总指标。
# 把每张图片的指标按 clear/light/heavy 分组，计算均值、标准差、数量，适合写进实验结果分析部分。
def summarize(scored: pd.DataFrame) -> pd.DataFrame:
    agg_cols = [
        "mean_brightness_orig", "mean_brightness_enh", "contrast_orig", "contrast_enh",
        "entropy_orig", "entropy_enh", "edge_strength_orig", "edge_strength_enh",
        "over_exposure_orig", "over_exposure_enh", "edge_keep_ratio", "ssim_to_original", "Q_score", "q_snow"
    ]
    valid = [c for c in agg_cols if c in scored.columns]
    return scored.groupby("scene")[valid].agg(["mean", "std", "count"]).reset_index()
# ------------------------------------------------------------
# 主函数：程序从这里开始执行。
# 负责读取参数、准备输出目录、处理三个场景、保存增强图片、指标 CSV、权重表和对比图。
def main() -> None:
    parser = argparse.ArgumentParser(description="A题道路图像增强与评价代码")

    # data_dir：如果数据集已经解压成文件夹，就填写这个文件夹路径。
    parser.add_argument("--data_dir", type=str, default="")

    # clear_zip/light_zip/heavy_zip：如果数据集还是压缩包，就分别填写三个场景的 zip 路径。
    parser.add_argument("--clear_zip", type=str, default="")
    parser.add_argument("--light_zip", type=str, default="")
    parser.add_argument("--heavy_zip", type=str, default="")

    # output_dir：输出目录，必须填写。增强图片、指标表格、对比图都会保存到这里。
    parser.add_argument("--output_dir", type=str, required=True)

    # max_images_per_scene：每个场景最多处理多少张图片。0 表示不限制，全部处理。
    parser.add_argument("--max_images_per_scene", type=int, default=0)

    # resize_long_side：处理时把长边缩放到这个数值以内，能加快速度并减少内存占用。
    parser.add_argument("--resize_long_side", type=int, default=960)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    safe_makedirs(output_dir)
    extract_root = output_dir / "extracted"
    safe_makedirs(extract_root)
    roots: Dict[str, List[Path]] = {"clear": [], "light": [], "heavy": []}
    if args.data_dir:
        data_root = Path(args.data_dir)
        if data_root.exists():
            for scene in roots:
                roots[scene].append(data_root)
    for scene, zp in [("clear", args.clear_zip), ("light", args.light_zip), ("heavy", args.heavy_zip)]:
        ext_root = extract_if_needed(zp, extract_root) if zp else None
        if ext_root:
            roots[scene].append(ext_root)
    max_images = None if args.max_images_per_scene == 0 else args.max_images_per_scene
    all_rows = []
    for scene in ["clear", "light", "heavy"]:
        rows, examples = process_scene(scene, roots[scene], output_dir, max_images, args.resize_long_side)
        all_rows.extend(rows)
        make_comparison_figure(scene, examples, output_dir / "figures" / f"{scene}_examples.png")
    if not all_rows:
        raise RuntimeError("没有找到可处理的图片，请检查数据路径或 zip 参数。")
    # 把所有图片的指标记录整理成表格
    df = pd.DataFrame(all_rows)

    # 计算每张图片的综合质量分，以及各指标的熵权法权重
    scored, weights = add_quality_score(df)

    # 按场景生成汇总统计表
    summary = summarize(scored)
    scored.to_csv(output_dir / "metrics_by_image.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    weights.to_csv(output_dir / "weights_entropy.csv", index=False, encoding="utf-8-sig")
    print("\n运行完成。主要输出：")
    print(f"1. 增强图片：{output_dir / 'enhanced'}")
    print(f"2. 每张图片指标：{output_dir / 'metrics_by_image.csv'}")
    print(f"3. 场景汇总指标：{output_dir / 'metrics_summary.csv'}")
    print(f"4. 熵权法权重：{output_dir / 'weights_entropy.csv'}")
    print(f"5. 样例对比图：{output_dir / 'figures'}")
if __name__ == "__main__":
    main()


