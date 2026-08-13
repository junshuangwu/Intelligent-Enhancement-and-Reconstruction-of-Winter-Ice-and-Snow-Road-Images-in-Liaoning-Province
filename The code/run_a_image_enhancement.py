# -*- coding: utf-8 -*-
"""
A题：辽宁冬季冰雪道路图像重构的智能增强
一键运行版代码：问题一、问题二、问题三

功能：
1. 自动读取 clear / light / heavy 三类数据；支持已解压目录或 zip 文件。
2. 对 clear 执行问题一“低光-雾霾-反光”多因素融合增强。
3. 对 light/heavy 执行问题二“冰雪强度判别-递进式增强”。
4. 计算问题三综合评价指标，输出 metrics_by_image.csv、metrics_summary.csv、weights_entropy.csv。
5. 保存增强图片和每类样例对比图。

依赖：opencv-python、numpy、pandas、scikit-image、matplotlib、tqdm
运行示例：
python run_a_image_enhancement.py \
  --clear_zip "/mnt/data/clear all.zip" \
  --light_zip "/mnt/data/light all.zip" \
  --heavy_zip "/mnt/data/heavy all.zip" \
  --output_dir "/mnt/data/A题输出" \
  --max_images_per_scene 50
"""

from __future__ import annotations

import argparse
import os
import re
import math
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ==============================
# 一、文件读取与保存
# ==============================

def safe_makedirs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def imread_unicode(path: Path) -> Optional[np.ndarray]:
    """支持中文和空格路径的 OpenCV 读取。返回 BGR uint8 图像。"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def imwrite_unicode(path: Path, img: np.ndarray) -> bool:
    """支持中文和空格路径的 OpenCV 保存。"""
    safe_makedirs(path.parent)
    ext = path.suffix.lower() or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(str(path))
    return bool(ok)


def extract_if_needed(zip_path: Optional[str], extract_root: Path) -> Optional[Path]:
    """若提供 zip，则解压到 output/extracted/zip_stem；已存在则不重复解压。"""
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


def collect_images(root: Path, scene_name: str) -> List[Path]:
    """递归搜图。scene_name 仅用于排序时优先 test/val/text。"""
    files = [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]
    def key_func(p: Path):
        s = str(p).lower()
        # 数模论文建议优先跑 test/text，其次 val，最后 train；这样实验更像测试集评价。
        split_priority = 0 if ("test" in s or "text" in s) else (1 if "val" in s else 2)
        return (split_priority, str(p))
    return sorted(files, key=key_func)


def infer_split(path: Path) -> str:
    s = str(path).lower()
    if "test" in s or "text" in s:
        return "test"
    if "val" in s:
        return "val"
    if "train" in s:
        return "train"
    return "unknown"


# ==============================
# 二、基础图像指标
# ==============================

def to_float01(img_bgr: np.ndarray) -> np.ndarray:
    return img_bgr.astype(np.float32) / 255.0


def calc_entropy(gray_u8: np.ndarray) -> float:
    hist = cv2.calcHist([gray_u8], [0], None, [256], [0, 256]).ravel()
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def calc_metrics(img_bgr: np.ndarray) -> Dict[str, float]:
    """问题三常用指标：亮度、对比度、信息熵、边缘、过曝、雪/冰比例。"""
    img = img_bgr.copy()
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    v = hsv[:, :, 2].astype(np.float32) / 255.0
    s = hsv[:, :, 1].astype(np.float32) / 255.0

    mean_brightness = float(v.mean())
    contrast = float(gray.std() / 255.0)
    entropy = calc_entropy(gray)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_map = np.sqrt(gx * gx + gy * gy)
    edge_strength = float(edge_map.mean() / 255.0)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # 高光：高亮且低饱和；积雪/结冰：较高亮、低饱和、纹理弱。
    over_exposure = float(np.mean(v > 0.96))
    highlight_ratio = float(np.mean((v > 0.82) & (s < 0.28)))
    dark_ratio = float(np.mean(v < 0.18))
    snow_ratio = float(np.mean((v > 0.62) & (s < 0.35)))

    # 简化噪声估计：拉普拉斯绝对值均值，数值越大表示锐利或噪声越多。
    noise_est = float(np.mean(np.abs(cv2.Laplacian(gray, cv2.CV_32F))) / 255.0)

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


def calc_pair_metrics(original_bgr: np.ndarray, enhanced_bgr: np.ndarray) -> Dict[str, float]:
    """增强前后配对指标，用于边缘保持率、结构相似度。"""
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


# ==============================
# 三、退化因子与掩膜
# ==============================

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


def highlight_mask(img_bgr: np.ndarray, v_thr: int = 215, s_thr: int = 70) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]
    mask = ((v > v_thr) & (s < s_thr)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))
    return mask


def snow_mask(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]
    mask = ((v > 160) & (s < 90)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask


# ==============================
# 四、增强算法模块
# ==============================

def clahe_lab(img_bgr: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)


def gamma_correction(img_bgr: np.ndarray, gamma: float) -> np.ndarray:
    gamma = max(0.25, min(2.5, gamma))
    inv = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img_bgr, table)


def retinex_v_channel(img_bgr: np.ndarray, sigma_list: Tuple[int, ...] = (15, 80, 250)) -> np.ndarray:
    """简化 MSR：在 HSV 的 V 通道做多尺度 Retinex。"""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    v = hsv[:, :, 2] + 1.0
    ret = np.zeros_like(v, dtype=np.float32)
    for sigma in sigma_list:
        blur = cv2.GaussianBlur(v, (0, 0), sigmaX=sigma, sigmaY=sigma) + 1.0
        ret += np.log(v) - np.log(blur)
    ret /= len(sigma_list)
    ret = cv2.normalize(ret, None, 0, 255, cv2.NORM_MINMAX)
    # 与原 V 通道融合，避免过度失真。
    hsv[:, :, 2] = 0.55 * hsv[:, :, 2] + 0.45 * ret
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)


def dark_channel(img_bgr: np.ndarray, patch_size: int = 15) -> np.ndarray:
    min_channel = np.min(img_bgr, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    return cv2.erode(min_channel, kernel)


def estimate_atmospheric_light(img_bgr: np.ndarray, dark: np.ndarray, top_percent: float = 0.001) -> np.ndarray:
    h, w = dark.shape
    n = max(1, int(h * w * top_percent))
    flat_dark = dark.ravel()
    flat_img = img_bgr.reshape(-1, 3)
    idx = np.argpartition(flat_dark, -n)[-n:]
    brightest = idx[np.argmax(np.sum(flat_img[idx], axis=1))]
    return flat_img[brightest].astype(np.float32)


def guided_filter_gray(I: np.ndarray, p: np.ndarray, r: int = 40, eps: float = 1e-3) -> np.ndarray:
    """简化引导滤波，用于细化透射率。I,p 为 0-1 float。"""
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


def dehaze_dark_channel(img_bgr: np.ndarray, omega: float = 0.85, t0: float = 0.12, patch_size: int = 15) -> np.ndarray:
    """暗通道先验去雾，强度适中，防止冰雪区域被过度拉暗。"""
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
    # 防止过度去雾，和原图融合。
    J = np.clip(J, 0, 255).astype(np.uint8)
    return cv2.addWeighted(J, 0.75, img_bgr, 0.25, 0)


def suppress_highlight(img_bgr: np.ndarray) -> np.ndarray:
    mask = highlight_mask(img_bgr)
    if mask.mean() < 1:
        return img_bgr
    # 先压缩 V 通道高光，再轻微修复。
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    m = mask > 0
    hsv[:, :, 2][m] = 0.72 * hsv[:, :, 2][m] + 45
    compressed = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    # 对小面积高光用 inpaint 修复颜色，大面积雪区不强行修复，避免假纹理。
    if np.mean(m) < 0.10:
        repaired = cv2.inpaint(compressed, mask, 3, cv2.INPAINT_TELEA)
        return cv2.addWeighted(repaired, 0.50, compressed, 0.50, 0)
    return compressed


def unsharp_mask(img_bgr: np.ndarray, amount: float = 0.45, radius: float = 1.2) -> np.ndarray:
    blur = cv2.GaussianBlur(img_bgr, (0, 0), radius)
    sharp = cv2.addWeighted(img_bgr, 1 + amount, blur, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def edge_preserving_smooth(img_bgr: np.ndarray, sigma_s: int = 50, sigma_r: float = 0.25) -> np.ndarray:
    try:
        return cv2.edgePreservingFilter(img_bgr, flags=1, sigma_s=sigma_s, sigma_r=sigma_r)
    except Exception:
        return cv2.bilateralFilter(img_bgr, d=7, sigmaColor=50, sigmaSpace=50)


# ==============================
# 五、三个问题的模型实现
# ==============================

def enhance_problem1_clear(img_bgr: np.ndarray) -> np.ndarray:
    """问题一：clear 正常道路图像的低光、雾霾、反光多因素融合增强。"""
    factors = degradation_factors(img_bgr)
    f_low, f_haze, f_ref = factors["F_low"], factors["F_haze"], factors["F_ref"]

    # 低光分支：Retinex + 自适应 gamma。
    gamma = 0.75 + 0.35 * f_low
    y_low = retinex_v_channel(gamma_correction(img_bgr, gamma))

    # 雾霾分支：暗通道去雾。
    y_haze = dehaze_dark_channel(img_bgr, omega=0.72 + 0.15 * f_haze, t0=0.18)

    # 反光分支：高光压缩/修复。
    y_ref = suppress_highlight(img_bgr)

    # 局部对比增强。
    y_clahe = clahe_lab(img_bgr, clip_limit=2.0, tile_grid_size=8)

    # 根据退化程度归一化权重。
    weights = np.array([0.25 + f_low, 0.15 + f_haze, 0.10 + f_ref, 0.25], dtype=np.float32)
    weights = weights / weights.sum()
    merged = (weights[0] * y_low.astype(np.float32) +
              weights[1] * y_haze.astype(np.float32) +
              weights[2] * y_ref.astype(np.float32) +
              weights[3] * y_clahe.astype(np.float32))
    merged = np.clip(merged, 0, 255).astype(np.uint8)
    merged = unsharp_mask(merged, amount=0.25, radius=1.0)
    return merged


def enhance_light(img_bgr: np.ndarray) -> np.ndarray:
    """问题二：轻度薄雪/轻雾，温和增强，保护已有纹理。"""
    base = enhance_problem1_clear(img_bgr)
    weak_dehaze = dehaze_dark_channel(base, omega=0.65, t0=0.22, patch_size=11)
    local = clahe_lab(weak_dehaze, clip_limit=1.8, tile_grid_size=8)
    sharp = unsharp_mask(local, amount=0.28, radius=1.0)
    # 与原图适度融合，防止过增强。
    return cv2.addWeighted(sharp, 0.72, img_bgr, 0.28, 0)


def enhance_heavy(img_bgr: np.ndarray) -> np.ndarray:
    """问题二：重度厚雪/结冰/浓雾，强增强 + 边缘保持 + 高光控制。
    注：这是传统算法可运行版；论文中的 U-Net/Restormer 可作为深度学习改进分支。
    """
    base = enhance_problem1_clear(img_bgr)
    strong = dehaze_dark_channel(base, omega=0.82, t0=0.16, patch_size=15)
    smooth = edge_preserving_smooth(strong, sigma_s=60, sigma_r=0.25)
    local = clahe_lab(smooth, clip_limit=2.2, tile_grid_size=8)
    sharp = unsharp_mask(local, amount=0.38, radius=1.2)
    sharp = suppress_highlight(sharp)

    # 对雪/冰区域降低过度锐化权重，对非雪区域保留纹理。
    m = snow_mask(img_bgr).astype(np.float32) / 255.0
    m = cv2.GaussianBlur(m, (0, 0), 3)[:, :, None]
    out = (1 - m) * sharp.astype(np.float32) + m * (0.65 * local.astype(np.float32) + 0.35 * img_bgr.astype(np.float32))
    return np.clip(out, 0, 255).astype(np.uint8)


def enhance_progressive(img_bgr: np.ndarray, scene: str, q_threshold: float = 0.46) -> Tuple[np.ndarray, Dict[str, float], str]:
    """问题二递进模型：根据场景标签和 q_snow 自适应选择/融合增强分支。"""
    factors = degradation_factors(img_bgr)
    q = factors["q_snow"]
    if scene == "clear":
        return enhance_problem1_clear(img_bgr), factors, "problem1_clear"

    y_light = enhance_light(img_bgr)
    y_heavy = enhance_heavy(img_bgr)
    # 门控函数：q 越高，越偏向重度分支。heavy 场景额外提高门控。
    bias = 0.06 if scene == "heavy" else -0.04
    g = 1.0 / (1.0 + math.exp(-12.0 * (q + bias - q_threshold)))
    if scene == "light":
        g *= 0.65
    out = (1 - g) * y_light.astype(np.float32) + g * y_heavy.astype(np.float32)
    branch = "progressive_light" if g < 0.45 else "progressive_heavy"
    factors["gate_g"] = float(g)
    return np.clip(out, 0, 255).astype(np.uint8), factors, branch


# ==============================
# 六、综合评价：熵权法
# ==============================

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


def add_quality_score(metrics_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # 正向指标：越大越好；逆向指标：越小越好。
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


# ==============================
# 七、可视化输出
# ==============================

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


# ==============================
# 八、主流程
# ==============================

def process_scene(scene: str, roots: List[Path], output_dir: Path, max_images: Optional[int], resize_long_side: int) -> Tuple[List[Dict], List[Tuple[Path, np.ndarray, np.ndarray]]]:
    all_files: List[Path] = []
    for root in roots:
        if root and root.exists():
            all_files.extend(collect_images(root, scene))
    # 去重并截断
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
        # 明度平衡：接近 0.55 较理想，防止过暗或过亮。
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


def summarize(scored: pd.DataFrame) -> pd.DataFrame:
    agg_cols = [
        "mean_brightness_orig", "mean_brightness_enh", "contrast_orig", "contrast_enh",
        "entropy_orig", "entropy_enh", "edge_strength_orig", "edge_strength_enh",
        "over_exposure_orig", "over_exposure_enh", "edge_keep_ratio", "ssim_to_original", "Q_score", "q_snow"
    ]
    valid = [c for c in agg_cols if c in scored.columns]
    return scored.groupby("scene")[valid].agg(["mean", "std", "count"]).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="A题道路图像增强与评价代码")
    parser.add_argument("--data_dir", type=str, default="", help="已解压数据根目录，内部可含 clear/light/heavy")
    parser.add_argument("--clear_zip", type=str, default="", help="clear all.zip 路径")
    parser.add_argument("--light_zip", type=str, default="", help="light all.zip 路径")
    parser.add_argument("--heavy_zip", type=str, default="", help="heavy all.zip 路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--max_images_per_scene", type=int, default=0, help="每类最多处理多少张；0 表示全部")
    parser.add_argument("--resize_long_side", type=int, default=960, help="处理时最长边缩放到该值；0 表示不缩放")
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

    df = pd.DataFrame(all_rows)
    scored, weights = add_quality_score(df)
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
