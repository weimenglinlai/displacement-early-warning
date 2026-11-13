# -*- coding: utf-8 -*-
"""图4 卡尔曼-小波联合滤波去噪原理框图"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

OUT = r"project\学习\智能导航\省赛\算法与图表"

fig, ax = plt.subplots(figsize=(12.6, 12))
ax.set_xlim(0, 10); ax.set_ylim(0, 15.2); ax.axis('off')

C_IN, C_PRE, C_WAV, C_KF, C_OUT, C_THR = ('#e8f0fe', '#eaf7ea', '#fff4e5',
                                          '#f3e8fd', '#e8f0fe', '#fdecea')
E_IN, E_PRE, E_WAV, E_KF, E_OUT, E_THR = ('#1a73e8', '#188038', '#e8710a',
                                          '#8430ce', '#1a73e8', '#c5221f')


def box(x, y, w, h, text, fc, ec, fs=10.5, bold=False, rad=0.12):
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle=f"round,pad=0.06,rounding_size={rad}",
                       fc=fc, ec=ec, lw=1.6)
    ax.add_patch(p)
    ax.text(x, y, text, ha='center', va='center', fontsize=fs,
            fontweight='bold' if bold else 'normal', linespacing=1.55)


def arrow(x1, y1, x2, y2, color='#555', style='-|>', lw=1.8, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=16, lw=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))


# ---- 标题 ----
ax.text(5, 14.85, '卡尔曼-小波联合滤波去噪原理框图', ha='center',
        fontsize=15.5, fontweight='bold')
ax.text(5, 14.35, '（北斗 GNSS 边坡位移监测数据实时去噪与三级阈值预警链路）',
        ha='center', fontsize=10, color='#666')

# ---- 1 数据输入 ----
box(5, 13.3, 8.2, 0.85, '北斗/GNSS 多模监测终端 原始观测序列  z(k)\n'
                        '（载波相位 / 伪距 / 多路径误差 / 随机噪声）',
    C_IN, E_IN, 10.5)

# ---- 2 预处理 ----
arrow(5, 12.85, 5, 12.25)
box(5, 11.8, 8.2, 0.85, '数据预处理：粗差探测剔除 → 基准点转换 → 相对变形解算\n'
                        '生成累计位移观测序列 x(k)（采样间隔 Δt）',
    C_PRE, E_PRE, 10.5)

# ---- 3 小波支路 ----
arrow(5, 11.35, 5, 10.75)
box(5, 10.25, 9.2, 1.0, '① 小波多尺度分解（db5 小波，N 层离散小波变换）\n'
                        r'$c = [a_N,\ d_N,\ d_{N-1},\ \ldots,\ d_1]$' '   （近似系数 + 各层细节系数）',
    C_WAV, E_WAV, 10.5)

arrow(5, 9.72, 5, 9.15)
box(5, 8.65, 9.2, 1.0, '② 细节系数软阈值量化（通用阈值 λ = α·σ·√(2lnN)，σ = MAD/0.6745）\n'
                       r'$\tilde{d}_j = \mathrm{sgn}(d_j)\cdot\max(|d_j|-\lambda,\ 0)$   —— 压制高频噪声、保留特征奇异点',
    C_WAV, E_WAV, 10.2)

arrow(5, 8.12, 5, 7.55)
box(5, 7.05, 9.2, 0.85, '③ 小波重构 → 一级去噪序列  x_w(k)',
    C_WAV, E_WAV, 11)

# ---- 4 卡尔曼支路 ----
arrow(5, 6.6, 5, 6.0)
box(5, 5.5, 9.5, 1.45, '④ 卡尔曼最优估计（状态方程 ' r'$x_k = x_{k-1} + w_k$' '，观测方程 ' r'$z_k = x_k + v_k$' '）\n'
                       r'状态预测：$\hat{x}(k|k-1) = \hat{x}(k-1)$　$P(k|k-1) = P(k-1) + Q$' '\n'
                       r'增益与更新：$K = P/(P+R)$，$\hat{x}(k) = \hat{x}(k|k-1) + K\cdot[z_k - \hat{x}(k|k-1)]$',
    C_KF, E_KF, 9.4)

# ---- 5 输出 ----
arrow(5, 4.9, 5, 4.32)
box(5, 3.85, 8.2, 0.85, '联合滤波输出：去噪位移序列  ' r'$\hat{x}(k)$' '\n'
                        '（保留真实变形趋势，抑制随机噪声与伪速率波动）',
    C_OUT, E_OUT, 10.5)

# ---- 6 速率 ----
arrow(5, 3.42, 5, 2.85)
box(5, 2.4, 8.2, 0.8, '位移速率计算　' r'$v(k) = [\hat{x}(k) - \hat{x}(k-1)]/\Delta t$',
    C_OUT, E_OUT, 11)

# ---- 7 阈值 ----
arrow(5, 2.0, 5, 1.42)
box(5, 0.95, 9.2, 0.85, '三级阈值判据：蓝色·三级(注意) ≥ 1/3 容许值　|　'
                        '黄色·二级(预警) ≥ 2/3　|　红色·一级(报警) ≥ 容许值',
    C_THR, E_THR, 10.2)
arrow(5, 0.52, 5, 0.18)
ax.text(5, 0.02, '→ 分级预警发布 / 加密观测/ 响应处置', ha='center',
        fontsize=11, fontweight='bold', color='#c5221f')

fp = os.path.join(OUT, "图4_卡尔曼-小波联合滤波原理框图.png")
plt.savefig(fp, bbox_inches='tight', facecolor='white')
plt.close()
from PIL import Image as _Img                                  # 右下角补白：预留可裁剪空白
_im = _Img.open(fp).convert('RGB'); _w, _h = _im.size
_cv = _Img.new('RGB', (int(_w * 1.25), int(_h * 1.25)), 'white'); _cv.paste(_im, (0, 0)); _cv.save(fp)
print("saved:", fp)
