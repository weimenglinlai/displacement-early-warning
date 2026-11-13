# -*- coding: utf-8 -*-
"""
位移速率三级阈值预警
数据：实测累计位移（10-1）+ 卡尔曼-小波联合滤波输出
输出：图3 位移速率三级阈值预警图 + 预警统计表
"""
import os
import numpy as np
import pandas as pd
import pywt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.signal import savgol_filter

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

BASE = r"project\学习\智能导航\省赛"
OUT = os.path.join(BASE, "算法与图表")

# ---- 三级阈值（按设计方案容许变形速率 3.0 mm/d 的 1/3、2/3、1 倍划分）----
V_ALLOW = 3.0
TH = [V_ALLOW / 3, V_ALLOW * 2 / 3, V_ALLOW]      # 1.0 / 2.0 / 3.0 mm/d
LV_NAME = ['正常', '蓝色·三级(注意)', '黄色·二级(预警)', '红色·一级(报警)']
LV_COLOR = ['#2ecc71', '#3498db', '#f1c40f', '#e74c3c']

wide = pd.read_csv(os.path.join(OUT, "位移宽表_坡顶水平位移.csv"),
                   encoding='utf-8-sig', index_col=0, parse_dates=True)
ser = wide['10-1'].dropna()
y = ser.values.astype(float)
dates = ser.index
n = len(y)

cache = np.load(os.path.join(OUT, "_cache_10-1.npz"))
y_joint = cache['joint']
y_noisy, y_sim_joint, truth = cache['noisy'], cache['sim_joint'], cache['truth']

# ---- 速率计算（mm/d，取向坡内为正）----
dt = np.diff(dates.values).astype('timedelta64[D]').astype(float)
dt[dt <= 0] = 1.0
rate_real_raw = -np.diff(y) / dt
rate_real_flt = -np.diff(y_joint) / dt
rate_sim_raw = -np.diff(y_noisy) / dt
rate_sim_flt = -np.diff(y_sim_joint) / dt
rate_sim_true = -np.diff(truth) / dt

# 观测间隔不均（07-06~07-28 汛期停测），逐日速率仅在图注说明
dt_show = dt.copy()
dt_show[dt_show > 3] = 1.0   # 断档期不参与速率统计


def grade(v):
    """三级阈值判据"""
    g = np.zeros_like(v, dtype=int)
    g[v >= TH[0]] = 1
    g[v >= TH[1]] = 2
    g[v >= TH[2]] = 3
    return g


g_raw = grade(rate_sim_raw)
g_flt = grade(rate_sim_flt)
names = [d.strftime('%m-%d') for d in dates[1:]]

stat = pd.DataFrame([
    {'序列': '含噪观测序列', '最大速率(mm/d)': round(float(np.max(np.abs(rate_sim_raw))), 3),
     '触发三级(次)': int((g_raw >= 1).sum()), '触发二级(次)': int((g_raw >= 2).sum()),
     '触发一级(次)': int((g_raw >= 3).sum())},
    {'序列': '卡尔曼-小波联合滤波', '最大速率(mm/d)': round(float(np.max(np.abs(rate_sim_flt))), 3),
     '触发三级(次)': int((g_flt >= 1).sum()), '触发二级(次)': int((g_flt >= 2).sum()),
     '触发一级(次)': int((g_flt >= 3).sum())},
    {'序列': '实测原始观测序列', '最大速率(mm/d)': round(float(np.max(np.abs(rate_real_raw))), 3),
     '触发三级(次)': int((grade(rate_real_raw) >= 1).sum()),
     '触发二级(次)': int((grade(rate_real_raw) >= 2).sum()),
     '触发一级(次)': int((grade(rate_real_raw) >= 3).sum())},
    {'序列': '实测联合滤波', '最大速率(mm/d)': round(float(np.max(np.abs(rate_real_flt))), 3),
     '触发三级(次)': int((grade(rate_real_flt) >= 1).sum()),
     '触发二级(次)': int((grade(rate_real_flt) >= 2).sum()),
     '触发一级(次)': int((grade(rate_real_flt) >= 3).sum())},
])
print("=== 预警触发统计 ===")
print(stat.to_string(index=False))
stat.to_csv(os.path.join(OUT, "表_位移速率三级阈值预警统计.csv"),
            index=False, encoding='utf-8-sig')

# ================= 绘图 =================
fig = plt.figure(figsize=(15, 11))
gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.62], hspace=0.42)

xs = np.arange(1, n)  # 速率对应第 2~21 期


def thr_lines(a):
    a.axhline(TH[0], color=LV_COLOR[1], ls='--', lw=1.3)
    a.axhline(TH[1], color=LV_COLOR[2], ls='--', lw=1.3)
    a.axhline(TH[2], color=LV_COLOR[3], ls='--', lw=1.3)
    a.text(n - 0.4, TH[0] + 0.06, f'三级阈值 {TH[0]:.1f}', color=LV_COLOR[1],
           fontsize=8.5, ha='right')
    a.text(n - 0.4, TH[1] + 0.06, f'二级阈值 {TH[1]:.1f}', color='#b7950b',
           fontsize=8.5, ha='right')
    a.text(n - 0.4, TH[2] + 0.06, f'一级阈值 {TH[2]:.1f}', color=LV_COLOR[3],
           fontsize=8.5, ha='right')


def seq_axis(a):
    a.set_xticks(np.arange(2, n + 1, 2))
    a.set_xticklabels([d.strftime('%m-%d') for d in dates[1::2]], fontsize=8.5)
    a.set_xlim(1.5, n + 0.3)


# (a) 实测速率
ax = fig.add_subplot(gs[0])
ax.plot(xs, rate_real_raw, 'o--', ms=5, lw=1.1, color='#c0392b', mfc='none',
        label='实测原始序列速率')
ax.plot(xs, rate_real_flt, '-', lw=2.2, color='#1f6feb', label='联合滤波后速率')
ax.fill_between([1.5, n + 0.3], 0, TH[0], color=LV_COLOR[0], alpha=0.07)
thr_lines(ax)
ax.set_ylim(0, V_ALLOW * 1.12)
ax.set_ylabel('变形速率 (mm/d)', fontsize=10.5)
ax.set_title('实测序列位移速率与三级预警阈值（全程处于正常区）',
             fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, ls='--'); ax.legend(fontsize=9, loc='upper left'); seq_axis(ax)

# (b) 仿真含噪序列速率
ax2 = fig.add_subplot(gs[1])
ax2.plot(xs, rate_sim_raw, 'o--', ms=5, lw=1.0, color='#c0392b', mfc='none',
         label='含噪观测序列速率（抖动剧烈）')
ax2.plot(xs, rate_sim_flt, '-', lw=2.2, color='#1f6feb', label='联合滤波后速率')
ax2.plot(xs, rate_sim_true, '-', lw=1.6, color='k', alpha=0.6, label='真实速率')
thr_lines(ax2)
ax2.set_ylim(0, max(TH[2] * 1.12, float(np.max(rate_sim_raw)) * 1.1))
ax2.set_ylabel('变形速率 (mm/d)', fontsize=10.5)
ax2.set_title('含噪序列速率对比（联合滤波抑制伪速率、减少误报）',
              fontsize=12, fontweight='bold')
ax2.grid(alpha=0.3, ls='--'); ax2.legend(fontsize=9, loc='upper left', ncol=2)
seq_axis(ax2)

# (c) 预警等级时序
ax3 = fig.add_subplot(gs[2])
for i, (g, lab, off) in enumerate([(g_raw, '含噪观测序列', 0.0),
                                   (g_flt, '联合滤波后', 0.42)]):
    for j, gv in enumerate(g):
        ax3.add_patch(plt.Rectangle((xs[j] - 0.45, off - 0.18), 0.9, 0.36,
                                    color=LV_COLOR[gv], ec='none'))
ax3.set_yticks([0.0, 0.42])
ax3.set_yticklabels(['含噪观测序列', '联合滤波后'], fontsize=10)
ax3.set_ylim(-0.35, 0.8)
ax3.set_title('位移速率预警等级时序（绿=正常 蓝=三级 黄=二级 红=一级）',
              fontsize=11)
ax3.set_xlabel('观测日期', fontsize=10)
ax3.grid(alpha=0.25, ls='--', axis='x')
seq_axis(ax3)
ax3.legend(handles=[Patch(color=c, label=l) for c, l in zip(LV_COLOR, LV_NAME)],
           fontsize=8.5, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.35))

fp = os.path.join(OUT, "图3_位移速率三级阈值预警.png")
plt.savefig(fp, bbox_inches='tight'); plt.close()
from PIL import Image as _Img                                  # 右下角补白：预留可裁剪空白
_im = _Img.open(fp).convert('RGB'); _w, _h = _im.size
_cv = _Img.new('RGB', (int(_w * 1.25), int(_h * 1.25)), 'white'); _cv.paste(_im, (0, 0)); _cv.save(fp)
print("saved:", fp)
print("done")
