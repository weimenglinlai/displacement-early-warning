# -*- coding: utf-8 -*-
"""
图1：基于绥宁草寨至洞口安顺公路边坡实测监测数据的真实位移-时间曲线
数据源：E:\\网页\\学习\\智能导航\\省赛\\监测数据.csv（由 4 份监测周报提取）
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

BASE = r"project\学习\智能导航\省赛"
OUT = os.path.join(BASE, "算法与图表")
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(os.path.join(BASE, "监测数据.csv"), encoding='utf-8-sig')
h = df[df['监测类型'] == '坡顶水平位移'].copy()
h['日期'] = pd.to_datetime(h['观测日期'], errors='coerce')
h['累计'] = pd.to_numeric(h['累计变化值(mm)'].replace('初始值', np.nan), errors='coerce')
h['累计'] = h['累计'].fillna(0.0)

wide = h.pivot_table(index='日期', columns='监测点号', values='累计', aggfunc='last').sort_index()
wide.to_csv(os.path.join(OUT, "位移宽表_坡顶水平位移.csv"), encoding='utf-8-sig')
print("宽表:", wide.shape)

# 各点最终累计位移排序（取绝对变形最大者）
final = wide.iloc[-1].abs().sort_values(ascending=False)
print("变形最大的10个点:\n", final.head(10))

# 代表点：变形量最大的 10 号断面 + 次大点
pts = ['10-1', '10-2', '11-1', '12-2']
reps = [p for p in pts if p in wide.columns]

# 断档区间
gap_start = pd.Timestamp('2026-07-07')
gap_end = pd.Timestamp('2026-07-27')

fig, axes = plt.subplots(2, 1, figsize=(13, 9), gridspec_kw={'height_ratios': [3, 1.4]})

# --- 上图：代表点累计位移 ---
ax = axes[0]
colors = ['#c0392b', '#e67e22', '#2980b9', '#27ae60']
for c, p in zip(colors, reps):
    s = wide[p].dropna()
    ax.plot(s.index, s.values, marker='o', ms=3.5, lw=1.4, color=c, label=f'监测点 {p}')

ax.axvspan(gap_start, gap_end, color='0.85', alpha=0.6, zorder=0)
ax.text(pd.Timestamp('2026-07-17'), -2, '无监测数据\n(07-07～07-27)', ha='center',
        va='center', fontsize=9, color='0.35')
ax.set_ylabel('累计水平位移 (mm)', fontsize=11)
ax.set_title('图1-a  K19+600 等断面坡顶水平位移累计变化-时间曲线（实测）', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, ls='--')
ax.legend(fontsize=9.5, loc='lower left')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=4))

# --- 下图：54点全体散点（反映整体变形分布） ---
ax2 = axes[1]
xall = np.tile(wide.index.values, wide.shape[1])
yall = wide.values.ravel()
ax2.scatter(xall, yall, s=6, c=yall, cmap='RdYlGn_r', alpha=0.75, edgecolors='none')
ax2.axvspan(gap_start, gap_end, color='0.85', alpha=0.6, zorder=0)
ax2.plot(wide.index, wide.abs().max(axis=1).values, color='#c0392b', lw=1.2,
         label='各期最大累计位移（包络）')
ax2.set_ylabel('累计位移 (mm)', fontsize=10)
ax2.set_xlabel('观测日期', fontsize=11)
ax2.set_title('图1-b  全部 54 个表面位移点实测分布', fontsize=11)
ax2.grid(alpha=0.3, ls='--')
ax2.legend(fontsize=9)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax2.xaxis.set_major_locator(mdates.DayLocator(interval=4))

plt.tight_layout()
fp1 = os.path.join(OUT, "图1_真实位移-时间曲线.png")
plt.savefig(fp1, bbox_inches='tight')
plt.close()
print("saved:", fp1)

# --- 单独每位移点导出（供后续滤波用）---
for p in wide.columns:
    s = wide[p].dropna()
    s.to_csv(os.path.join(OUT, f"点_{p}_累计位移.csv"), encoding='utf-8-sig')
print("done")
