# -*- coding: utf-8 -*-
"""
图1 重设计 v2 —— 汛期停测分两个时段分别绘制
- 时段一: 2026-06-23 ~ 2026-07-06（7期），时段二: 2026-07-28 ~ 2026-08-03（7期）
- 54个点按断面聚合：每断面取各监测点累计位移的算术平均，作为该断面代表线
- 11个断面分色，图内无图号/子图号/AI标注，右下角留白可裁剪
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
from matplotlib.lines import Line2D

# 中文字体
fontManager.addfont('./data')
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

BASE = r'project\学习\智能导航\省赛\算法与图表'
OUT = BASE

# ============ 加载数据 ============
wide = pd.read_csv(os.path.join(BASE, '位移宽表_坡顶水平位移.csv'),
                   encoding='utf-8-sig', index_col=0)
wide.index = pd.to_datetime(wide.index).sort_values()
wide = wide.sort_index()

# 断面定义：点号 -> (断面号, 桩号)
SEC_MAP = {}
SEC_NAMES = {
    1: 'K1+840～K1+940', 3: 'K4+140～K4+280', 4: 'K5+500～K5+600',
    5: 'K11+252～K11+360', 6: 'K12+700', 7: 'K15+830～K16+020',
    8: 'K16+090～K16+233', 9: 'K16+540～K16+620', 10: 'K19+600',
    11: 'K22+210～K22+410', 12: 'K25+100～K25+200',
}
for c in wide.columns:
    sec = int(c.split('-')[0])
    SEC_MAP[c] = (sec, SEC_NAMES[sec])

# 断面聚合：每断面取各监测点平均
SECS = sorted(SEC_MAP.values(), key=lambda x: x[0])
SEC_COLORS = ['#1f77b4', '#2ca02c', '#17becf', '#9467bd', '#8c564b',
              '#e377c2', '#7f7f7f', '#bcbd22', '#d62728', '#ff7f0e', '#393b79']
sec_avg = {}
for c in wide.columns:
    sec, _ = SEC_MAP[c]
    if sec not in sec_avg:
        sec_avg[sec] = []
    sec_avg[sec].append(c)

sec_rep = {}
for sec in sec_avg:
    pts = sec_avg[sec]
    mean_row = wide[pts].mean(axis=1)
    sec_rep[sec] = mean_row

# 时段分割
dates = list(wide.index)
p1_idx = [i for i, d in enumerate(dates) if pd.Timestamp(d) <= pd.Timestamp('2026-07-06')]
p2_idx = [i for i, d in enumerate(dates) if pd.Timestamp(d) >= pd.Timestamp('2026-07-28')]
p1_dates = [dates[i] for i in p1_idx]
p2_dates = [dates[i] for i in p2_idx]
p1_xticks = np.arange(len(p1_dates))
p2_xticks = np.arange(len(p2_dates))

print(f'时段一: {p1_dates[0].date()} ~ {p1_dates[-1].date()} ({len(p1_dates)}期)')
print(f'时段二: {p2_dates[0].date()} ~ {p2_dates[-1].date()} ({len(p2_dates)}期)')

# ============ 绘图辅助函数 ============
def setup_ax(ax, ylim, ylabel, title_text, xticks, dates_list):
    ax.grid(True, which='major', axis='both', color='#d9d9d9', ls='--', lw=0.7,
            alpha=0.85, zorder=0)
    ax.set_xticks(xticks)
    ax.set_xticklabels([d.strftime('%m-%d') for d in dates_list],
                        rotation=40, ha='right', fontsize=10.5)
    ax.set_xlim(-0.6, xticks[-1] + 0.6)
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_yticks(np.arange(ylim[1], ylim[0] - 0.5, -2))
    ax.set_xlabel('观测日期（2026 年，按有效观测期连续排列）', fontsize=12, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title_text, fontsize=16, fontweight='bold', pad=12)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.spines['left'].set_color('0.45')
    ax.spines['bottom'].set_color('0.45')
    ax.tick_params(axis='y', labelsize=11)

# ============ 图 A：时段一 6.23 - 7.06 ============
fig_a = plt.figure(figsize=(11.0, 6.6), dpi=180, facecolor='white')
ax_a = fig_a.add_axes([0.075, 0.12, 0.82, 0.82])   # 右下留白

for i, (sec, color) in enumerate(zip(sec_rep, SEC_COLORS)):
    vals = [sec_rep[sec].iloc[j] for j in p1_idx if not np.isnan(sec_rep[sec].iloc[j])]
    ax_a.plot(p1_xticks, vals, color=color, lw=1.6, marker='o', ms=4.0,
              mfc='white', mec=color, mew=1.0, zorder=2,
              label=f'{sec}号断面（{SEC_NAMES[sec]}）')

ax_a.legend(loc='upper right', fontsize=10, ncol=2, frameon=True, framealpha=0.96,
            edgecolor='0.7', handlelength=1.8, labelspacing=0.3, columnspacing=1.2,
            borderpad=0.6)
setup_ax(ax_a, (-16.0, 1.2), '累计水平位移 (mm)',
         '时段一：2026 年 6 月 23 日 ～ 7 月 6 日（汛期）',
         p1_xticks, p1_dates)

fp_a = os.path.join(OUT, '图1_A_时段一_6月23日至7月6日.png')
fig_a.savefig(fp_a, facecolor='white')
plt.close(fig_a)
print('saved A:', fp_a)

# ============ 图 B：时段二 7.28 - 8.03 ============
fig_b = plt.figure(figsize=(11.0, 6.6), dpi=180, facecolor='white')
ax_b = fig_b.add_axes([0.075, 0.12, 0.82, 0.82])

for i, (sec, color) in enumerate(zip(sec_rep, SEC_COLORS)):
    vals = [sec_rep[sec].iloc[j] for j in p2_idx if not np.isnan(sec_rep[sec].iloc[j])]
    ax_b.plot(p2_xticks, vals, color=color, lw=1.6, marker='o', ms=4.0,
              mfc='white', mec=color, mew=1.0, zorder=2,
              label=f'{sec}号断面（{SEC_NAMES[sec]}）')

ax_b.legend(loc='upper right', fontsize=10, ncol=2, frameon=True, framealpha=0.96,
            edgecolor='0.7', handlelength=1.8, labelspacing=0.3, columnspacing=1.2,
            borderpad=0.6)
setup_ax(ax_b, (-16.0, 1.2), '累计水平位移 (mm)',
         '时段二：2026 年 7 月 28 日 ～ 8 月 3 日（汛期）',
         p2_xticks, p2_dates)

fp_b = os.path.join(OUT, '图1_B_时段二_7月28日至8月3日.png')
fig_b.savefig(fp_b, facecolor='white')
plt.close(fig_b)
print('saved B:', fp_b)

# ============ 汇总信息 ============
print('\n--- 断面级聚合值（时段一）---')
for sec, color in zip(sec_rep, SEC_COLORS):
    vals = [sec_rep[sec].iloc[j] for j in p1_idx if not np.isnan(sec_rep[sec].iloc[j])]
    print(f'  {sec}号: {vals[0]:>6.1f}  →  {vals[-1]:>6.1f}  变化 {vals[-1]-vals[0]:>+6.1f} mm')
print('\n--- 断面级聚合值（时段二）---')
for sec, color in zip(sec_rep, SEC_COLORS):
    vals = [sec_rep[sec].iloc[j] for j in p2_idx if not np.isnan(sec_rep[sec].iloc[j])]
    print(f'  {sec}号: {vals[0]:>6.1f}  →  {vals[-1]:>6.1f}  变化 {vals[-1]-vals[0]:>+6.1f} mm')
