# -*- coding: utf-8 -*-
"""
图1 重绘 v3 —— 放大右下角留白、图例移至右下不挡线、标题正式化
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
from matplotlib.lines import Line2D

fontManager.addfont('./data')
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

BASE = r'project\学习\智能导航\省赛\算法与图表'
OUT = BASE

wide = pd.read_csv(os.path.join(BASE, '位移宽表_坡顶水平位移.csv'),
                   encoding='utf-8-sig', index_col=0)
wide.index = pd.to_datetime(wide.index).sort_values()

SEC_NAMES = {
    1: 'K1+840～K1+940', 3: 'K4+140～K4+280', 4: 'K5+500～K5+600',
    5: 'K11+252～K11+360', 6: 'K12+700', 7: 'K15+830～K16+020',
    8: 'K16+090～K16+233', 9: 'K16+540～K16+620', 10: 'K19+600',
    11: 'K22+210～K22+410', 12: 'K25+100～K25+200',
}
SEC_COLORS = ['#1f77b4', '#2ca02c', '#17becf', '#9467bd', '#8c564b',
              '#e377c2', '#7f7f7f', '#bcbd22', '#d62728', '#ff7f0e', '#393b79']

sec_avg = {}
for c in wide.columns:
    sec = int(c.split('-')[0])
    sec_avg.setdefault(sec, []).append(c)

sec_rep = {}
for sec in sec_avg:
    sec_rep[sec] = wide[sec_avg[sec]].mean(axis=1)

p1_idx = [i for i, d in enumerate(wide.index) if d <= pd.Timestamp('2026-07-06')]
p2_idx = [i for i, d in enumerate(wide.index) if d >= pd.Timestamp('2026-07-28')]
p1_dates = [wide.index[i] for i in p1_idx]
p2_dates = [wide.index[i] for i in p2_idx]
p1_xticks = np.arange(len(p1_dates))
p2_xticks = np.arange(len(p2_dates))

SEC_LIST = sorted(sec_rep.keys())

# ============ 绘图辅助 ============
def make_one(dates_list, xticks, period_str, fp, fig_size=(10.6, 6.6)):
    fig = plt.figure(figsize=fig_size, dpi=180, facecolor='white')
    # 放大右下角留白：右边距≈20%，底部边距≈20%，右下角空白区≈20%x20%
    fig.subplots_adjust(left=0.06, right=0.66, bottom=0.10, top=0.80, hspace=0.05)
    ax = fig.add_subplot(1, 1, 1)

    for i, sec in enumerate(SEC_LIST):
        vals = [sec_rep[sec].iloc[j] for j in range(len(dates_list)) if not np.isnan(sec_rep[sec].iloc[j])]
        ax.plot(xticks, vals, color=SEC_COLORS[i], lw=1.5, marker='o', ms=3.6,
                mfc='white', mec=SEC_COLORS[i], mew=0.9, zorder=2)

    ax.grid(True, which='major', axis='both', color='#d9d9d9', ls='--', lw=0.7, alpha=0.85, zorder=0)
    ax.set_xticks(xticks)
    ax.set_xticklabels([d.strftime('%m-%d') for d in dates_list], rotation=40, ha='right', fontsize=10)
    ax.set_xlim(-0.5, xticks[-1] + 0.5)
    ax.set_ylim(-16.0, 1.2)
    ax.set_yticks(np.arange(0, -15.1, -2))
    ax.set_xlabel('观测日期（2026 年，按有效观测期连续排列）', fontsize=12, labelpad=8)
    ax.set_ylabel('累计水平位移 (mm)', fontsize=12)
    ax.set_title(period_str, fontsize=16, fontweight='bold', pad=14)

    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.spines['left'].set_color('0.45')
    ax.spines['bottom'].set_color('0.45')
    ax.tick_params(axis='y', labelsize=10)

    # 图例置于右上角外部，不挡线
    leg = ax.legend(
        [f'{sec}号断面（{SEC_NAMES[sec]}）' for sec in SEC_LIST],
        loc='upper right', fontsize=8, ncol=2, frameon=True, framealpha=0.96,
        edgecolor='0.7', handlelength=1.6, labelspacing=0.15, columnspacing=0.7,
        borderpad=0.35,
        bbox_to_anchor=(1.01, 1.0),
    )
    leg.set_zorder(5)

    fig.savefig(fp, facecolor='white')
    plt.close(fig)

# ============ 图 A ============
make_one(p1_dates, p1_xticks,
         '坡顶水平位移监测曲线',
         os.path.join(OUT, '图1_A_时段一_6月23日至7月6日.png'))
print('saved A')

# ============ 图 B ============
make_one(p2_dates, p2_xticks,
         '坡顶水平位移监测曲线',
         os.path.join(OUT, '图1_B_时段二_7月28日至8月3日.png'))
print('saved B')
