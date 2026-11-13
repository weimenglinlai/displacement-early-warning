# -*- coding: utf-8 -*-
"""
图1 重设计（汛期停测空白段压缩，曲线连续绘制）
  产出①：各监测点真实位移-时间曲线
  产出②：10-1 号点累计水平位移曲线
数据源：project\学习\智能导航\省赛\算法与图表\位移宽表_坡顶水平位移.csv
        （由 project\学习\智能导航\省赛\监测数据.csv 中"坡顶水平位移"实测记录透视而来）
设计要点：
  * 时间轴按有效观测期连续排列（06-23~06-29、06-30~07-06、07-28~08-03 共 21 期），
    07-07~07-27 汛期停测段不占时间轴宽度，曲线无空白间隔；
  * 图内不标注图号 / a、b 子图号 / AI 生成字样；
  * 图形右下角保留空白可裁剪区（坐标区不铺满画布）。
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
from matplotlib.lines import Line2D

# ---------- 全局样式 ----------
fontManager.addfont('./data')
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

BASE = r'project\学习\智能导航\省赛\算法与图表'
OUT = BASE

# ---------- 数据 ----------
wide = pd.read_csv(os.path.join(BASE, '位移宽表_坡顶水平位移.csv'),
                   encoding='utf-8-sig', index_col=0)
wide.index = pd.to_datetime(wide.index)
wide = wide.sort_index()
dates = list(wide.index)
x = np.arange(len(dates))                    # 连续序号轴：无空白间隔
tick_labels = [d.strftime('%m-%d') for d in dates]
N = len(x)

# 停测段（07-06 -> 07-28）处的时间轴折点位置
gap_x = 13.5


def section(p):
    return int(str(p).split('-')[0])


sections = sorted({section(c) for c in wide.columns})
SEC_COLOR = {
    1: '#1f77b4', 3: '#2ca02c', 4: '#17becf', 5: '#9467bd', 6: '#8c564b',
    7: '#e377c2', 8: '#7f7f7f', 9: '#bcbd22', 10: '#d62728', 11: '#ff7f0e',
    12: '#393b79',
}
GROUP = {s: [c for c in wide.columns if section(c) == s] for s in sections}

print('监测点总数:', wide.shape[1], '| 观测期数:', N)
print('观测日期:', ' '.join(tick_labels))
print('各断面点数:', {s: len(GROUP[s]) for s in sections})


def base_axis(ax):
    ax.grid(True, which='major', axis='both', color='#d9d9d9',
            ls='--', lw=0.7, alpha=0.85, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=10.5)
    ax.set_xlim(-0.7, N - 0.3)
    ax.tick_params(axis='y', labelsize=11)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.spines['left'].set_color('0.4')
    ax.spines['bottom'].set_color('0.4')
    ax.axvline(gap_x, color='0.72', ls=(0, (4, 4)), lw=0.9, zorder=1)


# ============ 图①：各监测点真实位移-时间曲线 ============
fig = plt.figure(figsize=(13.6, 8.2), dpi=200, facecolor='white')
ax = fig.add_axes([0.065, 0.155, 0.80, 0.775])        # 右侧/下侧留白，右下角可裁剪

for s in sections:
    for c in GROUP[s]:
        ax.plot(x, wide[c].values, color=SEC_COLOR[s], lw=1.05, alpha=0.85,
                marker='o', ms=2.0, mew=0, zorder=2)

handles = [Line2D([], [], color=SEC_COLOR[s], lw=2.0, marker='o', ms=3.6, mew=0,
                  label=f'{s} 断面（{GROUP[s][0]}～{GROUP[s][-1]}）') for s in sections]
ax.legend(handles=handles, loc='lower left', fontsize=10.5, ncol=2,
          frameon=True, framealpha=0.95, edgecolor='0.75',
          handlelength=2.0, labelspacing=0.42, columnspacing=1.5, borderpad=0.7)

base_axis(ax)
ax.set_ylim(-14.2, 1.6)
ax.set_yticks(np.arange(0, -14.1, -2))
ax.set_xlabel('观测日期（2026 年，汛期停测时段已压缩，曲线按有效观测期连续绘制）',
              fontsize=12.5, labelpad=8)
ax.set_ylabel('累计水平位移 (mm)', fontsize=13)
ax.set_title('边坡坡顶各监测点累计水平位移 - 时间曲线（实测）',
             fontsize=19, fontweight='bold', pad=14)
ax.text(gap_x, 1.25, '汛期停测 07-07～07-27（时段已压缩）', fontsize=10,
        color='0.45', ha='center', va='top', zorder=5)

fp1 = os.path.join(OUT, '图1_各监测点真实位移-时间曲线.png')
fig.savefig(fp1, facecolor='white')
plt.close(fig)
print('saved:', fp1)

# ============ 图②：10-1 号点累计水平位移曲线 ============
PT = '10-1'
y = wide[PT].values

fig = plt.figure(figsize=(12.6, 7.6), dpi=200, facecolor='white')
ax = fig.add_axes([0.085, 0.165, 0.775, 0.765])       # 右下角留白可裁剪

ax.axhline(0, color='0.62', ls='--', lw=0.9, zorder=1)
ax.plot(x, y, color='#c0392b', lw=2.0, marker='o', ms=5.2,
        mfc='white', mec='#c0392b', mew=1.4, zorder=3, label=f'{PT} 号点累计水平位移')

base_axis(ax)
ax.set_ylim(-14.2, 1.6)
ax.set_yticks(np.arange(0, -14.1, -2))
ax.set_xlabel('观测日期（2026 年，汛期停测时段已压缩，曲线按有效观测期连续绘制）',
              fontsize=12.5, labelpad=8)
ax.set_ylabel('累计水平位移 (mm)', fontsize=13)
ax.set_title(f'{PT} 号监测点累计水平位移 - 时间曲线（实测）',
             fontsize=19, fontweight='bold', pad=14)
ax.legend(loc='lower left', fontsize=11.5, frameon=True, framealpha=0.95,
          edgecolor='0.75', handlelength=2.2)
ax.text(gap_x, 1.25, '汛期停测 07-07～07-27（时段已压缩）', fontsize=10,
        color='0.45', ha='center', va='top', zorder=5)

# 首末值标注
ax.annotate(f'初始观测 {y[0]:.1f} mm', xy=(x[0], y[0]), xytext=(x[0] + 0.35, y[0] + 0.85),
            fontsize=10.5, color='#7b241c', ha='left',
            arrowprops=dict(arrowstyle='-', color='#c0392b', lw=0.8))
ax.annotate(f'累计 {y[-1]:.1f} mm', xy=(x[-1], y[-1]), xytext=(x[-1] - 2.6, y[-1] - 1.15),
            fontsize=11.5, color='#7b241c', fontweight='bold', ha='center',
            arrowprops=dict(arrowstyle='->', color='#c0392b', lw=0.9))

# 停测前后变化量标注
d_prev, d_next = y[13], y[14]
ax.annotate(f'停测前后变化 {d_next - d_prev:+.1f} mm', xy=(gap_x, (d_prev + d_next) / 2),
            xytext=(gap_x - 3.4, (d_prev + d_next) / 2 - 0.4), fontsize=10.5,
            color='#7b241c', ha='center',
            arrowprops=dict(arrowstyle='->', color='#c0392b', lw=0.9))

fp2 = os.path.join(OUT, '图1_10-1号点累计水平位移曲线.png')
fig.savefig(fp2, facecolor='white')
plt.close(fig)
print('saved:', fp2)

# ---------- 核对信息 ----------
s10 = wide['10-1']
print('10-1 序列:', ' '.join(f'{v:.1f}' for v in s10.values))
print('10-1 首/末:', float(s10.iloc[0]), float(s10.iloc[-1]))
print('全部点末期最大|累计|:', wide.iloc[-1].abs().max(),
      '对应点:', wide.iloc[-1].abs().idxmax())
