# -*- coding: utf-8 -*-
"""
卡尔曼-小波联合滤波去噪（最终版）
结构：原始观测序列 -> 小波软阈值去噪(db5) -> 卡尔曼滤波 -> 去噪位移序列
数据：project\学习\智能导航\省赛\监测数据.csv（实测累计水平位移）
输出：图2 去噪前后对比曲线 + 精度指标表
"""
import os
import numpy as np
import pandas as pd
import pywt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

BASE = r"project\学习\智能导航\省赛"
OUT = os.path.join(BASE, "算法与图表")
os.makedirs(OUT, exist_ok=True)

POINT = '10-1'            # K19+600 断面，全坡变形最大点
WAVELET, LEVEL = 'db5', 2     # 逐日观测序列长度受限，取 2 层分解
Q_PROC = 0.4                  # 过程噪声方差 mm^2（变形动态机动性标定）
SIGMA_SIM = 2.0               # 仿真观测噪声标准差 mm（终端标称精度 ±2.5mm 量级）
SEED = 2026

# ================= 1. 数据 =================
wide = pd.read_csv(os.path.join(OUT, "位移宽表_坡顶水平位移.csv"),
                   encoding='utf-8-sig', index_col=0, parse_dates=True)
ser = wide[POINT].dropna()
y = ser.values.astype(float)
dates = ser.index
n = len(y)
idx = np.arange(n, dtype=float)

# ================= 2. 算法 =================
def wavelet_denoise(x, wavelet=WAVELET, level=LEVEL, alpha=1.0):
    """小波软阈值去噪：通用阈值 λ=α·σ·√(2lnN)，σ 由最高频细节系数 MAD 稳健估计"""
    c = pywt.wavedec(x, wavelet, mode='symmetric', level=level)
    sigma = np.median(np.abs(c[-1] - np.median(c[-1]))) / 0.6745
    lam = alpha * sigma * np.sqrt(2 * np.log(len(x)))
    ct = [c[0]] + [pywt.threshold(k, lam, mode='soft') for k in c[1:]]
    return pywt.waverec(ct, wavelet, mode='symmetric')[:len(x)], sigma, lam

def kalman_filter(z, Q=Q_PROC, R=None):
    """一维卡尔曼滤波：状态方程 x_k=x_{k-1}+w, 观测方程 z_k=x_k+v"""
    z = np.asarray(z, float)
    if R is None:
        R = max(np.var(z - savgol_filter(z, 7, 2)), 1e-9)
    x, P, out = z[0], 1.0, np.zeros_like(z)
    for k in range(len(z)):
        xp, Pp = x, P + Q          # 状态预测
        K = Pp / (Pp + R)          # 卡尔曼增益
        x = xp + K * (z[k] - xp)   # 状态更新
        P = (1 - K) * Pp
        out[k] = x
    return out, R

def joint_filter(obs):
    """卡尔曼-小波联合滤波主流程"""
    w, sg, lam = wavelet_denoise(obs)
    kw, R = kalman_filter(w, Q=Q_PROC)
    return dict(wave=w, joint=kw, sigma=sg, lam=lam, R=R)

def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
def snr(sig, err):
    return float(10 * np.log10(np.var(sig) / max(np.var(err), 1e-12)))
def roughness(a):
    return float(np.std(np.diff(np.asarray(a, float), 2)))

# ---- 2.1 实测序列 ----
r_real = joint_filter(y)
print(f"[实测] sigma={r_real['sigma']:.4f}  λ={r_real['lam']:.4f}  R={r_real['R']:.4f}")
print(f"[实测] 原始 rough={roughness(y):.4f} -> 联合 rough={roughness(r_real['joint']):.4f}")

# ---- 2.2 仿真验证 ----
truth = savgol_filter(y, 7, 2)
sigma_obs = float(np.std(y - truth))
print(f"[实测残差估计] σ_obs={sigma_obs:.3f} mm")

def make_noisy(s):
    """同一噪声水平使用固定随机种子，保证指标表口径一致"""
    return truth + np.random.default_rng(SEED + int(round(s * 10))).normal(0, s, n)

y_noisy = make_noisy(SIGMA_SIM)
r_sim = joint_filter(y_noisy)
kf_only, _ = kalman_filter(y_noisy, Q=Q_PROC)

items = [('含噪观测序列', y_noisy, '#c0392b'),
         ('仅小波去噪', r_sim['wave'], '#27ae60'),
         ('仅卡尔曼滤波', kf_only, '#e67e22'),
         ('卡尔曼-小波联合滤波', r_sim['joint'], '#1f6feb')]
base_r = rmse(y_noisy, truth)
rows = []
for k, v, _c in items:
    rows.append({'方法': k, 'RMSE(mm)': round(rmse(v, truth), 4),
                 'SNR(dB)': round(snr(truth, v - truth), 2),
                 '二阶差分粗糙度': round(roughness(v), 4),
                 'RMSE降低': f"{(1 - rmse(v, truth) / base_r) * 100:.1f}%"})
tab = pd.DataFrame(rows)
print(f"\n=== 仿真验证（σ={SIGMA_SIM}mm, 21 期）===")
print(tab.to_string(index=False))
tab.to_csv(os.path.join(OUT, "表_卡尔曼-小波联合滤波精度指标.csv"),
           index=False, encoding='utf-8-sig')

rob = []
for s in [1.0, 1.5, 2.0, 2.5, 3.0]:
    nn = make_noisy(s)
    r = joint_filter(nn)
    rob.append({'观测噪声σ(mm)': s,
                '含噪RMSE(mm)': round(rmse(nn, truth), 4),
                '联合滤波RMSE(mm)': round(rmse(r['joint'], truth), 4),
                '降幅': f"{(1 - rmse(r['joint'], truth) / rmse(nn, truth)) * 100:.1f}%",
                '粗糙度降幅': f"{(1 - roughness(r['joint']) / roughness(nn)) * 100:.1f}%"})
rob = pd.DataFrame(rob)
print("\n=== 稳健性（不同噪声水平）===")
print(rob.to_string(index=False))
rob.to_csv(os.path.join(OUT, "表_不同噪声水平滤波稳健性.csv"),
           index=False, encoding='utf-8-sig')

# ================= 3. 绘图 =================
fig = plt.figure(figsize=(15, 9.5))
gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1], hspace=0.34, wspace=0.24)

def seq_axis(a, fc=1):
    a.set_xticks(np.arange(1, n + 1, 2 * fc))
    a.set_xticklabels([d.strftime('%m-%d') for d in dates[::2 * fc]], fontsize=8.5)
    a.set_xlim(0.5, n + 0.5)

# (a) 实测序列去噪前后
ax = fig.add_subplot(gs[0, :])
ax.plot(idx, y, 'o', ms=5.5, mfc='none', mec='#c0392b', mew=1.4, label='实测原始观测序列')
ax.plot(idx, r_real['wave'], '--', lw=1.2, color='#27ae60', alpha=0.9, label='仅小波软阈值去噪')
ax.plot(idx, r_real['joint'], '-', lw=2.3, color='#1f6feb', label='卡尔曼-小波联合滤波输出')
ax.set_ylabel('累计水平位移 (mm)', fontsize=11)
ax.set_xlabel('观测日期（第 1～21 期）', fontsize=10)
ax.set_title(f'监测点 {POINT}（K19+600）实测序列去噪前后对比', fontsize=12.5, fontweight='bold')
ax.grid(alpha=0.3, ls='--'); ax.legend(fontsize=9.5, loc='lower left'); seq_axis(ax)

# (b) 仿真含噪序列对比
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(idx, y_noisy, '-', lw=0.9, color='#c0392b', alpha=0.6, label='含噪观测序列')
ax2.plot(idx, truth, '-', lw=2.6, color='k', label='真实变形趋势（参考真值）')
ax2.plot(idx, r_sim['wave'], '--', lw=1.3, color='#27ae60', label='小波去噪')
ax2.plot(idx, kf_only, '-.', lw=1.3, color='#e67e22', label='卡尔曼滤波')
ax2.plot(idx, r_sim['joint'], '-', lw=2.3, color='#1f6feb', label='卡尔曼-小波联合滤波')
ax2.set_ylabel('累计位移 (mm)', fontsize=10)
ax2.set_xlabel('观测期次', fontsize=9)
ax2.set_title(f'含噪序列去噪效果对比（仿真 σ={SIGMA_SIM}mm）', fontsize=11)
ax2.grid(alpha=0.3, ls='--'); ax2.legend(fontsize=8.2, ncol=2); seq_axis(ax2)

# (c) 精度指标
ax3 = fig.add_subplot(gs[1, 1])
names = [r['方法'] for r in rows]
v1 = [r['RMSE(mm)'] for r in rows]
v2 = [r['二阶差分粗糙度'] for r in rows]
xs = np.arange(len(names))
b1 = ax3.bar(xs - 0.19, v1, width=0.36, color='#1f6feb', label='RMSE (mm)')
b2 = ax3.bar(xs + 0.19, v2, width=0.36, color='#8ab4f8', label='二阶差分粗糙度')
for b, v in list(zip(b1, v1)) + list(zip(b2, v2)):
    ax3.text(b.get_x() + b.get_width() / 2, v + 0.03, f'{v:.3f}', ha='center', fontsize=8)
ax3.set_xticks(xs)
ax3.set_xticklabels([n_.replace('卡尔曼-小波联合滤波', '卡尔曼-小波\n联合滤波')
                     .replace('含噪观测序列', '含噪\n原始') for n_ in names], fontsize=8.5)
ax3.set_ylabel('数值（越小越好）', fontsize=10)
ax3.set_title('各方法保真度与平滑度对比', fontsize=11)
ax3.grid(alpha=0.3, ls='--', axis='y'); ax3.legend(fontsize=8.5)
ax3.set_ylim(0, max(v1 + v2) * 1.22)

fp = os.path.join(OUT, "图2_卡尔曼-小波联合滤波去噪前后对比.png")
plt.savefig(fp, bbox_inches='tight'); plt.close()
from PIL import Image as _Img                                  # 右下角补白：预留可裁剪空白
_im = _Img.open(fp).convert('RGB'); _w, _h = _im.size
_cv = _Img.new('RGB', (int(_w * 1.25), int(_h * 1.25)), 'white'); _cv.paste(_im, (0, 0)); _cv.save(fp)
print("saved:", fp)

# ================= 4. 导出 =================
pd.DataFrame({'观测日期': list(dates) * 3,
              '累计位移_mm': list(y) + list(r_real['wave']) + list(r_real['joint']),
              '序列': ['实测原始'] * n + ['仅小波去噪'] * n + ['卡尔曼-小波联合滤波'] * n}
             ).to_csv(os.path.join(OUT, f"滤波结果_{POINT}.csv"), index=False, encoding='utf-8-sig')

np.savez(os.path.join(OUT, f"_cache_{POINT}.npz"), idx=idx, y=y,
         wave=r_real['wave'], joint=r_real['joint'],
         truth=truth, noisy=y_noisy, sim_wave=r_sim['wave'], sim_joint=r_sim['joint'],
         sim_kf=kf_only, sigma_obs=sigma_obs)
print("done")
