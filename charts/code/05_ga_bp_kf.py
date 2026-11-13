# -*- coding: utf-8 -*-
"""
GA-BP-KF 变形预测（真实数据）
流程：遗传算法(GA)优化 BP 神经网络初始权值 → BP 误差反传精调 → 卡尔曼滤波(KF)对预测残差校正
输入特征：[x(t-1), x(t-2), Δt(t)]，显式引入观测间隔以适配汛期停测断档
数据：project\学习\智能导航\省赛\监测数据.csv（10-1 点累计水平位移）
输出：图5 GA-BP-KF 预测曲线 + 精度对比表
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

BASE = r"project\学习\智能导航\省赛"
OUT = os.path.join(BASE, "算法与图表")

M_IN, N_HID, N_OUT = 3, 7, 1
DIM = M_IN * N_HID + N_HID + N_HID * N_OUT + N_OUT
N_TRAIN = 14
N_POP, N_GEN = 40, 80
SEED = 42

wide = pd.read_csv(os.path.join(OUT, "位移宽表_坡顶水平位移.csv"),
                   encoding='utf-8-sig', index_col=0, parse_dates=True)
ser = wide['10-1'].dropna()
x = ser.values.astype(float)
dates = ser.index
n = len(x)
xmin, xmax = x.min(), x.max()
xn = (x - xmin) / (xmax - xmin)
dt = np.r_[1.0, np.diff(dates.values).astype('timedelta64[D]').astype(float)]
dt_n = dt / 30.0

# ---------------- 样本构造 ----------------
X, Y, t_idx = [], [], []
for t in range(2, n):
    X.append([xn[t - 1], xn[t - 2], dt_n[t]])
    Y.append(xn[t]); t_idx.append(t)
X, Y, t_idx = np.array(X), np.array(Y), np.array(t_idx)
Xtr, Ytr = X[:N_TRAIN], Y[:N_TRAIN]
Xte, Yte = X[N_TRAIN:], Y[N_TRAIN:]
t_te = t_idx[N_TRAIN:]
print(f"样本总数 {len(X)}，训练 {len(Xtr)}，检验 {len(Xte)}；"
      f"检验期次 {t_te[0] + 1}~{t_te[-1] + 1}")

# ---------------- 网络 ----------------
def unpack(p):
    i1 = M_IN * N_HID; i2 = i1 + N_HID; i3 = i2 + N_HID * N_OUT
    return (p[:i1].reshape(M_IN, N_HID), p[i1:i2],
            p[i2:i3].reshape(N_HID, N_OUT), p[i3:i3 + N_OUT])

def forward(p, Xa):
    W1, b1, W2, b2 = unpack(p)
    H = np.tanh(Xa @ W1 + b1)
    return (H @ W2 + b2).ravel(), H

def mse(p, Xa, Ya):
    return float(np.mean((forward(p, Xa)[0] - Ya) ** 2))

def bp_train(p, Xa, Ya, lr=0.03, mom=0.9, epochs=1500):
    v = np.zeros_like(p)
    for _ in range(epochs):
        yh, H = forward(p, Xa)
        W1, b1, W2, b2 = unpack(p)
        e = (yh - Ya).reshape(-1, 1)
        gW2, gb2 = H.T @ e, e.sum(0)
        dH = (e @ W2.T) * (1 - H ** 2)
        grad = np.concatenate([(Xa.T @ dH).ravel(), dH.sum(0), gW2.ravel(), gb2])
        gn = np.linalg.norm(grad)                      # 梯度裁剪，防止发散
        if gn > 1.0:
            grad = grad / gn
        v = mom * v - lr * grad
        p = p + v
    return p

# ---------------- GA 优化初始权值 ----------------
rng = np.random.default_rng(SEED)
pop = rng.normal(0, 0.6, (N_POP, DIM))
best_hist = []
for g in range(N_GEN):
    fit = np.array([1.0 / (1.0 + mse(ind, Xtr, Ytr)) for ind in pop])
    order = np.argsort(-fit)
    best_hist.append(mse(pop[order[0]], Xtr, Ytr))
    new = [pop[order[0]].copy()]
    while len(new) < N_POP:
        a, b = rng.choice(N_POP, 2, replace=False)
        p1 = pop[a] if fit[a] > fit[b] else pop[b]
        p3 = pop[rng.choice(N_POP, 1)[0]]
        child = np.where(rng.random(DIM) < 0.5, p1, p3)
        mut = rng.random(DIM) < 0.12
        child = child + mut * rng.normal(0, 0.25 * max(0.2, 1 - g / N_GEN), DIM)
        new.append(child)
    pop = np.array(new)

p_ga0 = pop[np.argmin([mse(i, Xtr, Ytr) for i in pop])].copy()
print(f"[GA] 最优初始权值 训练集 MSE={mse(p_ga0, Xtr, Ytr):.6f}")

# ---------------- 三种方案 ----------------
def kf1d(z, Q, R):
    z = np.asarray(z, float)
    xx, P, out = z[0], 1.0, np.zeros_like(z)
    for k in range(len(z)):
        xp, Pp = xx, P + Q
        K = Pp / (Pp + R)
        xx = xp + K * (z[k] - xp); P = (1 - K) * Pp
        out[k] = xx
    return out

p_bp = bp_train(rng.normal(0, 0.6, DIM), Xtr, Ytr)      # 常规 BP
p_gabp = bp_train(p_ga0, Xtr, Ytr)                       # GA-BP
pred_raw = forward(p_gabp, X)[0]
e_tr = Ytr - pred_raw[:N_TRAIN]
Qk = max(float(np.var(np.diff(e_tr))), 1e-9)
Rk = max(float(np.var(e_tr)), 1e-9)
full = kf1d(pred_raw, Qk, Rk)                            # GA-BP-KF
p_bp_curve = forward(p_bp, X)[0]
p_gabp_curve = pred_raw
print(f"[KF 残差校正] Q={Qk:.3e}  R={Rk:.3e}")

def denorm(a):
    return np.asarray(a) * (xmax - xmin) + xmin

def metrics(pred_n, ytrue_n):
    e = denorm(pred_n) - denorm(ytrue_n)
    return dict(RMSE=float(np.sqrt(np.mean(e ** 2))), MAE=float(np.mean(np.abs(e))),
                MaxAE=float(np.max(np.abs(e))))

rows = []
for name, curve in [('常规 BP 神经网络', p_bp_curve),
                    ('GA-BP（遗传算法优化）', p_gabp_curve),
                    ('GA-BP-KF（本文方法）', full)]:
    rows.append({'预测模型': name, **{k: round(v, 4) for k, v in metrics(curve[N_TRAIN:], Yte).items()}})
tab = pd.DataFrame(rows)
print("\n=== 检验段预测精度 ===")
print(tab.to_string(index=False))
tab.to_csv(os.path.join(OUT, "表_GA-BP-KF预测精度对比.csv"), index=False, encoding='utf-8-sig')

det = pd.DataFrame({
    '观测日期': [d.strftime('%Y-%m-%d') for d in dates[t_te]],
    '实测位移(mm)': np.round(denorm(Yte), 3),
    '常规BP预测(mm)': np.round(denorm(p_bp_curve[N_TRAIN:]), 3),
    'GA-BP预测(mm)': np.round(denorm(p_gabp_curve[N_TRAIN:]), 3),
    'GA-BP-KF预测(mm)': np.round(denorm(full[N_TRAIN:]), 3),
})
det['GA-BP-KF绝对误差(mm)'] = np.round(np.abs(det['GA-BP-KF预测(mm)'] - det['实测位移(mm)']), 3)
print("\n=== 检验段逐期明细 ===")
print(det.to_string(index=False))
det.to_csv(os.path.join(OUT, "表_GA-BP-KF预测逐期明细.csv"), index=False, encoding='utf-8-sig')

# ---------------- 递归外推未来 5 期 ----------------
seq = list(xn)
fut = []
for _ in range(5):
    inp = np.array([seq[-1], seq[-2], 1.0 / 30]).reshape(1, -1)
    yv = float(forward(p_gabp, inp)[0][0])
    seq.append(yv); fut.append(yv)
fut = np.array(fut)
fut_kf = kf1d(np.r_[full[-3:], fut], Qk, Rk)[3:]

# ================= 绘图 =================
fig = plt.figure(figsize=(15, 9.6))
gs = fig.add_gridspec(2, 2, hspace=0.36, wspace=0.24)

ax = fig.add_subplot(gs[0, 0])
ax.plot(np.arange(1, N_GEN + 1), best_hist, '-', lw=2, color='#1f6feb')
ax.set_yscale('log'); ax.set_xlabel('遗传代数', fontsize=10)
ax.set_ylabel('最优个体 MSE（对数坐标）', fontsize=10)
ax.set_title('遗传算法适应度收敛曲线', fontsize=11.5, fontweight='bold')
ax.grid(alpha=0.3, ls='--')

ax2 = fig.add_subplot(gs[0, 1])
tn = np.arange(1, n + 1)
ax2.plot(tn, x, 'o-', ms=4.5, lw=1.8, color='k', label='实测位移')
te_x = t_te + 1
ax2.plot(te_x, denorm(p_bp_curve[N_TRAIN:]), 's--', ms=5, lw=1.3, color='#e67e22', label='常规 BP')
ax2.plot(te_x, denorm(p_gabp_curve[N_TRAIN:]), '^--', ms=5, lw=1.3, color='#27ae60', label='GA-BP')
ax2.plot(te_x, denorm(full[N_TRAIN:]), 'D-', ms=5, lw=2.1, color='#1f6feb', label='GA-BP-KF（本文）')
fn = np.arange(n + 1, n + 6)
ax2.plot(fn, denorm(fut_kf), 'D:', ms=5, lw=1.8, color='#c0392b', label='未来 5 期外推预测')
ax2.axvline(t_te[0] + 0.5, color='#999', ls='--', lw=1.2)
ax2.text(t_te[0] + 0.7, x.min() * 0.78, '训练段 | 检验段', fontsize=8.5, color='#666')
ax2.set_xlabel('观测期次', fontsize=10); ax2.set_ylabel('累计水平位移 (mm)', fontsize=10)
ax2.set_title('GA-BP-KF 变形预测曲线（实测数据）', fontsize=11.5, fontweight='bold')
ax2.grid(alpha=0.3, ls='--'); ax2.legend(fontsize=8.3, loc='lower left')

ax3 = fig.add_subplot(gs[1, 0])
labs = [d.strftime('%m-%d') for d in dates[t_te]]
w = 0.26; xx = np.arange(len(labs))
for k, (name, curve, c) in enumerate([('常规 BP', p_bp_curve, '#e67e22'),
                                      ('GA-BP', p_gabp_curve, '#27ae60'),
                                      ('GA-BP-KF', full, '#1f6feb')]):
    err = np.abs(denorm(curve[N_TRAIN:]) - denorm(Yte))
    b = ax3.bar(xx + (k - 1) * w, err, width=w, color=c, label=name)
    for rect, v in zip(b, err):
        ax3.text(rect.get_x() + rect.get_width() / 2, v + 0.004, f'{v:.2f}', ha='center', fontsize=7)
ax3.set_xticks(xx); ax3.set_xticklabels(labs, fontsize=8.5)
ax3.set_xlabel('检验段观测日期', fontsize=10); ax3.set_ylabel('绝对预测误差 (mm)', fontsize=10)
ax3.set_title('检验段逐期绝对预测误差对比', fontsize=11.5, fontweight='bold')
ax3.grid(alpha=0.3, ls='--', axis='y'); ax3.legend(fontsize=9)

ax4 = fig.add_subplot(gs[1, 1])
names = [r['预测模型'].replace('（遗传算法优化）', '\n(GA 优化)').replace('（本文方法）', '\n(本文)') for r in rows]
v1 = [r['RMSE'] for r in rows]; v2 = [r['MAE'] for r in rows]; v3 = [r['MaxAE'] for r in rows]
xx = np.arange(len(names)); w = 0.26
for k, (val, c, lab) in enumerate([(v1, '#1f6feb', 'RMSE'), (v2, '#8ab4f8', 'MAE'), (v3, '#c5221f', '最大绝对误差')]):
    b = ax4.bar(xx + (k - 1) * w, val, width=w, color=c, label=lab)
    for rect, v in zip(b, val):
        ax4.text(rect.get_x() + rect.get_width() / 2, v + 0.006, f'{v:.3f}', ha='center', fontsize=8)
ax4.set_xticks(xx); ax4.set_xticklabels(names, fontsize=9)
ax4.set_ylabel('误差 (mm)', fontsize=10)
ax4.set_title('三种模型预测精度对比（检验段）', fontsize=11.5, fontweight='bold')
ax4.grid(alpha=0.3, ls='--', axis='y'); ax4.legend(fontsize=9)
ax4.set_ylim(0, max(v3) * 1.28)

fp = os.path.join(OUT, "图5_GA-BP-KF变形预测.png")
plt.savefig(fp, bbox_inches='tight'); plt.close()
from PIL import Image as _Img                                  # 右下角补白：预留可裁剪空白
_im = _Img.open(fp).convert('RGB'); _w, _h = _im.size
_cv = _Img.new('RGB', (int(_w * 1.25), int(_h * 1.25)), 'white'); _cv.paste(_im, (0, 0)); _cv.save(fp)
print("saved:", fp)

np.savez(os.path.join(OUT, "_cache_ga.npz"), best_hist=best_hist, fut=fut_kf,
         full=full, gabp=p_gabp_curve, bpcurve=p_bp_curve, t_te=t_te, Yte=Yte,
         x=x, dates=np.array([d.strftime('%Y-%m-%d') for d in dates]))
print("done")
