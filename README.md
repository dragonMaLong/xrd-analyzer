# XRD Crystallite Size Distribution Analyzer

**A regularized inverse-problem approach to extract crystallite size distributions from XRD line profiles — designed for Pt/C fuel cell catalysts and extensible to general nanoparticle systems.**

---

## 背景 / Background

传统 Scherrer 方程只能给出一个平均晶粒尺寸，无法反映真实的粒径分布。本工具将 XRD 峰形分析转化为一个**正则化反问题**（Regularized Inverse Problem），通过非负最小二乘法（NNLS）+ Tikhonov 正则化，直接从 XRD 衍射峰提取**晶粒尺寸分布（Crystallite Size Distribution, CSD）**，无需假设分布函数形式。

The conventional Scherrer equation yields only a single mean crystallite size. This tool reformulates XRD line-profile analysis as a **regularized inverse problem**: it extracts the full **Crystallite Size Distribution (CSD)** directly from XRD data via Non-Negative Least Squares (NNLS) with Tikhonov regularization, without assuming a priori distribution shape.

### 与其他方法的类比 / Analogy to Other Fields

| 领域 Field | 信号 Signal | 核函数 Kernel | 提取量 Output |
|---|---|---|---|
| **本工具 This work** | 衍射强度 vs 2θ | Pearson VII | 晶粒尺寸分布 CSD |
| DRT 阻抗谱 | 阻抗 vs 频率 | RC 传递函数 | 时间常数分布 |
| NLDFT-BET | 吸附量 vs 相对压力 | 孔内等温线 | 孔径分布 |
| 激光粒度（Mie 散射） | 散射强度 vs 角度 | Mie 散射矩阵 | 粒径分布 |

---

## 主要功能 / Features

- **多峰联合拟合**：同时处理多个衍射峰（如 Pt 的 111、200、220、311），约束更强、结果更可靠
- **Kα₂ 双线分离**：自动计算并分离 Cu/Co/Fe/Mo 靶的 Kα₂ 贡献
- **L-Curve 自动选参**：自动扫描正则化参数 α，用最大曲率法定位最优拐点
- **图例交互**：点击图例可显示/隐藏各峰的分布曲线
- **CSV 导出**：一键导出粒径分布和拟合曲线数据

---

## 安装 / Installation

### 1. 安装 Python

推荐下载安装 [Anaconda](https://www.anaconda.com/download)，已包含大部分依赖库。

### 2. 下载本工具

点击页面右上角绿色的 **Code → Download ZIP**，解压到任意文件夹。

或者用 Git：
```bash
git clone https://github.com/YOUR_USERNAME/xrd-csd-analyzer.git
cd xrd-csd-analyzer
```

### 3. 安装依赖

打开终端（Windows 推荐用 Anaconda Prompt），进入解压后的文件夹：

```bash
pip install -r requirements.txt
```

### 4. 运行程序

```bash
python run.py
```

---

## 数据格式 / Data Format

程序支持两列 TXT 文件，**第一行为样品名称**，之后每行为 `2θ（°）  强度`：

```
Pt-C-sample-01
38.100   523
38.200   687
38.300   912
...
```

---

## 使用步骤 / Workflow

```
1. 点击「导入 TXT 文件」加载数据
        ↓
2. 用左侧滑块设置分析角度范围（蓝色虚线）
        ↓
3. 勾选需要分析的峰，拖动峰位标记到对应峰中心
        ↓
4. （可选）点击「L-Curve 分析」自动选择最优平滑因子 α
        ↓
5. 点击「极速计算」（快速预览）或「精细计算」（推荐）
        ↓
6. 点击「保存结果」导出 CSV
```

---

## 理论背景 / Theory

### 正问题 / Forward Model

对于直径为 $D$ 的单一纳米晶粒，其衍射峰形为 Pearson VII 函数：

$$p(2\theta;\,D,\mu) = \left[1 + \left(\frac{2\theta - \mu}{\gamma(D)}\right)^2\right]^{-m(D)}$$

半宽参数 $\gamma(D)$ 由 Scherrer 方程给出：

$$\gamma(D) = \frac{K\lambda}{2D\cos\theta}\cdot\frac{180°}{\pi}$$

形状参数 $m(D)$ 随晶粒尺寸连续变化（$m \in [0.5,\ 5.0]$），反映从大晶粒（洛伦兹主导）到小晶粒（高斯主导）的峰形演变。

### 反问题 / Inverse Problem

观测到的 XRD 峰形是所有晶粒尺寸贡献的线性叠加：

$$I(2\theta) = \int_0^\infty f(D)\,p(2\theta;\,D,\mu)\,\mathrm{d}D + \text{background}$$

离散化后构造基矩阵 $\mathbf{A} \in \mathbb{R}^{N \times P}$，求解正则化最小二乘问题：

$$\min_{f \geq 0}\ \left\|\mathbf{A}f - y\right\|^2 + \alpha^2\left\|\mathbf{L}f\right\|^2$$

其中 $\mathbf{L}$ 为一阶差分矩阵（Tikhonov 正则化），$\alpha$ 为正则化参数（由 L-Curve 自动确定）。

---

## 文件结构 / Repository Structure

```
xrd-csd-analyzer/
├── run.py                      # 启动入口（直接运行这个）
├── requirements.txt            # Python 依赖
├── LICENSE                     # MIT License
├── README.md                   # 本文件
│
└── xrd_analyzer/               # 主程序包
    ├── app.py                  # 程序入口（支持从包内直接运行）
    ├── utils.py                # 工具函数
    │
    ├── core/                   # 核心计算（无 UI 依赖）
    │   ├── peak_functions.py   # Numba JIT 峰形核函数
    │   ├── fitting.py          # NNLS 求解器 + 正则化
    │   └── analysis.py         # 分布后处理
    │
    ├── io/                     # 文件读写
    │   └── file_reader.py      # TXT 读取
    │
    └── ui/                     # 界面层
        ├── app_window.py       # 主窗口
        ├── control_panel_mixin.py
        ├── plot_panel_mixin.py
        └── l_curve_mixin.py
```

---

## 引用 / Citation

如果本工具对你的研究有帮助，请引用：

> [论文发表后在此添加 / Citation will be added after publication]

---

## 参考文献 / References

1. Warren, B.E. (1969). *X-Ray Diffraction*. Addison-Wesley.
2. Langford, J.I. & Wilson, A.J.C. (1978). Scherrer after sixty years. *J. Appl. Cryst.*, **11**, 102–113.
3. Thompson, P., Cox, D.E. & Hastings, J.B. (1987). *J. Appl. Cryst.*, **20**, 79–83.
4. Tikhonov, A.N. & Arsenin, V.Y. (1977). *Solutions of Ill-Posed Problems*. Wiley.

---

## 许可证 / License

MIT License — 可自由用于学术和商业用途，请保留原始署名。
