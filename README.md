# XRD Crystallite Size Distribution Analyzer

**A regularized inverse-problem approach to extract crystallite size distributions from XRD line profiles — designed for Pt/C fuel cell catalysts and extensible to general nanoparticle systems.**

---

## 背景 / Background

传统 Scherrer 方程只能给出一个平均晶粒尺寸，无法反映真实的粒径分布。本工具将 XRD 峰形分析转化为一个**正则化反问题**（Regularized Inverse Problem），通过非负最小二乘法（NNLS）+ Tikhonov 正则化，直接从 XRD 衍射峰提取**晶粒尺寸分布（Crystallite Size Distribution, CSD）**，无需假设分布函数形式。

The conventional Scherrer equation yields only a single mean crystallite size. This tool reformulates XRD line-profile analysis as a **regularized inverse problem**: it extracts the full **Crystallite Size Distribution (CSD)** directly from XRD data via Non-Negative Least Squares (NNLS) with Tikhonov regularization, without assuming a priori distribution shape.

### 与常用粒径/晶粒尺寸表征方法的关系 / Comparison with Common Methods

本工具关注的是 **XRD 衍射峰展宽中包含的晶粒尺寸分布信息**。它并不是替代 SAXS、DLS、激光粒度或 TEM，而是补充这些方法：这些技术通常测量颗粒、团聚体或投影尺寸，而 XRD 对相干衍射晶畴（crystallite/domain）更敏感。因此，对于 Pt/C 等纳米催化剂，本工具更适合回答“晶粒尺寸如何分布、不同衍射峰给出的晶畴信息是否一致”这类问题。

This tool extracts the **crystallite/domain size distribution encoded in XRD line broadening**. It is complementary to SAXS, DLS, laser diffraction, and TEM: those methods usually probe particles, aggregates, or projected sizes, whereas XRD is sensitive to coherent crystalline domains.

| 方法 Method | 主要测量对象 Main quantity | 优点 Strengths | 局限性 / 与本工具的区别 Limitations / Difference |
|---|---|---|---|
| Scherrer 方程 | 单一平均晶粒尺寸 | 简单、快速、适合粗略估算 | 只给出一个平均值，不能解析尺寸分布；对峰形、背景、仪器展宽和峰重叠较敏感 |
| Williamson-Hall / 传统线宽分析 | 平均尺寸与微应变趋势 | 可区分部分尺寸展宽与应变展宽 | 通常仍输出平均参数，难以恢复非单峰或宽分布的 CSD |
| FormFit / Whole-powder-pattern XRD 方法 | 基于假设模型的晶粒尺寸/形貌参数 | 能利用完整谱图和结构模型，物理约束强 | 通常需要预设粒子形状、尺寸分布或结构模型；模型不匹配时结果会偏倚 |
| SAXS | 纳米颗粒/孔结构的散射尺寸分布 | 对 1-100 nm 结构敏感，统计性好 | 需要电子密度对比和形状模型；结果常对应颗粒或聚集体，不一定等同于晶粒 |
| DLS | 溶液中的水合动力学粒径 | 快速、原位、适合分散液 | 强度加权，偏向大颗粒/团聚体；不适用于干粉或负载催化剂晶畴分析 |
| 激光粒度 | 微米级颗粒/团聚体粒径 | 测量范围宽，适合粉体与浆料 | 对纳米晶粒不敏感；结果通常是颗粒或团聚体尺寸，不是晶畴尺寸 |
| TEM | 投影颗粒尺寸与形貌 | 可直接观察形貌、分散状态和局部结构 | 视野有限、统计量依赖采样和图像分割；测得的是投影颗粒尺寸，不一定是相干晶畴 |
| **本工具** | XRD 峰形反演得到的晶粒尺寸分布 CSD | 不预设分布函数，可多峰联合拟合，并用正则化稳定反演 | 结果仍依赖峰选择、背景处理、仪器展宽校正和“峰展宽主要来自尺寸效应”的假设；微应变/缺陷展宽可能需要额外建模 |

简言之：SAXS/DLS/激光粒度/TEM 更偏向“颗粒或团聚体有多大”，传统 XRD 方法多给“平均晶粒多大”，而本工具尝试从 XRD 峰形中恢复“晶粒尺寸如何分布”。

---

## 主要功能 / Features

- **多峰联合拟合**：同时处理多个衍射峰（如 Pt 的 111、200、220、311），约束更强、结果更可靠
- **Kα₂ 双线分离**：自动计算并分离 Cu/Co/Fe/Mo 靶的 Kα₂ 贡献
- **L-Curve 自动选参**：自动扫描正则化参数 α，用最大曲率法定位最优拐点
- **仪器展宽修正入口**：可设置仪器展宽 FWHM（°2θ），用于减弱仪器分辨率对尺寸反演的影响
- **RAW/TXT 自动读取**：支持两列 TXT、Bruker RAW（v1/v3/v4）以及 Rigaku RAW/Ultima 数据
- **图例交互**：点击图例可显示/隐藏各峰的分布曲线
- **CSV 导出**：一键导出粒径分布和拟合曲线数据

---

## 安装 / Installation

### 1. 安装 Python

推荐下载安装 [Anaconda](https://www.anaconda.com/download) 或 Miniconda。本程序包含较多矩阵运算，使用 conda 默认的 NumPy/SciPy 通常会调用 Intel MKL，计算速度明显快于某些普通 pip 环境。

### 2. 下载本工具

点击页面右上角绿色的 **Code → Download ZIP**，解压到任意文件夹。

或者用 Git：
```bash
git clone https://github.com/dragonMaLong/xrd-analyzer.git
cd xrd-analyzer
```

### 3. 安装依赖（推荐 conda / MKL）

打开终端（Windows 推荐用 Anaconda Prompt），进入解压后的文件夹：

```bash
conda env create -f environment.yml
conda activate xrd-analyzer
```

如果不使用 conda，也可以用 pip 安装依赖：

```bash
pip install -r requirements.txt
```

> 性能提示：如果使用 pip 安装的 NumPy/SciPy 没有链接到 MKL 或高性能 OpenBLAS，精细计算和 L-Curve 分析可能会明显变慢。建议优先使用上面的 conda 环境。

### 4. 运行程序

```bash
python run.py
```

---

## 数据格式 / Data Format

程序可自动识别以下 XRD 数据格式：

- 两列 TXT：**第一行为样品名称**，之后每行为 `2θ（°）  强度`
- Bruker RAW：v1、v3（RAW1.01）、v4（RAW4.00）
- Rigaku RAW：ASCII `.raw` 以及 Ultima IV / RINT 二进制格式

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
1. 点击「导入文件」加载 TXT 或 RAW 数据
        ↓
2. 用左侧滑块设置分析角度范围（蓝色虚线）
        ↓
3. 勾选需要分析的峰，拖动峰位标记到对应峰中心
        ↓
4. 设置粒径范围、平滑因子 α 和仪器展宽 FWHM；必要时点击「L-Curve 分析」自动选择 α
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

其中 $\mathbf{L}$ 为一阶差分矩阵（Tikhonov 正则化），$\alpha$ 为正则化参数（可手动设置，也可由 L-Curve 自动确定）。程序还提供仪器展宽 FWHM 参数，用于在峰形核函数中考虑仪器分辨率贡献。

---

## 文件结构 / Repository Structure

```
xrd-analyzer/
├── run.py                      # 启动入口（直接运行这个）
├── environment.yml             # 推荐 conda 环境（含 MKL）
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
    │   └── file_reader.py      # TXT/RAW 自动识别与读取
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
