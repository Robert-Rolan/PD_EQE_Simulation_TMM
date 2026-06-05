# PD_EQE_Simulation_TMM

这是一个用于薄膜二极管型光电器件的桌面端 TMM（Transfer Matrix Method，传输矩阵法）仿真工具。它可以基于材料的实测或文献 n、k 光学参数，模拟器件的外量子效率（EQE）光谱、电场强度分布、反射/透射/吸收结果，并分析不同膜层厚度对 EQE 的影响。

本项目主要面向有机/无机薄膜光电二极管、光探测器和类似多层薄膜器件的结构设计与数据拟合辅助。

## 主要功能

- 构建自定义多层薄膜器件结构，并设置每一层厚度、颜色、材料和是否为活性层。
- 计算理想 EQE、反射率、透射率、总吸收以及各膜层吸收。
- 绘制器件结构示意图和膜层内部归一化电场强度分布。
- 支持从顶部入射、底部入射，以及两种入射方向对比。
- 支持膜层厚度批量扫描，观察厚度变化对 EQE 光谱和指定波长 EQE 的影响。
- 支持导入 CSV、TXT 和部分 Lumerical MDF 格式的 n、k 光学常数文件。
- 支持导出光谱数据、电场分布数据、批量扫描数据和图像。

## 环境要求

- Python 3.10 或更新版本
- Windows、macOS 或 Linux
- 主要依赖：
  - NumPy
  - Matplotlib
  - PySide6
  - pytest（用于测试）

## 安装方法

在项目根目录中运行：

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

如果使用 macOS 或 Linux，请将虚拟环境激活命令替换为：

```bash
source .venv/bin/activate
```

## 运行程序

从项目根目录直接运行：

```powershell
python -m tmm_device_sim
```

完成可编辑安装后，也可以使用命令入口：

```powershell
tmm-device-sim
```

Windows 用户也可以双击或运行：

```powershell
.\start_tmm_device_sim.bat
```

## 运行测试

```powershell
python -m pytest
```

测试覆盖内容包括 TMM 核心计算、材料数据导入、批量厚度扫描、CSV 导出、GUI 基础加载和 Windows 启动脚本。

## n、k 光学数据格式

程序支持导入包含波长、n、k 三列的光学常数数据文件。推荐 CSV 格式，例如：

```csv
wavelength_nm,n,k
400,1.80,0.02
500,1.85,0.03
600,1.90,0.04
```

说明：

- 波长单位可以是 nm 或 um。
- 表头可使用 `wavelength`、`lambda`、`wl` 等常见写法。
- n 列表示折射率实部。
- k 列表示消光系数。
- 如果导入数据的波长范围不能覆盖仿真范围，程序会提示用户选择截取有效波长或外推。

示例材料文件位于：

```text
examples/materials/
```

## 项目结构

```text
PD_EQE_Simulation_TMM/
├── examples/materials/       # 示例 n、k 光学参数文件
├── src/tmm_device_sim/       # 主程序源码
│   ├── gui.py                # PySide6 图形界面
│   ├── simulation.py         # TMM 仿真核心
│   ├── materials.py          # 光学常数读取与插值
│   ├── batch.py              # 厚度批量扫描
│   ├── exports.py            # CSV 导出
│   └── model.py              # 器件层结构数据模型
├── tests/                    # 自动化测试
├── pyproject.toml            # Python 项目配置
├── requirements.txt          # 依赖列表
└── start_tmm_device_sim.bat  # Windows 快速启动脚本
```

## 数据与隐私说明

仓库中只包含工具本体、测试文件和示例材料数据。个人测量数据、本地输出结果、参考文献 PDF、缓存文件和环境变量文件不会纳入版本控制。

`.gitignore` 已排除以下常见本地内容：

- `Ref/`
- `data/`
- `outputs/`
- `results/`
- `__pycache__/`
- `.pytest_cache/`
- `.env`

## 适用场景

- 薄膜光电二极管结构设计
- 有机光探测器 EQE 光谱拟合辅助
- 多层薄膜干涉与吸收趋势分析
- 不同入射方向下器件响应对比
- 膜层厚度优化与参数扫描
