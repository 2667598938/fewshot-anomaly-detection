# fewshot-anomaly-detection

工业异常检测（少样本 / 跨类别）研究代码。

一句话目标：现有异常检测方法在正常样本极少（1/2/4-shot）或训练类别与测试类别不重叠时性能明显下降，本项目只在一个环节做改动并给出完整消融。

## 环境

- Windows + RTX 4070（12G）
- Python 3.11
- PyTorch（CUDA 12.x）、anomalib

## 快速开始

```powershell
# 1. 建虚拟环境
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. 装依赖（torch 请去 pytorch.org 复制对应 CUDA 版本的安装命令）
pip install -r requirements.txt

# 3. 验证环境
python check_env.py

# 4. 跑第一个基线（bottle 单类）
python run_baseline.py --category bottle
```

## 目录

```
check_env.py        环境自检，输出 torch / CUDA / anomalib 版本
run_baseline.py     PatchCore 基线，单类跑通
experiments/log.md  实验记录本，每次运行必须记一行
datasets/           数据（不入库）
results/            输出（不入库）
```

## W1 验收标准

- `python check_env.py` 输出 `cuda available: True` 且能打印出显卡名
- `python run_baseline.py --category bottle` 跑完，image AUROC ≥ 0.98
- 在 `experiments/log.md` 里补上这一行记录

## 约定

- 所有实验固定 seed，seed 写进记录本
- 结果只认三次以上重复的均值，单次跑高不算数
- 数据集和权重不入库
