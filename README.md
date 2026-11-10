# fewshot-anomaly-detection

工业缺陷检测 / 异常检测研究代码。

方向：工业（尚未确定具体问题）。当前仓库处于方向探索与基线搭建阶段，只有 PatchCore 基线可运行，还没有属于本项目的改动。

> 选题状态：**未确认**。当前代码只是基线环境，尚未绑定到任何选题。

## 环境

- Windows / Python 3.12.0
- RTX 4070 Laptop（8G 显存）
- 16G 内存：batch 不能开大，否则 `bad allocation`
- torch 2.6.0+cu124 / anomalib 2.7.0

版本以 `requirements.txt` 为准，已锁定精确版本。

## 快速开始

```powershell
# 1. 建虚拟环境
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. 先装 PyTorch（CUDA 12.4 版，必须用这个 index）
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# 3. 再装其余依赖
pip install -r requirements.txt

# 4. 验证环境
python check_env.py

# 5. 下载数据（anomalib 内置链接已失效，走这个脚本）
python prepare_data.py --category bottle

# 6. 跑基线
python run_baseline.py --category bottle --batch 8 --workers 2
```

## 目录

```
check_env.py        环境自检，输出 torch / CUDA / anomalib 版本
prepare_data.py     从 HF 镜像下载 MVTec AD 并整理成 anomalib 目录结构
run_baseline.py     PatchCore 基线，单类运行，支持 --out-json
run_all.py          批量跑多个类别，汇总到 experiments/baseline_patchcore.csv
experiments/log.md  实验记录本，每次运行必须记一行
datasets/           数据（不入库）
results/            输出（不入库）
```

## 数据来源与许可

- **MVTec AD**：官方站需填表申请，本项目通过 HuggingFace 第三方镜像 `TheoM55/mvtec_anomaly_detection` 获取，由 `prepare_data.py` 下载并重排成 anomalib 的目录结构。
- **许可**：MVTec AD 面向非商业研究用途（CC BY-NC-SA 4.0，以官网声明为准）。数据不入库，仓库只保存下载脚本。
- **镜像失效时的处理**：修改 `prepare_data.py` 里的 `REPO_ID`，或改回官方渠道手工下载后按同一目录结构放置。

## W1 验收标准

- `python check_env.py` 输出 `cuda 可用 : True` 且能打印出显卡名
- `python run_baseline.py --category bottle --batch 8 --workers 2` 跑完，image AUROC ≥ 0.98
- 在 `experiments/log.md` 里补上这一行记录

已知数值（2026-11-10，seed=42）：image AUROC 1.0000 / pixel AUROC 0.9856，与原论文一致。

## 约定

- 所有实验固定 seed，seed 写进记录本
- 结果只认三次以上重复的均值，单次跑高不算数
- 数据集和权重不入库
- 记录本里必须区分「环境验证 / 预实验 / 正式实验」，避免以后分不清哪些数据可用
