"""PatchCore 基线，单类跑通。

用法：
    python run_baseline.py --category bottle --root ./datasets/MVTecAD

说明：
    本脚本按 anomalib 2.x 的 Engine API 写。若你的版本接口不同，
    报错信息里会出现 ImportError 或 TypeError，把报错原文记下来，
    改用官方 quickstart 的写法即可，不要在这里硬改。
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="bottle", help="MVTec AD 类别名")
    parser.add_argument("--root", default="./datasets/MVTecAD", help="数据集根目录")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch
    from anomalib.data import MVTecAD
    from anomalib.engine import Engine
    from anomalib.models import Patchcore

    torch.manual_seed(args.seed)

    datamodule = MVTecAD(
        root=args.root,
        category=args.category,
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=4,
    )
    model = Patchcore()
    engine = Engine(accelerator="gpu", devices=1)

    engine.fit(model=model, datamodule=datamodule)
    engine.test(model=model, datamodule=datamodule)

    print("\n结果在 results/ 下，把 image AUROC 记进 experiments/log.md")


if __name__ == "__main__":
    main()
