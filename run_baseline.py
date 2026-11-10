"""PatchCore 基线，单类运行。

用法：
    python run_baseline.py --category bottle --batch 8 --workers 2
    python run_baseline.py --category bottle --out-json .cache/metrics/bottle.json

说明：
    本脚本按 anomalib 2.x 的 Engine API 写，已在本机验证通过（anomalib 2.7.0）。
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="bottle", help="MVTec AD 类别名")
    parser.add_argument("--root", default="./datasets/MVTecAD", help="数据集根目录")
    parser.add_argument("--seed", type=int, default=42)
    # 16G 内存的机器上 batch 开大会直接 bad allocation，默认给保守值
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out-json", default=None, help="把指标写到这个 json 文件")
    args = parser.parse_args()

    import torch
    from anomalib.data import MVTecAD
    from anomalib.engine import Engine
    from anomalib.models import Patchcore

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    datamodule = MVTecAD(
        root=args.root,
        category=args.category,
        train_batch_size=args.batch,
        eval_batch_size=args.batch,
        num_workers=args.workers,
    )
    model = Patchcore()
    engine = Engine(accelerator="gpu", devices=1)

    engine.fit(model=model, datamodule=datamodule)
    results = engine.test(model=model, datamodule=datamodule)

    metrics = {}
    if results:
        for key, value in results[0].items():
            if isinstance(value, (int, float)):
                metrics[key] = float(value)

    print(f"\n==== {args.category} ====")
    for key in sorted(metrics):
        print(f"{key}: {metrics[key]:.4f}")

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "category": args.category,
                    "seed": args.seed,
                    "batch": args.batch,
                    "workers": args.workers,
                    "metrics": metrics,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
