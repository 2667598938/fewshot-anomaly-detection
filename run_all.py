"""跑完 MVTec AD 全部 15 类的 PatchCore 基线，汇总成一张表。

用法：
    python run_all.py
    python run_all.py --categories bottle,cable

每个类别单独起一个子进程，避免 15 类连着跑把内存吃满。
结果写到 experiments/baseline_patchcore.csv。
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

# 与 prepare_data.py 保持一致的 15 类
ALL_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
    "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]
METRIC_KEYS = ["image_AUROC", "pixel_AUROC", "image_F1Score", "pixel_F1Score"]


def run_one(category: str, batch: int, workers: int, metrics_dir: Path) -> dict:
    out_json = metrics_dir / f"{category}.json"
    cmd = [
        sys.executable, "run_baseline.py",
        "--category", category,
        "--batch", str(batch),
        "--workers", str(workers),
        "--out-json", str(out_json),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out_json.exists():
        tail = "\n".join(proc.stderr.strip().splitlines()[-5:])
        print(f"[{category}] 失败\n{tail}", flush=True)
        return {"category": category}
    data = json.loads(out_json.read_text(encoding="utf-8"))
    row = {"category": category, **data.get("metrics", {})}
    print(
        f"[{category}] image_AUROC={row.get('image_AUROC', float('nan')):.4f} "
        f"pixel_AUROC={row.get('pixel_AUROC', float('nan')):.4f}",
        flush=True,
    )
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", default=None, help="逗号分隔，默认全部 15 类")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="experiments/baseline_patchcore.csv")
    args = ap.parse_args()

    categories = args.categories.split(",") if args.categories else ALL_CATEGORIES
    metrics_dir = Path(".cache/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for cat in categories:
        print(f"\n>>> {cat}", flush=True)
        rows.append(run_one(cat, args.batch, args.workers, metrics_dir))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["category"] + METRIC_KEYS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in writer.fieldnames})

    ok = [r for r in rows if "image_AUROC" in r]
    if ok:
        mean_img = sum(r["image_AUROC"] for r in ok) / len(ok)
        mean_pix = sum(r.get("pixel_AUROC", 0) for r in ok) / len(ok)
        print(f"\n完成 {len(ok)}/{len(rows)} 类")
        print(f"均值 image_AUROC={mean_img:.4f}  pixel_AUROC={mean_pix:.4f}")
    print(f"明细：{out}")


if __name__ == "__main__":
    main()
