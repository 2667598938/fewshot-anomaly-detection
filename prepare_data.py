"""从 HuggingFace 镜像下载 MVTec AD，并整理成 anomalib 要求的目录结构。

anomalib 内置的官方下载链接已失效（mydrive.ch 返回 404），改用 HF 上的镜像。

用法：
    python prepare_data.py --category bottle
    python prepare_data.py --category all      # 全部 15 类

输出结构：
    datasets/MVTecAD/<类别>/train/good/*.png
    datasets/MVTecAD/<类别>/test/<缺陷>/*.png
    datasets/MVTecAD/<类别>/ground_truth/<缺陷>/*.png
"""

import argparse
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "TheoM55/mvtec_anomaly_detection"
ALL_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
    "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


def copy_images(src: Path, dst: Path) -> int:
    """把 src 下的图片复制到 dst，返回张数。"""
    if not src.is_dir():
        return 0
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in src.iterdir():
        if f.suffix.lower() in IMG_EXT:
            shutil.copy2(f, dst / f.name)
            count += 1
    return count


def prepare(category: str, cache: Path, out_root: Path) -> None:
    patterns = [
        f"images/train/{category}/**",
        f"images/test/{category}/**",
        f"masks/test/{category}/**",
    ]
    print(f"[{category}] 下载中……")
    src = Path(
        snapshot_download(
            REPO_ID,
            repo_type="dataset",
            allow_patterns=patterns,
            local_dir=str(cache),
        )
    )

    dst = out_root / category
    n_train = copy_images(src / "images" / "train" / category / "good", dst / "train" / "good")

    test_root = src / "images" / "test" / category
    gt_root = src / "masks" / "test" / category
    n_test = n_gt = 0
    if test_root.is_dir():
        for defect_dir in sorted(test_root.iterdir()):
            if not defect_dir.is_dir():
                continue
            n_test += copy_images(defect_dir, dst / "test" / defect_dir.name)
            n_gt += copy_images(gt_root / defect_dir.name, dst / "ground_truth" / defect_dir.name)

    print(f"[{category}] train/good={n_train}  test={n_test}  mask={n_gt}  -> {dst}")
    if n_train == 0 or n_test == 0:
        raise SystemExit(f"[{category}] 数据不完整，检查下载是否被中断")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="bottle", help="类别名，或 all")
    ap.add_argument("--out", default="./datasets/MVTecAD")
    ap.add_argument("--cache", default="./.hf_cache")
    args = ap.parse_args()

    cache = Path(args.cache)
    out_root = Path(args.out)
    targets = ALL_CATEGORIES if args.category == "all" else [args.category]
    for cat in targets:
        prepare(cat, cache, out_root)
    print("\n完成。下一步：python run_baseline.py --category", targets[0])


if __name__ == "__main__":
    main()
