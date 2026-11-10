"""环境自检：确认显卡、PyTorch、anomalib 都就位。

用法：
    python check_env.py
"""

import platform
import sys


def main() -> int:
    print(f"python      : {sys.version.split()[0]} ({platform.system()})")

    try:
        import torch
    except ImportError:
        print("torch       : 未安装。请去 https://pytorch.org 复制 CUDA 版安装命令")
        return 1

    print(f"torch       : {torch.__version__}")
    print(f"cuda 编译版 : {torch.version.cuda}")

    available = torch.cuda.is_available()
    print(f"cuda 可用   : {available}")
    if available:
        print(f"显卡        : {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"显存        : {props.total_memory / 1024**3:.1f} GB")
    else:
        print("            -> 装成了 CPU 版，重装 torch 的 CUDA 版本")

    try:
        import anomalib
    except ImportError:
        print("anomalib    : 未安装。执行 pip install anomalib")
        return 1

    print(f"anomalib    : {anomalib.__version__}")

    if not available:
        return 1
    print("\n环境就绪。下一步：python run_baseline.py --category bottle")
    return 0


if __name__ == "__main__":
    sys.exit(main())
