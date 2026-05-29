from __future__ import annotations

import os
import platform
import sys


def main() -> None:
    print(f"python={sys.version.replace(chr(10), ' ')}")
    print(f"platform={platform.platform()}")
    print(f"HF_HOME={os.environ.get('HF_HOME', '')}")
    print(f"DAGIG_DATA_ROOT={os.environ.get('DAGIG_DATA_ROOT', '')}")
    try:
        import torch

        print(f"torch={torch.__version__}")
        print(f"cuda_available={torch.cuda.is_available()}")
        print(f"cuda_device_count={torch.cuda.device_count()}")
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            total_gb = props.total_memory / (1024**3)
            print(f"cuda:{idx} name={props.name} total_memory_gb={total_gb:.1f}")
    except Exception as exc:
        print(f"torch_probe_failed={exc}")

    try:
        import transformers

        print(f"transformers={transformers.__version__}")
    except Exception as exc:
        print(f"transformers_probe_failed={exc}")


if __name__ == "__main__":
    main()

