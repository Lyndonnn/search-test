#!/usr/bin/env python3
"""Fail-fast compatibility checks for the MMSearch-R1 CUDA training stack."""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

LOCKED_VERSIONS = {
    "ray": "2.44.1",
    "tensordict": "0.6.2",
    "torch": "2.6.0",
    "torchvision": "0.21.0",
    "transformers": "4.51.0",
    "vllm": "0.8.2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-nccl", action="store_true")
    parser.add_argument("--require-vllm", action="store_true")
    parser.add_argument("--require-exact-verl", action="store_true")
    parser.add_argument("--require-locked-versions", action="store_true")
    return parser.parse_args()


def package_version(name: str) -> str:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return f"IMPORT_ERROR: {exc}"
    return str(getattr(module, "__version__", "<no __version__>"))


def run_nvidia_smi() -> None:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
    )
    print(f"nvidia-smi={result.stdout.strip() or result.stderr.strip()}")


def check_exact_verl() -> None:
    import verl

    expected_root = Path(os.environ["MMSEARCH_R1_VERL_ROOT"]).resolve()
    actual_path = Path(verl.__file__).resolve()
    print(f"verl_path={actual_path}")
    if expected_root not in actual_path.parents:
        raise RuntimeError(f"Imported veRL from {actual_path}, expected under {expected_root}")


def check_cuda(require_nccl: bool) -> None:
    import torch

    print(f"torch={torch.__version__}")
    print(f"torch_cuda={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"torch_arch_list={torch.cuda.get_arch_list() if torch.cuda.is_available() else []}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    capability = torch.cuda.get_device_capability(device)
    print(f"gpu={torch.cuda.get_device_name(device)}")
    print(f"gpu_capability={capability[0]}.{capability[1]}")

    expected_arch = f"sm_{capability[0]}{capability[1]}"
    arch_list = torch.cuda.get_arch_list()
    if arch_list and expected_arch not in arch_list:
        raise RuntimeError(f"PyTorch wheel does not contain {expected_arch}; compiled arches: {arch_list}")

    lhs = torch.randn((256, 256), device=device, dtype=torch.float32)
    rhs = torch.randn((256, 256), device=device, dtype=torch.float32)
    fp32_sum = float((lhs @ rhs).sum().item())
    bf16_sum = float((lhs.to(torch.bfloat16) @ rhs.to(torch.bfloat16)).float().sum().item())
    torch.cuda.synchronize(device)
    print(f"fp32_matmul_sum={fp32_sum:.4f}")
    print(f"bf16_matmul_sum={bf16_sum:.4f}")

    if require_nccl:
        import torch.distributed as dist

        with tempfile.TemporaryDirectory() as temp_dir:
            rendezvous_path = Path(temp_dir) / "nccl_rendezvous"
            dist.init_process_group(
                backend="nccl",
                init_method=f"file://{rendezvous_path}",
                rank=0,
                world_size=1,
            )
            dist.barrier()
            torch.cuda.synchronize(device)
            print("nccl_single_rank_barrier=ok")
            if dist.is_initialized():
                dist.destroy_process_group()


def check_locked_versions() -> None:
    from importlib.metadata import version

    for package, expected in LOCKED_VERSIONS.items():
        actual = version(package)
        print(f"locked_version {package}={actual} expected={expected}")
        if actual != expected:
            raise RuntimeError(f"{package}=={actual}, expected {expected}")


def main() -> None:
    args = parse_args()
    print(f"python={sys.version}")
    print(f"platform={platform.platform()}")
    run_nvidia_smi()
    for package in ("transformers", "ray", "tensordict", "triton"):
        print(f"{package}={package_version(package)}")
    if args.require_vllm:
        print(f"vllm={package_version('vllm')}")
    if args.require_locked_versions:
        check_locked_versions()
    if args.require_exact_verl:
        check_exact_verl()
    check_cuda(require_nccl=args.require_nccl)
    print("MMSearch-R1 CUDA preflight passed.")


if __name__ == "__main__":
    main()
