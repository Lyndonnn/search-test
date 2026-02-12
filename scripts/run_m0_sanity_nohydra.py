#!/usr/bin/env python3
import os

from omegaconf import OmegaConf

from mmsearch_r1.trainer.multimodal.main_ppo import run_ppo


def main() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cfg_dir = os.path.join(repo_root, "mmsearch_r1", "trainer", "multimodal", "config")
    base_cfg = OmegaConf.load(os.path.join(cfg_dir, "ppo_trainer.yaml"))
    exp_cfg = OmegaConf.load(os.path.join(cfg_dir, "exp", "m0_sanity.yaml"))
    cfg = OmegaConf.merge(base_cfg, exp_cfg)
    run_ppo(cfg)


if __name__ == "__main__":
    main()
