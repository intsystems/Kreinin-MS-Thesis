import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses.ssim_loss import SSIMLoss
from monai.networks.nets import SegResNet, UNet, SwinUNETR
from models.UMambaBot_2d import UMambaBot
from dynamic_network_architectures.building_blocks.helper import (
    get_matching_instancenorm,
    convert_dim_to_conv_op,
)

DATASETS = ["test", "holdout"]
DIRECTION = "contrast_to_native"


class InitWeights_He:
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            module.weight = nn.init.kaiming_normal_(
                module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


def build_net(name, cfg):
    if name == "UNet":
        return UNet(**cfg)
    elif name == "SegResNet":
        return SegResNet(**cfg)
    elif name == "UMambaBot":
        conv_op = convert_dim_to_conv_op(2)
        net = UMambaBot(
            input_channels=cfg["input_channels"],
            n_stages=cfg["n_stages"],
            features_per_stage=cfg["features_per_stage"],
            conv_op=conv_op,
            kernel_sizes=cfg["kernel_sizes"],
            strides=cfg["strides"],
            num_classes=cfg["num_classes"],
            deep_supervision=cfg["deep_supervision"],
            n_conv_per_stage=cfg["n_conv_per_stage"],
            n_conv_per_stage_decoder=cfg["n_conv_per_stage_decoder"],
            conv_bias=cfg["conv_bias"],
            norm_op=get_matching_instancenorm(conv_op),
            norm_op_kwargs=cfg["norm_op_kwargs"],
            dropout_op=cfg["dropout_op"],
            dropout_op_kwargs=cfg["dropout_op_kwargs"],
            nonlin=nn.LeakyReLU,
            nonlin_kwargs=cfg["nonlin_kwargs"],
        )
        net.apply(InitWeights_He(1e-2))
        return net
    elif name == "SwinUNETR":
        valid_keys = {
            "in_channels", "out_channels", "depths", "num_heads",
            "feature_size", "norm_name", "normalize", "spatial_dims",
            "downsample", "patch_size", "window_size", "use_v2",
        }
        filtered = {k: v for k, v in cfg.items() if k in valid_keys}
        return SwinUNETR(**filtered)
    raise ValueError(f"Unknown network: {name}")


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    net_name = ckpt["config"]["hyperparams"]["net_name"]
    net_cfg = ckpt["config"]["net"]
    net = build_net(net_name, net_cfg)
    net.load_state_dict(ckpt["net"])
    net.to(device).eval()
    return net, net_name


def load_patient(pid, data_dir):
    nat = sorted(glob.glob(os.path.join(data_dir, pid, "nat_*.npy")))
    art = sorted(glob.glob(os.path.join(data_dir, pid, "art_*.npy")))
    assert len(nat) == len(art) > 0
    native = np.stack([np.load(f).clip(-1000, 1000) for f in nat])
    contrast = np.stack([np.load(f).clip(-1000, 1000) for f in art])
    return native, contrast


@torch.inference_mode()
def run_inference(net, input_vol, batch_size, device):
    n = input_vol.shape[0]
    chunks = []
    for i in range(0, n, batch_size):
        inp = input_vol[i:i+batch_size, None].to(device)
        out = net(inp)
        chunks.append(out[:, 0].cpu())
        del inp, out
    return torch.cat(chunks, dim=0)


def compute_metrics(pred_hu, target_hu):
    ssim_fn = SSIMLoss(spatial_dims=2, data_range=1.0)
    mae_val = F.l1_loss(pred_hu, target_hu).item()
    mse = F.mse_loss(pred_hu + 1000, target_hu + 1000)
    psnr_val = 10 * math.log10(2000**2 / mse.item()
                               ) if mse > 0 else float("inf")
    p = (pred_hu / 2000 + 0.5).unsqueeze(1)
    t = (target_hu / 2000 + 0.5).unsqueeze(1)
    ssim_val = 1.0 - ssim_fn(p, t).item()
    return {"mae": mae_val, "psnr": psnr_val, "ssim": ssim_val}


@torch.inference_mode()
def validate_model(ckpt_path, gpu_id, data_dir, split_path, results_dir, batch_size=8):
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    net, net_name = load_model(ckpt_path, device)
    splits = torch.load(split_path, weights_only=False)

    for ds_name in DATASETS:
        patients = splits[ds_name] if ds_name == "test" else splits["holdout"]
        print(f"{net_name} | {ds_name} | {len(patients)} patients")

        for pid in patients:
            out_path = os.path.join(
                results_dir, net_name, DIRECTION, ds_name, f"{pid}.json")
            if os.path.exists(out_path):
                continue

            native, contrast = load_patient(pid, data_dir)
            inp = torch.from_numpy(contrast / 1000.0).float()
            tgt = torch.from_numpy(native).float()

            t0 = time.time()
            out_norm = run_inference(net, inp, batch_size, device)
            elapsed = time.time() - t0

            out_hu = (out_norm * 1000).clamp(-1000, 1000)
            metrics = compute_metrics(out_hu, tgt)
            metrics["time"] = elapsed

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(metrics, f, indent=2)

            del inp, out_norm
            torch.cuda.empty_cache()

    print(f"Done: {net_name}")


def aggregate_results(results_dir, split_path):
    splits = torch.load(split_path, weights_only=False)
    summary = {}

    for net_name in sorted(os.listdir(results_dir)):
        net_dir = os.path.join(results_dir, net_name, DIRECTION)
        if not os.path.isdir(net_dir):
            continue
        for ds_name in DATASETS:
            ds_path = os.path.join(net_dir, ds_name)
            if not os.path.isdir(ds_path):
                continue
            patients = splits[ds_name] if ds_name == "test" else splits["holdout"]
            all_m = []
            for pid in patients:
                p = os.path.join(ds_path, f"{pid}.json")
                if os.path.exists(p):
                    with open(p) as f:
                        all_m.append(json.load(f))
            if all_m:
                key = f"{net_name}/{DIRECTION}/{ds_name}"
                summary[key] = {
                    "n": len(all_m),
                    "mae_mean": float(np.mean([m["mae"] for m in all_m])),
                    "mae_std": float(np.std([m["mae"] for m in all_m])),
                    "psnr_mean": float(np.mean([m["psnr"] for m in all_m])),
                    "psnr_std": float(np.std([m["psnr"] for m in all_m])),
                    "ssim_mean": float(np.mean([m["ssim"] for m in all_m])),
                    "ssim_std": float(np.std([m["ssim"] for m in all_m])),
                }

    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {results_dir}/summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default="../data/npy")
    parser.add_argument("--split-path", type=str, default="../data/split.data")
    parser.add_argument("--results-dir", type=str,
                        default="./validation_results_reg")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoint")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()

    if args.aggregate:
        aggregate_results(args.results_dir, args.split_path)
    else:
        ckpt = os.path.join(args.checkpoint_dir, args.checkpoint)
        validate_model(ckpt, args.gpu, args.data_dir,
                       args.split_path, args.results_dir, args.batch_size)
