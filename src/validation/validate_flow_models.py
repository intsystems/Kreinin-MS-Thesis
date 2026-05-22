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

from models.flow_model import TimeResNet
from models.flow_model_wo_time import TimeResNet_wo_time
from models.flow_model_wo_attention import TimeResNet_wo_attention
from models.flow_model_wo_time_attention import TimeResNet_wo_time_attention
from monai.networks.nets import SegResNet, SwinUNETR
from dynamic_network_architectures.building_blocks.helper import (
    get_matching_instancenorm,
    convert_dim_to_conv_op,
)
from models.UMambaBot_2d import UMambaBot


SOLVER_NAMES = ["euler_1", "euler_2",
                "euler_3", "rk2_1", "rk4_1", "midpoint_1"]
DIRECTIONS = ["contrast_to_native", "native_to_contrast"]
DATASETS = ["test", "holdout"]


NET_TYPE_MAP = {
    "TimeResNet": "flow",
    "TimeResNet_wo_attention": "flow",
    "TimeResNet_wo_time": "not_flow",
    "TimeResNet_wo_time_attention": "not_flow",
    "SegResNet": "not_flow",
    "SwinUNETR": "not_flow",
    "UMambaBot": "not_flow",
}


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
    if name == "TimeResNet":
        return TimeResNet(**cfg)
    elif name == "SegResNet":
        return SegResNet(**cfg)
    elif name == "TimeResNet_wo_time":
        return TimeResNet_wo_time(**cfg)
    elif name == "TimeResNet_wo_attention":
        return TimeResNet_wo_attention(**cfg)
    elif name == "TimeResNet_wo_time_attention":
        return TimeResNet_wo_time_attention(**cfg)
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
    in_ch = net_cfg.get("in_channels", 2)
    net = build_net(net_name, net_cfg)
    net.load_state_dict(ckpt["net"])
    net.to(device).eval()
    net_type = NET_TYPE_MAP.get(net_name, "not_flow")
    return net, net_name, net_type, in_ch


def load_patient(pid, data_dir):
    nat = sorted(glob.glob(os.path.join(data_dir, pid, "nat_*.npy")))
    art = sorted(glob.glob(os.path.join(data_dir, pid, "art_*.npy")))
    assert len(nat) == len(art) > 0
    native = np.stack([np.load(f).clip(-1000, 1000) for f in nat])
    contrast = np.stack([np.load(f).clip(-1000, 1000) for f in art])
    return native, contrast


# ── Velocity field evaluation ──
@torch.inference_mode()
def compute_v(net, state, t_val, batch_size, net_type, device, in_ch=2):
    n = state.shape[0]
    chunks = []
    for i in range(0, n, batch_size):
        s = state[i:i+batch_size].to(device)
        B, H, W = s.shape
        t_chan = torch.full((B, 1, H, W), t_val, device=device, dtype=s.dtype)
        t_vec = torch.full((B, 1, 1, 1), t_val, device=device, dtype=s.dtype)
        inp = torch.cat([s[:, None], t_chan], dim=1)
        if net_type == "flow":
            out = net(inp, t_vec)
        else:
            out = net(inp)
        chunks.append(out[:, 0].cpu())
        del s, inp, out
    return torch.cat(chunks, dim=0)


@torch.inference_mode()
def run_all_solvers(net, input_vol, direction, batch_size, net_type, device, in_ch=2):
    """Run Euler(1,2,3), RK2, RK4, Midpoint — reusing shared velocity evaluations."""
    x = input_vol
    if direction == "contrast_to_native":
        t0, sign = 1.0, -1.0
    else:
        t0, sign = 0.0, 1.0

    results = {}

    # v at starting point
    v1 = compute_v(net, x, t0, batch_size, net_type, device, in_ch)

    # Euler 1 step
    euler1 = x + sign * v1
    results["euler_1"] = euler1

    # midpoint velocity
    x_half = x + sign * 0.5 * v1
    t_mid = t0 + sign * 0.5
    v_mid = compute_v(net, x_half, t_mid, batch_size, net_type, device, in_ch)

    # Euler 2 steps
    results["euler_2"] = x_half + sign * 0.5 * v_mid

    # Midpoint 1 step
    results["midpoint_1"] = x + sign * v_mid

    # Euler 3 steps
    dt3 = 1.0 / 3.0
    x1 = x + sign * dt3 * v1
    v2 = compute_v(net, x1, t0 + sign * dt3,
                   batch_size, net_type, device, in_ch)
    x2 = x1 + sign * dt3 * v2
    v3 = compute_v(net, x2, t0 + sign * 2 * dt3,
                   batch_size, net_type, device, in_ch)
    results["euler_3"] = x2 + sign * dt3 * v3

    # RK2 (Heun)
    t_end = t0 + sign
    v_end = compute_v(net, euler1, t_end, batch_size, net_type, device, in_ch)
    results["rk2_1"] = x + sign * 0.5 * (v1 + v_end)

    # RK4
    k3 = compute_v(net, x + sign * 0.5 * v_mid, t_mid,
                   batch_size, net_type, device, in_ch)
    k4 = compute_v(net, x + sign * k3, t_end,
                   batch_size, net_type, device, in_ch)
    results["rk4_1"] = x + sign * (v1 + 2*v_mid + 2*k3 + k4) / 6.0

    torch.cuda.empty_cache()
    return results


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
def validate_model(ckpt_path, gpu_id, data_dir, split_path, results_dir, batch_size=4):
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    net, net_name, net_type, in_ch = load_model(ckpt_path, device)
    splits = torch.load(split_path, weights_only=False)

    for ds_name in DATASETS:
        patients = splits[ds_name] if ds_name == "test" else splits["holdout"]
        for direction in DIRECTIONS:
            print(f"{net_name} | {ds_name} | {direction} | {len(patients)} patients")
            for pid in patients:
                # skip if done
                out_dir = os.path.join(
                    results_dir, net_name, direction, ds_name)
                if all(os.path.exists(os.path.join(out_dir, s, f"{pid}.json")) for s in SOLVER_NAMES):
                    continue

                native, contrast = load_patient(pid, data_dir)
                if direction == "contrast_to_native":
                    inp_hu, tgt_hu = contrast, native
                else:
                    inp_hu, tgt_hu = native, contrast

                inp = torch.from_numpy(inp_hu / 1000.0).float()
                tgt = torch.from_numpy(tgt_hu).float()

                t0 = time.time()
                solver_out = run_all_solvers(
                    net, inp, direction, batch_size, net_type, device, in_ch)
                elapsed = time.time() - t0

                for solver_name, out_norm in solver_out.items():
                    out_hu = (out_norm * 1000).clamp(-1000, 1000)
                    metrics = compute_metrics(out_hu, tgt)
                    metrics["time"] = elapsed / len(SOLVER_NAMES)

                    save_dir = os.path.join(out_dir, solver_name)
                    os.makedirs(save_dir, exist_ok=True)
                    with open(os.path.join(save_dir, f"{pid}.json"), "w") as f:
                        json.dump(metrics, f, indent=2)

                del inp, solver_out
                torch.cuda.empty_cache()

    print(f"Done: {net_name}")


def aggregate_results(results_dir, split_path):
    splits = torch.load(split_path, weights_only=False)
    summary = {}

    for net_name in sorted(os.listdir(results_dir)):
        net_dir = os.path.join(results_dir, net_name)
        if not os.path.isdir(net_dir):
            continue
        for direction in DIRECTIONS:
            for ds_name in DATASETS:
                patients = splits[ds_name] if ds_name == "test" else splits["holdout"]
                for solver in SOLVER_NAMES:
                    sol_dir = os.path.join(net_dir, direction, ds_name, solver)
                    if not os.path.isdir(sol_dir):
                        continue
                    all_m = []
                    for pid in patients:
                        p = os.path.join(sol_dir, f"{pid}.json")
                        if os.path.exists(p):
                            with open(p) as f:
                                all_m.append(json.load(f))
                    if all_m:
                        key = f"{net_name}/{direction}/{ds_name}/{solver}"
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
                        default="./validation_results")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoint")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()

    if args.aggregate:
        aggregate_results(args.results_dir, args.split_path)
    else:
        ckpt = os.path.join(args.checkpoint_dir, args.checkpoint)
        validate_model(ckpt, args.gpu, args.data_dir,
                       args.split_path, args.results_dir, args.batch_size)
