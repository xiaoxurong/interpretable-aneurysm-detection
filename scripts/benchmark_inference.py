"""Deployment profile: parameters, CPU latency / throughput, and ONNX export.

    python scripts/benchmark_inference.py --out results/inference_benchmark.json

For each backbone (wrapped in the full PC-CBM) it reports
  * trainable parameters and fp32 weight size,
  * PyTorch eager latency at batch 1 and throughput at batch 32 (CPU, N=2048),
  * ONNX export + ONNX Runtime latency, with a max-abs-diff parity check.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import tempfile
import time

import numpy as np
import torch

from pointcbm.backbones import build_backbone
from pointcbm.models import ConceptBottleneck3D, count_parameters


class Exportable(torch.nn.Module):
    """Bakes the frozen concept bank in, so the deployed graph takes only points."""

    def __init__(self, model, E):
        super().__init__()
        self.model = model
        self.register_buffer("E", E)

    def forward(self, x):
        logits, C, _ = self.model(x, self.E)
        return torch.softmax(logits, dim=1)[:, 1], C


def timeit(fn, warmup=3, reps=15):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_points", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=torch.get_num_threads())
    ap.add_argument("--backbones", nargs="*", default=["PointNetLite", "PointNet", "DGCNN", "PointNetPlusPlus"])
    ap.add_argument("--out", default="results/inference_benchmark.json")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    E = torch.nn.functional.normalize(torch.randn(6, 768), dim=-1)
    N = args.num_points
    rows = []

    for name in args.backbones:
        torch.manual_seed(0)
        model = ConceptBottleneck3D(build_backbone(name, 128, 6), d_f=128, d_t=768, d_hidden=256, num_concepts=6, num_classes=2)
        net = Exportable(model, E).eval()
        x1, x32 = torch.randn(1, N, 6), torch.randn(32, N, 6)
        row = {"backbone": name, "params_M": count_parameters(model) / 1e6, "weights_MB": count_parameters(model) * 4 / 2**20}
        with torch.inference_mode():
            row["torch_b1_ms"] = 1e3 * timeit(lambda: net(x1))
            reps = 3 if name in ("DGCNN", "PointNetPlusPlus") else 8
            row["torch_b32_samples_per_s"] = 32 / timeit(lambda: net(x32), warmup=1, reps=reps)
        try:
            import onnxruntime as ort

            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, f"{name}.onnx")
                torch.onnx.export(
                    net,
                    (x1,),
                    path,
                    input_names=["points"],
                    output_names=["p_aneurysm", "concepts"],
                    dynamic_axes={"points": {0: "batch"}},
                    opset_version=17,
                    dynamo=False,
                )
                row["onnx_MB"] = os.path.getsize(path) / 2**20
                so = ort.SessionOptions()
                so.intra_op_num_threads = args.threads
                sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
                feed = {"points": x1.numpy()}
                row["onnx_b1_ms"] = 1e3 * timeit(lambda: sess.run(None, feed))
                with torch.inference_mode():
                    ref = net(x1)
                got = sess.run(None, feed)
                row["onnx_max_abs_diff"] = float(max(np.abs(r.numpy() - g).max() for r, g in zip(ref, got)))
        except Exception as e:  # FPS/ball-query graphs may not export on every opset
            row["onnx_error"] = f"{type(e).__name__}: {str(e)[:120]}"
        rows.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {
                "num_points": N,
                "threads": args.threads,
                "cpu": platform.processor() or platform.machine(),
                "torch": torch.__version__,
                "results": rows,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
