"""
Parse training log and plot losses.
Usage: python scripts/plot_train_loss.py --log-file output_3d/train_log.txt --out-dir output_3d
"""
import re
import argparse
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def parse_log(log_path):
    pattern = re.compile(
        r"\[(\d+)/\d+\]\s+loss=([\d.]+)\s+"
        r"dino_g=([\d.]+)\s+dino_l=([\d.]+)\s+dino_diff=([\d.e+\-]+)\s+"
        r"ibot=([\d.]+)\s+lr=([\d.e+\-]+)"
    )
    iters, losses, dino_g, dino_l, ibot, lr = [], [], [], [], [], []
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                iters.append(int(m.group(1)))
                losses.append(float(m.group(2)))
                dino_g.append(float(m.group(3)))
                dino_l.append(float(m.group(4)))
                ibot.append(float(m.group(6)))
                lr.append(float(m.group(7)))
    return iters, losses, dino_g, dino_l, ibot, lr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-file", default="output_3d/train_log.txt")
    ap.add_argument("--out-dir", default="output_3d")
    ap.add_argument("--smooth", type=int, default=50, help="Smoothing window (number of points)")
    args = ap.parse_args()

    iters, losses, dino_g, dino_l, ibot, lr = parse_log(args.log_file)
    print(f"Parsed {len(iters)} entries (iter {iters[0]} to {iters[-1]})")

    def smooth(vals, w):
        if w <= 1:
            return vals
        out = []
        for i in range(len(vals)):
            start = max(0, i - w // 2)
            end = min(len(vals), i + w // 2 + 1)
            out.append(sum(vals[start:end]) / (end - start))
        return out

    w = args.smooth
    it = [i / 1000 for i in iters]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training Losses", fontsize=14, fontweight="bold")

    axes[0, 0].plot(it, losses, alpha=0.25, color="tab:blue", linewidth=0.5)
    axes[0, 0].plot(it, smooth(losses, w), color="tab:blue", linewidth=1.5, label=f"smoothed (w={w})")
    axes[0, 0].set_title("Total Loss")
    axes[0, 0].set_xlabel("Iteration (x1000)")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(it, dino_g, alpha=0.25, color="tab:orange", linewidth=0.5)
    axes[0, 1].plot(it, smooth(dino_g, w), color="tab:orange", linewidth=1.5, label=f"DINO Global (w={w})")
    axes[0, 1].plot(it, dino_l, alpha=0.25, color="tab:green", linewidth=0.5)
    axes[0, 1].plot(it, smooth(dino_l, w), color="tab:green", linewidth=1.5, label=f"DINO Local (w={w})")
    axes[0, 1].set_title("DINO Losses")
    axes[0, 1].set_xlabel("Iteration (x1000)")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(it, ibot, alpha=0.25, color="tab:red", linewidth=0.5)
    axes[1, 0].plot(it, smooth(ibot, w), color="tab:red", linewidth=1.5, label=f"iBOT (w={w})")
    axes[1, 0].set_title("iBOT Loss")
    axes[1, 0].set_xlabel("Iteration (x1000)")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(it, lr, color="tab:purple", linewidth=1.5)
    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].set_xlabel("Iteration (x1000)")
    axes[1, 1].set_ylabel("LR")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    import os
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "train_loss_plot.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
