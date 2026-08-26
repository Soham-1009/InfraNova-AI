"""
Generate Full 250-Epoch Master Telemetry & Analysis Visualizations
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

def generate_report():
    print("Loading telemetry fragments...")
    p1 = Path("logs/training.csv")
    p2 = Path("logs/training_pix2pixhd_100epochs.csv")
    p3 = Path("kaggle_kernel/run_output/outputs/pix2pixhd_band10_band11/logs/training.csv")
    
    df1 = pd.read_csv(p1)
    df2 = pd.read_csv(p2)
    df3 = pd.read_csv(p3)
    
    # Clean and merge chronologically
    df_combined = pd.concat([df1, df2, df3], ignore_index=True)
    df_combined = df_combined.drop_duplicates(subset=["epoch"], keep="last")
    df_combined = df_combined.sort_values(by="epoch").reset_index(drop=True)
    
    out_dir = Path("outputs/final")
    out_dir.mkdir(parents=True, exist_ok=True)
    master_csv = out_dir / "training_master_250epochs.csv"
    df_combined.to_csv(master_csv, index=False)
    print(f"Master telemetry saved to {master_csv} ({len(df_combined)} epochs: {df_combined['epoch'].min()} to {df_combined['epoch'].max()})")
    
    fig_dir = Path("reports/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Set aesthetics
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 10
    
    epochs = df_combined["epoch"]
    
    # Figure 1: Comprehensive Loss Curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    
    # Total Generator & Discriminator Loss
    ax = axes[0, 0]
    ax.plot(epochs, df_combined["g_loss"], color="#2563eb", label="Total G Loss", lw=1.8)
    ax.set_ylabel("Loss Value", color="#2563eb")
    ax_d = ax.twinx()
    ax_d.plot(epochs, df_combined["d_loss"], color="#dc2626", label="Total D Loss", lw=1.8, alpha=0.85)
    ax_d.set_ylabel("D Loss Value", color="#dc2626")
    ax.set_title("Total Generator & MultiScale Discriminator Loss", fontweight="bold")
    ax.axvline(100, color="gray", linestyle="--", alpha=0.7, label="Decay Start (Ep 100)")
    ax.legend(loc="upper left")
    ax_d.legend(loc="upper right")
    
    # Reconstruction & Perceptual Components
    ax = axes[0, 1]
    ax.plot(epochs, df_combined["l1"], label=r"L1 Loss ($\lambda=10$)", color="#059669", lw=1.5)
    ax.plot(epochs, df_combined["perc"], label=r"VGG Perceptual ($\lambda=10$)", color="#7c3aed", lw=1.5)
    ax.set_title("Reconstruction & VGG Feature Content Losses", fontweight="bold")
    ax.set_ylabel("Loss Value")
    ax.axvline(100, color="gray", linestyle="--", alpha=0.7)
    ax.legend()
    
    # Structural & Chromatic Losses
    ax = axes[1, 0]
    ax.plot(epochs, df_combined["ssim_loss"], label=r"SSIM Loss ($\lambda=5$)", color="#d97706", lw=1.5)
    ax.plot(epochs, df_combined["chroma"], label=r"Chroma Loss ($\lambda=2$)", color="#db2777", lw=1.5)
    ax.set_title("Structural (SSIM) & Chrominance Alignment Losses", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss Value")
    ax.axvline(100, color="gray", linestyle="--", alpha=0.7)
    ax.legend()
    
    # Feature Matching & Adversarial Losses
    ax = axes[1, 1]
    ax.plot(epochs, df_combined["feat"], label=r"Feature Matching ($\lambda=5$)", color="#4f46e5", lw=1.5)
    ax.plot(epochs, df_combined["adv"], label=r"Adversarial Loss ($\lambda=1$)", color="#ea580c", lw=1.5)
    ax.set_title("MultiScale Feature Matching & Adversarial Loss", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss Value")
    ax.axvline(100, color="gray", linestyle="--", alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
    loss_fig_path = fig_dir / "loss_convergence_250epochs.png"
    plt.savefig(loss_fig_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {loss_fig_path}")
    
    # Figure 2: Validation Quality Metrics (SSIM, PSNR, MAE, RMSE)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    
    # SSIM
    ax = axes[0, 0]
    ax.plot(epochs, df_combined["val_ssim"], color="#2563eb", lw=1.8, label="Validation SSIM")
    ax.axhline(0.2514266, color="#dc2626", linestyle=":", lw=1.5, label="Benchmark Peak (0.2514)")
    ax.axvline(100, color="gray", linestyle="--", alpha=0.7, label="Decay Start")
    ax.set_title("Validation Structural Similarity Index (SSIM)", fontweight="bold")
    ax.set_ylabel("SSIM (Higher is Better)")
    ax.legend()
    
    # PSNR
    ax = axes[0, 1]
    ax.plot(epochs, df_combined["val_psnr"], color="#059669", lw=1.8, label="Validation PSNR (dB)")
    ax.axhline(11.5715, color="#dc2626", linestyle=":", lw=1.5, label="Benchmark Peak (11.57 dB)")
    ax.axvline(100, color="gray", linestyle="--", alpha=0.7)
    ax.set_title("Validation Peak Signal-to-Noise Ratio (PSNR)", fontweight="bold")
    ax.set_ylabel("PSNR in dB (Higher is Better)")
    ax.legend()
    
    # MAE & RMSE
    ax = axes[1, 0]
    ax.plot(epochs, df_combined["val_mae"], color="#7c3aed", lw=1.8, label="Validation MAE")
    ax.plot(epochs, df_combined["val_rmse"], color="#d97706", lw=1.8, label="Validation RMSE")
    ax.axvline(100, color="gray", linestyle="--", alpha=0.7)
    ax.set_title("Mean Absolute Error (MAE) & Root Mean Square Error (RMSE)", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Error (Lower is Better)")
    ax.legend()
    
    # Spectral Angle Mapper (SAM)
    ax = axes[1, 1]
    ax.plot(epochs, df_combined["val_sam"], color="#db2777", lw=1.8, label="Validation SAM (rad)")
    ax.axhline(0.20913, color="#dc2626", linestyle=":", lw=1.5, label="Benchmark SAM (0.2091)")
    ax.axvline(100, color="gray", linestyle="--", alpha=0.7)
    ax.set_title("Spectral Angle Mapper (SAM)", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("SAM in Radians (Lower is Better)")
    ax.legend()
    
    plt.tight_layout()
    metrics_fig_path = fig_dir / "quality_metrics_250epochs.png"
    plt.savefig(metrics_fig_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {metrics_fig_path}")
    
    # Figure 3: Radiometric & Color Fidelity Metrics (CIE Lab, Saturation Ratio, Histogram Dist)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    
    # CIE Lab Error
    ax = axes[0]
    ax.plot(epochs, df_combined["val_lab_error"], color="#4f46e5", lw=1.8, label="CIE ΔE*ab Error")
    ax.axhline(28.7341, color="#dc2626", linestyle=":", lw=1.5, label="Benchmark ΔE*ab (28.73)")
    ax.axvline(100, color="gray", linestyle="--", alpha=0.7)
    ax.set_title("Perceptual Color Difference (CIE ΔE*ab)", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ΔE*ab (Lower is Better)")
    ax.legend()
    
    # Saturation Ratio
    ax = axes[1]
    ax.plot(epochs, df_combined["val_sat_ratio"], color="#0891b2", lw=1.8, label="Saturation Ratio (Fake/Real)")
    ax.axhline(1.0, color="#059669", linestyle="-", lw=1.5, label="Ideal Target (1.0)")
    ax.axhline(1.4, color="#ef4444", linestyle=":", alpha=0.5)
    ax.axhline(0.6, color="#ef4444", linestyle=":", alpha=0.5)
    ax.axvline(100, color="gray", linestyle="--", alpha=0.7)
    ax.set_title("Optical Color Saturation Ratio", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Saturation Ratio")
    ax.legend()
    
    # Histogram Distance
    ax = axes[2]
    ax.plot(epochs, df_combined["val_hist_dist"], color="#b45309", lw=1.8, label="Histogram Wasserstein Dist")
    ax.axvline(100, color="gray", linestyle="--", alpha=0.7)
    ax.set_title("RGB Distribution Distance (Wasserstein)", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Distance (Lower is Better)")
    ax.legend()
    
    plt.tight_layout()
    radiometric_fig_path = fig_dir / "radiometric_color_250epochs.png"
    plt.savefig(radiometric_fig_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {radiometric_fig_path}")

    # Figure 4: Optimization Dynamics (Learning Rate, Gradient Norms, Duration)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    
    # Learning Rate Schedule
    ax = axes[0]
    ax.plot(epochs, df_combined["g_lr"], color="#2563eb", lw=2.0, label="G & D Learning Rate")
    ax.set_yscale("log")
    ax.axvline(100, color="gray", linestyle="--", alpha=0.7, label="Decay Start (Ep 100)")
    ax.set_title("Linear Learning Rate Annealing Schedule", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate (Log Scale)")
    ax.legend()
    
    # Gradient Norms
    ax = axes[1]
    ax.plot(epochs, df_combined["g_grad_norm"], color="#7c3aed", lw=1.5, alpha=0.85, label="Generator Grad Norm")
    ax.plot(epochs, df_combined["d_grad_norm"], color="#dc2626", lw=1.5, alpha=0.85, label="Discriminator Grad Norm")
    ax.axvline(100, color="gray", linestyle="--", alpha=0.7)
    ax.set_title("Optimization Gradient Norm Dynamics", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L2 Gradient Norm")
    ax.legend()
    
    # Epoch Duration
    ax = axes[2]
    ax.plot(epochs, df_combined["duration_sec"], color="#059669", lw=1.5, label="Epoch Duration (sec)")
    ax.axhline(df_combined["duration_sec"].mean(), color="#059669", linestyle="--", alpha=0.7, label=f"Mean ({df_combined['duration_sec'].mean():.1f}s)")
    ax.set_title("Per-Epoch Computational Runtime", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Seconds")
    ax.legend()
    
    plt.tight_layout()
    opt_fig_path = fig_dir / "optimization_dynamics_250epochs.png"
    plt.savefig(opt_fig_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {opt_fig_path}")

    # Compute Statistical Summaries
    stats = {
        "total_epochs": len(df_combined),
        "total_training_hours": float(df_combined["duration_sec"].sum() / 3600.0),
        "best_ssim": {
            "value": float(df_combined["val_ssim"].max()),
            "epoch": int(df_combined.loc[df_combined["val_ssim"].idxmax(), "epoch"])
        },
        "best_psnr": {
            "value": float(df_combined["val_psnr"].max()),
            "epoch": int(df_combined.loc[df_combined["val_psnr"].idxmax(), "epoch"])
        },
        "best_lab_error": {
            "value": float(df_combined["val_lab_error"].min()),
            "epoch": int(df_combined.loc[df_combined["val_lab_error"].idxmin(), "epoch"])
        },
        "best_sam": {
            "value": float(df_combined["val_sam"].min()),
            "epoch": int(df_combined.loc[df_combined["val_sam"].idxmin(), "epoch"])
        },
        "best_saturation_distance": {
            "value": float(abs(df_combined["val_sat_ratio"] - 1.0).min()),
            "epoch": int(df_combined.loc[abs(df_combined["val_sat_ratio"] - 1.0).idxmin(), "epoch"]),
            "ratio": float(df_combined.loc[abs(df_combined["val_sat_ratio"] - 1.0).idxmin(), "val_sat_ratio"])
        },
        "stage_averages": {
            "stage1_epochs_1_100": {
                "mean_ssim": float(df_combined[df_combined["epoch"] <= 100]["val_ssim"].mean()),
                "mean_psnr": float(df_combined[df_combined["epoch"] <= 100]["val_psnr"].mean()),
                "mean_lab": float(df_combined[df_combined["epoch"] <= 100]["val_lab_error"].mean()),
                "mean_sat_ratio": float(df_combined[df_combined["epoch"] <= 100]["val_sat_ratio"].mean()),
                "sat_ratio_std": float(df_combined[df_combined["epoch"] <= 100]["val_sat_ratio"].std())
            },
            "stage2_epochs_101_200": {
                "mean_ssim": float(df_combined[(df_combined["epoch"] > 100) & (df_combined["epoch"] <= 200)]["val_ssim"].mean()),
                "mean_psnr": float(df_combined[(df_combined["epoch"] > 100) & (df_combined["epoch"] <= 200)]["val_psnr"].mean()),
                "mean_lab": float(df_combined[(df_combined["epoch"] > 100) & (df_combined["epoch"] <= 200)]["val_lab_error"].mean()),
                "mean_sat_ratio": float(df_combined[(df_combined["epoch"] > 100) & (df_combined["epoch"] <= 200)]["val_sat_ratio"].mean()),
                "sat_ratio_std": float(df_combined[(df_combined["epoch"] > 100) & (df_combined["epoch"] <= 200)]["val_sat_ratio"].std())
            },
            "stage3_epochs_201_250": {
                "mean_ssim": float(df_combined[df_combined["epoch"] > 200]["val_ssim"].mean()),
                "mean_psnr": float(df_combined[df_combined["epoch"] > 200]["val_psnr"].mean()),
                "mean_lab": float(df_combined[df_combined["epoch"] > 200]["val_lab_error"].mean()),
                "mean_sat_ratio": float(df_combined[df_combined["epoch"] > 200]["val_sat_ratio"].mean()),
                "sat_ratio_std": float(df_combined[df_combined["epoch"] > 200]["val_sat_ratio"].std())
            }
        },
        "final_epoch_250": {
            "ssim": float(df_combined.iloc[-1]["val_ssim"]),
            "psnr": float(df_combined.iloc[-1]["val_psnr"]),
            "mae": float(df_combined.iloc[-1]["val_mae"]),
            "rmse": float(df_combined.iloc[-1]["val_rmse"]),
            "lab_error": float(df_combined.iloc[-1]["val_lab_error"]),
            "sam": float(df_combined.iloc[-1]["val_sam"]),
            "sat_ratio": float(df_combined.iloc[-1]["val_sat_ratio"]),
            "hist_dist": float(df_combined.iloc[-1]["val_hist_dist"])
        }
    }
    
    stats_json_path = out_dir / "master_summary_statistics.json"
    with open(stats_json_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Summary statistics saved to {stats_json_path}")
    print("\nReport generation complete!")

if __name__ == "__main__":
    generate_report()
