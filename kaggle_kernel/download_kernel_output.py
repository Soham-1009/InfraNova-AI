"""
Final Artifact Downloader for Completed Epoch 250 Experiment
"""

import time
from pathlib import Path

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest


def download_file_resumable(url: str, dest_path: Path, max_retries: int = 10, chunk_size: int = 4 * 1024 * 1024):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")

    total_size = 0
    try:
        with requests.head(url, timeout=30) as r:
            total_size = int(r.headers.get("content-length", 0))
    except Exception:
        pass

    for attempt in range(1, max_retries + 1):
        existing_bytes = temp_path.stat().st_size if temp_path.exists() else 0
        if total_size > 0 and existing_bytes >= total_size:
            temp_path.replace(dest_path)
            print(f"  [OK] Finished {dest_path.name} ({dest_path.stat().st_size / 1e6:.2f} MB)", flush=True)
            return True

        headers = {}
        mode = "wb"
        if existing_bytes > 0:
            headers["Range"] = f"bytes={existing_bytes}-"
            mode = "ab"

        try:
            print(
                f"  Downloading {dest_path.name} starting from {existing_bytes / 1e6:.1f} MB (Attempt {attempt}/{max_retries})...",
                flush=True,
            )
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                if r.status_code == 416:
                    if total_size > 0 and existing_bytes == total_size:
                        temp_path.replace(dest_path)
                        return True
                    mode = "wb"
                    existing_bytes = 0
                    r = requests.get(url, stream=True, timeout=60)
                elif r.status_code not in (200, 206):
                    mode = "wb"
                    existing_bytes = 0

                if total_size == 0:
                    total_size = int(r.headers.get("content-length", 0)) + existing_bytes

                downloaded = existing_bytes
                with open(temp_path, mode) as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                pct = (downloaded / total_size) * 100
                                print(
                                    f"\r    {dest_path.name}: {downloaded / 1e6:.1f}/{total_size / 1e6:.1f} MB ({pct:.1f}%)",
                                    end="",
                                    flush=True,
                                )
                print()

            if temp_path.exists() and temp_path.stat().st_size > 0:
                if total_size > 0 and temp_path.stat().st_size == total_size:
                    temp_path.replace(dest_path)
                    print(f"  [OK] Finished {dest_path.name} ({dest_path.stat().st_size / 1e6:.2f} MB)", flush=True)
                    return True
                elif total_size == 0:
                    temp_path.replace(dest_path)
                    return True
        except Exception as e:
            print(f"\n  Warning on {dest_path.name}: {e}. Resuming from byte offset...")
            time.sleep(1)

    return False


def main():
    kernel = "sohamdeshpande10/infranova-pix2pixhd-b10-b11-epoch101to250"
    target_dir = Path("kaggle_kernel/run_output")
    target_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    owner_slug, kernel_slug, _ = api.parse_kernel_string(kernel)

    token = None
    all_files = []

    while True:
        with api.build_kaggle_client() as kaggle_client:
            request = ApiListKernelSessionOutputRequest()
            request.user_name = owner_slug
            request.kernel_slug = kernel_slug
            api._set_paging(request, 50, token)
            response = kaggle_client.kernels.kernels_api_client.list_kernel_session_output(request)

            for item in response.files or []:
                all_files.append(item)

            token = response.next_page_token
            if not token:
                break

    print(f"Found {len(all_files)} files in completed kernel output.")

    # Save log if present
    if hasattr(response, "log") and response.log:
        log_path = target_dir / f"{kernel_slug}.log"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(response.log)
        print(f"Saved full kernel execution log to {log_path}")

    # High priority files first: metrics, config, final epoch
    priority_order = [
        "experiment.json",
        "config_snapshot.json",
        "training.csv",
        "epoch_250.pth",
        "best_ssim.pth",
        "best_psnr.pth",
        "best_lab.pth",
        "best_sam.pth",
        "best_sat_ratio.pth",
        "latest.pth",
        "checkpoint_epoch_250.pth",
        "checkpoint_epoch_249.pth",
        "best_checkpoint.pth",
    ]

    def get_priority(item):
        name = Path(item.file_name).name
        if name in priority_order:
            return priority_order.index(name)
        return 999

    all_files.sort(key=get_priority)

    # Small files: overwrite immediately
    for item in all_files:
        name = Path(item.file_name).name
        dest = target_dir / item.file_name
        if name in ["experiment.json", "config_snapshot.json", "training.csv"]:
            dest.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading summary metadata: {name}...")
            r = requests.get(item.url)
            with open(dest, "wb") as f:
                f.write(r.content)
            print(f"  [OK] Saved {dest} ({len(r.content)} bytes)")

    # Large checkpoint files: resumable download
    for item in all_files:
        name = Path(item.file_name).name
        if name in ["experiment.json", "config_snapshot.json", "training.csv"]:
            continue
        dest = target_dir / item.file_name
        download_file_resumable(item.url, dest)

    print("\nAll final artifacts downloaded successfully!")


if __name__ == "__main__":
    main()
