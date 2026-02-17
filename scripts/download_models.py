import os
import requests
from tqdm import tqdm

MODELS = {
    "gfpgan": {
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
        "path": "weights/GFPGANv1.3.pth"
    },
    "nafnet": {
        "url": "https://github.com/megvii-research/NAFNet/releases/download/v1.0/NAFNet-RedS-DeBlur.pth",
        "path": "weights/NAFNet-DeBlur.pth"
    },
    "rmbg": {
        "url": "https://huggingface.co/briaai/RMBG-2.0/resolve/main/model.safetensors",
        "path": "weights/rmbg_2.0.safetensors"
    }
}

def download_file(url: str, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        print(f"File already exists: {path}")
        return

    print(f"Downloading {url} to {path}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(path, "wb") as f, tqdm(
        total=total_size, unit='iB', unit_scale=True, desc=os.path.basename(path)
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def main():
    # Only try to download if requests is installed
    try:
        for name, info in MODELS.items():
            download_file(info["url"], info["path"])
    except Exception as e:
        print(f"Error downloading models: {e}")
        print("Please ensure 'requests' and 'tqdm' are installed.")

if __name__ == "__main__":
    main()
