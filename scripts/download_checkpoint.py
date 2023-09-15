import requests
import os

def download_checkpoint(source_url, dest_folder, dest_filename, overwrite_existing=False):

    os.makedirs(dest_folder, exist_ok=True)
    filepath = os.path.join(dest_folder, dest_filename)

    if os.path.exists(filepath) and not overwrite_existing:
        print("found existing checkpoint and `overwrite_existing` is False. Skipping download...")
    else:
        print("downloading checkpoints ......")
        response = requests.get(source_url, stream=True)
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print("download successfully!")

    return filepath

if __name__ == '__main__':
    download_checkpoint(
        "https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem-s01.pth",
        './XMem/saves',
        'XMem-s01.pth'
    )