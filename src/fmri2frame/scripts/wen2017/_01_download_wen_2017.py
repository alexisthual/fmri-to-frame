"""Download fMRI and stimuli from WEN2017 dataset.

Each downloaded folder contains a bundle.zip file
which needs to be manually unzipped after running this script.
Moreover, one file needs to be renamed for subject 1:
10_4231_R7X63K3M/video_fmri_dataset/subject1/fmri/test1/mni/test1_9.mni.nii.gz
should be named test1_9_mni.nii.gz
"""

# %%
import os
from ftplib import FTP
from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm import tqdm


# %%
def download_file(url, destination):
    parsed_url = urlparse(url)

    if parsed_url.scheme == "http" or parsed_url.scheme == "https":
        download_http_file(url, destination)
    elif parsed_url.scheme == "ftp":
        download_ftp_file(parsed_url, destination)
    else:
        print("Unsupported URL scheme. Only HTTP and FTP are supported.")


def download_http_file(url, destination):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024 * 1024  # 1 MB
    t = tqdm(total=total_size, unit="B", unit_scale=True)

    with open(destination, "wb") as file:
        for data in response.iter_content(block_size):
            t.update(len(data))
            file.write(data)

    t.close()

    if total_size != 0 and t.n != total_size:
        print("HTTP download failed. Please try again.")
    else:
        print("HTTP download completed successfully.")


def download_ftp_file(parsed_url, destination):
    ftp_host = parsed_url.hostname
    remote_path = parsed_url.path

    ftp = FTP(ftp_host)
    ftp.login()
    total_size = ftp.size(remote_path)

    block_size = 1024 * 1024  # 1 MB
    t = tqdm(total=total_size, unit="B", unit_scale=True)

    with open(destination, "wb") as local_file:

        def callback(data):
            t.update(len(data))
            local_file.write(data)

        ftp.retrbinary("RETR " + remote_path, callback, block_size)

    t.close()
    ftp.quit()

    if total_size != 0 and t.n != total_size:
        print("FTP download failed. Please try again.")
    else:
        print("FTP download completed successfully.")


# %%
if __name__ == "__main__":
    # %%
    all_outputs = Path("/storage/store2/data/wen2017")
    all_outputs.mkdir(parents=True, exist_ok=True)
    os.chmod(all_outputs, 0o777)

    # %%
    dataset_urls = {
        "stimuli": (
            "https://purr.purdue.edu/publications/2808/serve/1?render=archive"
        ),
        "subject1": "ftp://purr.purdue.edu/10_4231_R7X63K3M.zip",
        "subject2": "ftp://purr.purdue.edu/10_4231_R7NS0S1F.zip",
        "subject3": "ftp://purr.purdue.edu/10_4231_R7J101BV.zip",
    }

    dataset_folders = {
        "stimuli": "10_4231_R71Z42KK",
        "subject1": "10_4231_R7X63K3M",
        "subject2": "10_4231_R7NS0S1F",
        "subject3": "10_4231_R7J101BV",
    }

    # %%
    for key, url in dataset_urls.items():
        output_file = all_outputs / f"{dataset_folders[key]}.zip"

        print(f"Downloading {key} from {url}")
        if not output_file.exists():
            download_file(url, output_file)
        else:
            print(f"File {key}.zip already exists, skipping download")

        print()
