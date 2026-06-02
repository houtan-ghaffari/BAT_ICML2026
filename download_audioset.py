import os
import shutil
import zipfile
import time
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
from requests.exceptions import ReadTimeout, ConnectionError
from tqdm import tqdm


def download_and_extract(target_dir='/home/Datasets/AudioSet', download_unbalanced=False):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    repo_id = "confit/audioset-full"

    # history of the completed files to avoid re-downloading deleted zips
    progress_file = target_dir / "extracted_zips.txt"
    extracted_zips = set()
    if progress_file.exists():
        with open(progress_file, "r") as f:
            extracted_zips = set(line.strip() for line in f if line.strip())

    api = HfApi()
    all_repo_files = api.list_repo_files(repo_id, repo_type="dataset")

    files_to_download = []
    for f in all_repo_files:
        if f.endswith('.csv') or f.endswith('.json'):
            files_to_download.append(f)
        elif f.startswith('eval/') and f.endswith('.zip'):
            files_to_download.append(f)
        elif f.startswith('balanced/') and f.endswith('.zip'):
            files_to_download.append(f)
        elif download_unbalanced and f.startswith('unbalanced/') and f.endswith('.zip'):
            files_to_download.append(f)

    folder_mapping = {'eval': target_dir / 'eval_segments',
                      'balanced': target_dir / 'balanced_train_segments',
                      'unbalanced': target_dir / 'unbalanced_train_segments'}

    for f in folder_mapping.values():
        f.mkdir(parents=True, exist_ok=True)

    print(f"\ndownloading and extracting ({len(files_to_download)} files)...")

    for file_path in tqdm(files_to_download):

        if file_path in extracted_zips:
            continue

        success = False
        local_file_path = None

        while not success:
            try:
                local_file_path = hf_hub_download(repo_id=repo_id, filename=file_path, repo_type="dataset",
                                                  local_dir=target_dir, local_dir_use_symlinks=False,
                                                  resume_download=True)
                success = True

            except (ReadTimeout, ConnectionError) as e:
                print(f"Network error: {e}. Retrying in 10s...")
                time.sleep(10)

            except Exception as e:
                print(f"Unexpected error: {e}. Retrying in 10s...")
                time.sleep(10)

        if local_file_path and local_file_path.endswith('.zip'):
            split_name = file_path.split('/')[0]
            dest_path = folder_mapping[split_name]

            try:
                with zipfile.ZipFile(local_file_path, 'r') as zf:
                    wav_files = [m for m in zf.namelist() if m.endswith('.wav')]
                    for member in wav_files:
                        filename = os.path.basename(member)
                        out_file = dest_path / filename

                        if out_file.exists() and out_file.stat().st_size > 0:
                            continue

                        try:
                            with zf.open(member) as source, open(out_file, "wb") as target:
                                shutil.copyfileobj(source, target)

                        except zipfile.BadZipFile as e:
                            print(f"\n[!] Corrupted file skipped: {filename} ({e})")
                            if out_file.exists():
                                out_file.unlink()  # delete the broken file

                        except Exception as e:
                            print(f"\n[!] Unexpected error extracting {filename}: {e}")
                            if out_file.exists():
                                out_file.unlink()

                print(f"Extracted {local_file_path} to {dest_path.name}.")

                try:
                    os.remove(local_file_path)
                    print(f"Deleted temporary zip file: {local_file_path}")

                except Exception as e:
                    print(f"Failed to delete {local_file_path}: {e}")

                # save to progress file so it won't download again on restart
                with open(progress_file, "a") as f:
                    f.write(f"{file_path}\n")
                extracted_zips.add(file_path)

            except zipfile.BadZipFile as e:
                print(f"\n[!] Entire zip file is corrupted: {local_file_path} ({e}). Deleting to retry later.")
                if os.path.exists(local_file_path):
                    os.remove(local_file_path)

    print(f"\nAll files are downloaded and extracted into {target_dir}.")


if __name__ == '__main__':
    download_and_extract(download_unbalanced=False)
