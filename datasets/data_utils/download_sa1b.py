import os
import tarfile
from multiprocessing import Pool
import argparse
import requests


def download_and_extract(args, skip_existing=False):
    file_name, url, raw_dir, images_dir, masks_dir = args
    
    # Check if the file already exists
    if not os.path.exists(f'{raw_dir}/{file_name}'):
        # Download the file
        print(f'Downloading {file_name} from {url}...')
        response = requests.get(url, stream=True)
        with open(f'{raw_dir}/{file_name}', 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        print(f'{file_name} already exists in {raw_dir}. Skipping download.')

    # Extract the file if it's a .tar file
    if file_name.endswith('.tar'):
        # Check if the file has already been extracted
        if os.path.exists(f'{images_dir}/{os.path.splitext(file_name)[0]}/') and os.path.exists(f'{masks_dir}/{os.path.splitext(file_name)[0]}/') and skip_existing:
            print(f'{file_name} has already been extracted. Skipping extraction.')
        else:
            print(f'Extracting {file_name}...')
            with tarfile.open(f'{raw_dir}/{file_name}') as tar:
                for member in tar.getmembers():
                    if member.name.endswith(".jpg"):
                        tar.extract(member, path=images_dir)
                    elif member.name.endswith(".json"):
                        tar.extract(member, path=masks_dir)
                
            print(f'{file_name} extracted!')
    else:
        print(f'{file_name} is not a tar file. Skipping extraction.')


def extract(args, skip_existing=False):
    file_name, url, raw_dir, images_dir, masks_dir = args

    # Check if the file already exists
    if os.path.exists(f'{raw_dir}/{file_name}'):
        print(f'{file_name} already exists in {raw_dir}. Skipping download.')

        # Extract the file if it's a .tar file
        if file_name.endswith('.tar'):
            # Check if the file has already been extracted
            if os.path.exists(f'{images_dir}/{os.path.splitext(file_name)[0]}/') and os.path.exists(
                    f'{masks_dir}/{os.path.splitext(file_name)[0]}/') and skip_existing:
                print(f'{file_name} has already been extracted. Skipping extraction.')
            else:
                print(f'Extracting {file_name}...')
                with tarfile.open(f'{raw_dir}/{file_name}') as tar:
                    for member in tar.getmembers():
                        if member.name.endswith(".jpg"):
                            tar.extract(member, path=images_dir)
                        elif member.name.endswith(".json"):
                            tar.extract(member, path=masks_dir)

                print(f'{file_name} extracted!')
        else:
            print(f'{file_name} is not a tar file. Skipping extraction.')

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Download and extract files.')
parser.add_argument('--processes', type=int, default=4, help='Number of processes to use for downloading and extracting files.')
parser.add_argument('--input_file', type=str, default='sa1b_links.txt', help='Path to the input file containing file names and URLs.')
parser.add_argument('--root_dir', type=str, default='datasets', help='Directory to store downloaded files.')
parser.add_argument('--raw_dir', type=str, default='raw', help='Directory to store downloaded files.')
parser.add_argument('--images_dir', type=str, default='images', help='Directory to store extracted image files.')
parser.add_argument('--masks_dir', type=str, default='annotations', help='Directory to store extracted JSON mask files.')
parser.add_argument('--skip_existing', action='store_true', help='Skip extraction if the file has already been extracted')
args = parser.parse_args()

# Read the file names and URLs
with open(args.input_file, 'r') as f:
    lines = f.readlines()[1:]

args.raw_dir = os.path.join(args.root_dir, args.raw_dir)
args.images_dir = os.path.join(args.root_dir, args.images_dir)
args.masks_dir = os.path.join(args.root_dir, args.masks_dir)

# Create the directories if they do not exist
os.makedirs(args.raw_dir, exist_ok=True)
os.makedirs(args.images_dir, exist_ok=True)
os.makedirs(args.masks_dir, exist_ok=True)

# Download and extract the files in parallel
with Pool(processes=args.processes) as pool:
    pool.starmap(extract, [(line.strip().split('\t') + [args.raw_dir, args.images_dir, args.masks_dir], args.skip_existing) for line in lines])

    # download & extract
    # pool.starmap(download_and_extract,
    #              [(line.strip().split('\t') + [args.raw_dir, args.images_dir, args.masks_dir], args.skip_existing) for
    #               line in lines])

print('All files downloaded successfully!')    
