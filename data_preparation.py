import os
import shutil
import re
import csv
from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image

SOURCE_DIR = Path("/srv/storage/talc@storage4.nancy.grid5000.fr/multispeech/corpus/audio_visual/corsican/all_db_fire")
TARGET_BASE_DIR = Path("/srv/storage/talc3@storage4.nancy.grid5000.fr/multispeech/calcul/users/ptuannam/WFireSegDiff/ISIC")

TEST_SIZE = 0.2
RANDOM_SEED = 42

DIRS = {
    "train_data": TARGET_BASE_DIR / "Train" / "ISBI2016_ISIC_Part1_Training_Data",
    "train_gt": TARGET_BASE_DIR / "Train" / "ISBI2016_ISIC_Part1_Training_GroundTruth",
    "test_data": TARGET_BASE_DIR / "Test" / "ISBI2016_ISIC_Part1_Test_Data",
    "test_gt": TARGET_BASE_DIR / "Test" / "ISBI2016_ISIC_Part1_Test_GroundTruth",
}

CSV_PATHS = {
    "train": TARGET_BASE_DIR / "Train" / "ISBI2016_ISIC_Part1_Training_GroundTruth.csv",
    "test": TARGET_BASE_DIR / "Test" / "ISBI2016_ISIC_Part1_Test_GroundTruth.csv",
}


def setup_directories():
    """Creates the target directory tree if it doesn't exist."""
    for folder in DIRS.values():
        folder.mkdir(parents=True, exist_ok=True)
    print("✓ Target directory structure verified/created.")

def pair_images(source_path):
    """
    Scans the directory and pairs RGB images with their corresponding GT masks.
    Filters out NIR images.
    """
    all_files = os.listdir(source_path)
    rgb_dict = {}
    gt_dict = {}
    
    rgb_pattern = re.compile(r'^(.*?)_rgb(?:_(\d+))?\.png$')
    gt_pattern = re.compile(r'^(.*?)_gt(?:_(\d+))?\.png$')
    
    for f in all_files:
        if not f.endswith('.png'):
            continue
            
        rgb_match = rgb_pattern.match(f)
        if rgb_match:
            base, num = rgb_match.groups()
            key = f"{base}_{num}" if num else base
            rgb_dict[key] = f
            continue
            
        gt_match = gt_pattern.match(f)
        if gt_match:
            base, num = gt_match.groups()
            key = f"{base}_{num}" if num else base
            gt_dict[key] = f

    valid_pairs = []
    for key in rgb_dict:
        if key in gt_dict:
            valid_pairs.append({
                "id": key,
                "rgb_file": rgb_dict[key],
                "gt_file": gt_dict[key]
            })
            
    print(f"✓ Found {len(valid_pairs)} matching RGB-Mask pairs. (Ignored un-paired or NIR files)")
    return valid_pairs

def process_and_copy(pairs, dataset_type):
    """Copies, renames, converts RGB format, and logs metadata for CSV generation."""
    data_dir = DIRS[f"{dataset_type}_data"]
    gt_dir = DIRS[f"{dataset_type}_gt"]
    
    csv_rows = []
    
    for pair in pairs:
        clean_id = f"ISIC_{pair['id']}"
        
        target_rgb_name = f"{clean_id}.jpg"
        target_gt_name = f"{clean_id}_Segmentation.png"
        
        src_rgb_path = SOURCE_DIR / pair['rgb_file']
        src_gt_path = SOURCE_DIR / pair['gt_file']
        
        dst_rgb_path = data_dir / target_rgb_name
        dst_gt_path = gt_dir / target_gt_name
        
        try:
            with Image.open(src_rgb_path) as img:
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                img.save(dst_rgb_path, "JPEG", quality=95)
        except Exception as e:
            print(f"Failed converting {src_rgb_path}: {e}")
            continue
            
        shutil.copy2(src_gt_path, dst_gt_path)
    
        csv_rows.append([clean_id, "1.0" if dataset_type == "train" else "0.0"]) 

    csv_file_path = CSV_PATHS[dataset_type]
    with open(csv_file_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["ISIC_id", "label"]) 
        writer.writerows(csv_rows)
        
    print(f"✓ Finished processing {len(pairs)} samples for {dataset_type.upper()}. CSV generated.")

if __name__ == "__main__":
    setup_directories()
    
    all_pairs = pair_images(SOURCE_DIR)
    
    if not all_pairs:
        print("Error: No valid RGB/GT matching pairs found. Check your file patterns.")
        exit()
        
    train_pairs, test_pairs = train_test_split(
        all_pairs, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_SEED
    )
    
    print(f"--- Splitting Dataset (Ratio: {100*(1-TEST_SIZE)}/{100*TEST_SIZE}) ---")
    print(f"Training samples: {len(train_pairs)}")
    print(f"Testing samples:  {len(test_pairs)}")
    print("---------------------------------------")
    
    print("Processing Training data...")
    process_and_copy(train_pairs, "train")
    
    print("\nProcessing Testing data...")
    process_and_copy(test_pairs, "test")
    
    print("🎉 All done! Your new ISIC dataset structure is ready at:")
    print(TARGET_BASE_DIR)