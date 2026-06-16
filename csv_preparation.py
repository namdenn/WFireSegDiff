import csv
from pathlib import Path


TARGET_BASE_DIR = Path("/srv/storage/talc3@storage4.nancy.grid5000.fr/multispeech/calcul/users/ptuannam/WFireSegDiff/ISIC")

SETS = {
    "Train": {
        "data_dir": TARGET_BASE_DIR / "Train" / "ISBI2016_ISIC_Part1_Training_Data",
        "csv_path": TARGET_BASE_DIR / "Train" / "ISBI2016_ISIC_Part1_Training_GroundTruth.csv"
    },
    "Test": {
        "data_dir": TARGET_BASE_DIR / "Test" / "ISBI2016_ISIC_Part1_Test_Data",
        "csv_path": TARGET_BASE_DIR / "Test" / "ISBI2016_ISIC_Part1_Test_GroundTruth.csv"
    }
}

for set_name, paths in SETS.items():
    data_dir = paths["data_dir"]
    csv_path = paths["csv_path"]
    
    if not data_dir.exists():
        print(f"Skipping {set_name}: Directory {data_dir} does not exist.")
        continue

    csv_rows = []

    for index, file_name in enumerate(sorted(data_dir.glob("*.jpg"))):
        img_id = file_name.stem  
        
        img_file = f"{img_id}.jpg"
        seg_file = f"{img_id}_Segmentation.png"
        
        csv_rows.append([index, img_file, seg_file])

    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        
        writer.writerow(["", "img", "seg"]) 
        writer.writerows(csv_rows)
        
    print(f"✓ Successfully rewrote {set_name} CSV with index column at: {csv_path} ({len(csv_rows)} rows)")