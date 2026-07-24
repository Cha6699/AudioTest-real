"""
ESC-50 Processing - Using HuggingFace Transformers
"""

import os
import sys
import torch
import torchaudio
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

print("=" * 70)
print("ESC-50 Processing - HuggingFace Transformers")
print("=" * 70)

# ============================================================================
# STEP 1: 检查数据
# ============================================================================
print("\n[STEP 1] Checking ESC-50 data...")

data_dir = Path("data/input/esc50")
csv_path = data_dir / "meta" / "esc50.csv"
audio_dir = data_dir / "audio"

if not csv_path.exists():
    print(f"✗ ERROR: {csv_path} not found!")
    sys.exit(1)

df = pd.read_csv(csv_path)
print(f"✓ CSV loaded: {len(df)} samples")
print(f"  Categories: {df['category'].nunique()}")
print(f"  Folds: {sorted(df['fold'].unique())}")

# ============================================================================
# STEP 2: 加载CLAP模型 - HuggingFace
# ============================================================================
print("\n[STEP 2] Loading CLAP model from HuggingFace...")

try:
    from transformers import ClapModel, ClapProcessor
    
    # 使用与模型匹配的版本
    model_name = "laion/clap-htsat-fused"
    
    print(f"Loading {model_name}...")
    print("This may take a few minutes (downloading model)...")
    
    model = ClapModel.from_pretrained(model_name)
    processor = ClapProcessor.from_pretrained(model_name)
    
    print("✓ Model loaded successfully!")
    print(f"  Audio embedding dimension: {model.config.projection_dim}")
    
except ImportError:
    print("✗ transformers not installed")
    print("Install: pip install transformers")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error loading model: {e}")
    sys.exit(1)

# ============================================================================
# STEP 3: 提取音频嵌入 (测试模式)
# ============================================================================
print("\n[STEP 3] Extracting audio embeddings...")

test_mode = False
if test_mode:
    process_count = min(10, len(df))
    print(f"TEST MODE: Processing first {process_count} samples")
    df_process = df.head(process_count)
else:
    process_count = len(df)
    print(f"Processing all {process_count} samples")
    df_process = df

embeddings_list = []
labels_list = []
folds_list = []
filenames_list = []

for idx, row in tqdm(df_process.iterrows(), total=len(df_process), desc="Extracting"):
    filename = row['filename']
    audio_path = audio_dir / filename
    
    if not audio_path.exists():
        print(f"⚠ File not found: {filename}")
        continue
    
    try:
        # 加载音频
        audio, sr = torchaudio.load(str(audio_path))
        
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        # 重采样到48000Hz
        if sr != 48000:
            resampler = torchaudio.transforms.Resample(sr, 48000)
            audio = resampler(audio)
        
        # 确保至少1秒
        if audio.shape[1] < 48000:
            pad_len = 48000 - audio.shape[1]
            audio = torch.nn.functional.pad(audio, (0, pad_len))
        
        # 转换为numpy并确保是float32
        audio_np = audio.numpy().astype(np.float32)
        if audio_np.max() > 1.0:
            audio_np = audio_np / 32768.0
        
        # 使用processor处理音频
        inputs = processor(
            audios=audio_np, 
            sampling_rate=48000, 
            return_tensors="pt"
        )
        
        # 提取嵌入
        with torch.no_grad():
            audio_embed = model.get_audio_features(**inputs)
            if len(audio_embed.shape) > 1:
                audio_embed = audio_embed.squeeze()
        
        embeddings_list.append(audio_embed.cpu().numpy())
        labels_list.append(int(row['target']))
        folds_list.append(int(row['fold']))
        filenames_list.append(filename)
        
    except Exception as e:
        print(f"✗ Error processing {filename}: {e}")
        continue

# ============================================================================
# STEP 4: 保存嵌入
# ============================================================================
print("\n[STEP 4] Saving embeddings...")

if len(embeddings_list) > 0:
    embeddings_array = np.vstack(embeddings_list)
    
    save_data = {
        'embeddings': torch.tensor(embeddings_array),
        'labels': torch.tensor(labels_list),
        'folds': torch.tensor(folds_list),
        'filenames': filenames_list,
        'dataset': 'esc50',
        'num_classes': len(df['category'].unique())
    }
    
    Path("data/embeddings").mkdir(parents=True, exist_ok=True)
    
    if test_mode:
        output_path = "data/embeddings/esc50_test.pt"
    else:
        output_path = "data/embeddings/esc50_full.pt"
    
    torch.save(save_data, output_path)
    
    print(f"✓ Embeddings saved to: {output_path}")
    print(f"  Shape: {embeddings_array.shape}")
    print(f"  Samples: {len(labels_list)}")
else:
    print("✗ No embeddings extracted!")

# ============================================================================
# STEP 5: 创建类别标签文件
# ============================================================================
print("\n[STEP 5] Creating class labels file...")

class_labels = df.sort_values('target')['category'].unique().tolist()

label_file = Path("class_labels_esc50.txt")
with open(label_file, 'w') as f:
    for label in class_labels:
        f.write(label + '\n')

print(f"✓ Class labels saved to: {label_file}")
print(f"  Total: {len(class_labels)} classes")

print("\n" + "=" * 70)
print("✓ ESC-50 Processing Complete!")
print("=" * 70)
print("\nNext steps:")
print("1. To extract all data, set test_mode = False")
print("2. Run classification: python classify_esc50.py")
print("=" * 70)