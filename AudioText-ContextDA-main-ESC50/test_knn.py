"""
ESC-50 KNN Classification
"""

import os
import torch
import numpy as np
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

print("=" * 70)
print("ESC-50 KNN Classification")
print("=" * 70)

# 创建结果目录
results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = results_dir / f"knn_{timestamp}"
run_dir.mkdir(parents=True, exist_ok=True)

# 加载数据
embed_path = Path("data/embeddings/esc50_full.pt")
if not embed_path.exists():
    print(f"✗ Embeddings not found!")
    sys.exit(1)

data = torch.load(embed_path, map_location='cpu')
embeddings = data['embeddings']
labels = data['labels']
folds = data['folds']

print(f"Loaded {len(embeddings)} samples")
print(f"Embedding shape: {embeddings.shape}")

# 加载类别标签
label_file = Path("class_labels_esc50.txt")
if label_file.exists():
    with open(label_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
else:
    class_names = [str(i) for i in range(50)]

# KNN 交叉验证
print("\n" + "-" * 70)
print("KNN Classification (5-fold Cross Validation)")
print("-" * 70)

fold_accuracies = []
fold_results = []

for fold in [1, 2, 3, 4, 5]:
    test_mask = folds == fold
    train_mask = ~test_mask
    
    train_emb = embeddings[train_mask].numpy()
    train_labels = labels[train_mask].numpy()
    test_emb = embeddings[test_mask].numpy()
    test_labels = labels[test_mask].numpy()
    
    # 标准化
    scaler = StandardScaler()
    train_emb_scaled = scaler.fit_transform(train_emb)
    test_emb_scaled = scaler.transform(test_emb)
    
    # KNN
    knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
    knn.fit(train_emb_scaled, train_labels)
    
    pred_labels = knn.predict(test_emb_scaled)
    acc = accuracy_score(test_labels, pred_labels)
    fold_accuracies.append(acc)
    
    print(f"Fold {fold}: {acc:.4f} ({len(test_labels)} samples)")

# 总结
mean_acc = np.mean(fold_accuracies)
std_acc = np.std(fold_accuracies)

print("-" * 70)
print(f"Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
print("=" * 70)

# 保存结果
results = {
    'method': 'KNN',
    'timestamp': timestamp,
    'fold_accuracies': [float(a) for a in fold_accuracies],
    'mean_accuracy': float(mean_acc),
    'std_accuracy': float(std_acc),
    'num_samples': len(embeddings),
    'embedding_dim': embeddings.shape[1],
    'num_classes': 50
}

# 保存为JSON
import json
with open(run_dir / 'results.json', 'w') as f:
    json.dump(results, f, indent=2)

# 保存为CSV
df = pd.DataFrame({
    'fold': [1, 2, 3, 4, 5],
    'accuracy': fold_accuracies
})
df.to_csv(run_dir / 'results.csv', index=False)

# 保存详细报告
with open(run_dir / 'report.txt', 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("ESC-50 KNN Classification Results\n")
    f.write("=" * 70 + "\n")
    f.write(f"\nTimestamp: {timestamp}")
    f.write(f"\nTotal Samples: {len(embeddings)}")
    f.write(f"\nEmbedding Dimension: {embeddings.shape[1]}")
    f.write(f"\nNumber of Classes: 50")
    f.write(f"\n\nFold-wise Accuracies:")
    for i, acc in enumerate(fold_accuracies, 1):
        f.write(f"\n  Fold {i}: {acc:.4f}")
    f.write(f"\n\nMean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    f.write("\n" + "=" * 70)

print(f"\n✓ Results saved to: {run_dir}")
print(f"  - results.json")
print(f"  - results.csv")
print(f"  - report.txt")