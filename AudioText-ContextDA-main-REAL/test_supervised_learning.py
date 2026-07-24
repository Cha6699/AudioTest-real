"""
监督学习模式 - 多标签声音分类
使用你的标注数据训练分类器
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import average_precision_score, accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 导入laion_clap
try:
    import laion_clap
    print("成功导入 laion_clap")
except ImportError as e:
    print(f"导入 laion_clap 失败: {e}")
    print("请运行: pip install laion-clap")
    sys.exit(1)

class SupervisedLearningEvaluator:
    def __init__(self, model_path="models/LAION-CLAP/630k-audioset-fusion-best.pt"):
        """初始化评估器"""
        print("正在加载模型...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        if not os.path.exists(model_path):
            print(f"错误: 模型文件不存在: {model_path}")
            sys.exit(1)
        
        try:
            self.model = laion_clap.CLAP_Module(enable_fusion=True)
            self.model.load_ckpt(model_path)
            if self.device == 'cuda':
                self.model = self.model.cuda()
            print("模型加载完成！")
        except Exception as e:
            print(f"模型加载失败: {e}")
            sys.exit(1)
    
    def get_audio_embedding(self, audio_path):
        """获取音频嵌入向量"""
        try:
            audio_embedding = self.model.get_audio_embedding_from_filelist([audio_path])
            audio_embedding = audio_embedding / np.linalg.norm(audio_embedding, axis=1, keepdims=True)
            return audio_embedding.squeeze()
        except Exception as e:
            print(f"处理音频 {audio_path} 时出错: {e}")
            return None
    
    def parse_label_studio_csv_multilabel(self, csv_path):
        """解析Label Studio导出的CSV文件（多标签）"""
        df = pd.read_csv(csv_path)
        print(f"\n读取CSV文件: {csv_path}")
        print(f"包含 {len(df)} 条记录")
        
        target_class_mapping = {
            'Rain': 'rain',
            'Frog': 'frog',
            'construction site': 'Construction Site',
            'Bird': 'bird',
            'bird': 'bird'
        }
        
        audio_labels_map = {}
        
        for idx, row in df.iterrows():
            audio_path = row['audio']
            filename = os.path.basename(audio_path)
            
            parts = filename.split('-', 1)
            actual_filename = parts[1] if len(parts) > 1 else filename
            
            label_str = row['label']
            try:
                label_data = json.loads(label_str)
                all_labels = []
                for item in label_data:
                    if 'labels' in item:
                        all_labels.extend(item['labels'])
                all_labels = list(set(all_labels))
                
                mapped_labels = []
                for label in all_labels:
                    if label in target_class_mapping:
                        mapped_labels.append(target_class_mapping[label])
                
                if mapped_labels:
                    mapped_labels = list(set(mapped_labels))
                    audio_labels_map[actual_filename] = mapped_labels
                    print(f"  解析: {actual_filename} -> {mapped_labels}")
                    
            except json.JSONDecodeError:
                continue
        
        print(f"\n成功解析 {len(audio_labels_map)} 个音频的标签")
        return audio_labels_map
    
    def prepare_data(self, audio_folder, label_csv, target_classes):
        """准备训练数据"""
        print(f"\n准备数据...")
        
        # 1. 解析标签
        audio_labels_map = self.parse_label_studio_csv_multilabel(label_csv)
        
        if len(audio_labels_map) == 0:
            print("错误: 没有成功解析任何标签！")
            return None, None, None
        
        # 2. 获取实际音频文件
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg']
        actual_files = {}
        for audio_file in Path(audio_folder).rglob('*'):
            if audio_file.suffix.lower() in audio_extensions:
                actual_files[audio_file.name] = str(audio_file)
        
        print(f"\n音频文件夹中有 {len(actual_files)} 个音频文件")
        
        # 3. 匹配文件
        audio_files = []
        audio_true_labels = []
        audio_filenames = []
        
        print("\n正在匹配文件...")
        for csv_filename, labels in audio_labels_map.items():
            filtered_labels = [label for label in labels if label in target_classes]
            if not filtered_labels:
                continue
            
            if csv_filename in actual_files:
                audio_files.append(actual_files[csv_filename])
                audio_true_labels.append(filtered_labels)
                audio_filenames.append(csv_filename)
                print(f"  匹配成功: {csv_filename} -> {filtered_labels}")
        
        print(f"\n成功匹配 {len(audio_files)} 个文件")
        
        if len(audio_files) == 0:
            print("错误: 没有找到任何匹配的音频文件！")
            return None, None, None
        
        # 4. 提取音频嵌入
        print("\n正在提取音频嵌入...")
        embeddings = []
        valid_labels = []
        
        for audio_path, true_labels in tqdm(zip(audio_files, audio_true_labels), total=len(audio_files)):
            embedding = self.get_audio_embedding(audio_path)
            if embedding is not None:
                embeddings.append(embedding)
                valid_labels.append(true_labels)
        
        if len(embeddings) == 0:
            print("错误: 没有成功提取任何音频的嵌入！")
            return None, None, None
        
        # 5. 转换标签为多标签格式
        mlb = MultiLabelBinarizer(classes=target_classes)
        y_multilabel = mlb.fit_transform(valid_labels)
        
        X = np.array(embeddings)
        y = y_multilabel
        
        print(f"\n成功提取 {len(X)} 个音频的嵌入")
        print(f"嵌入维度: {X.shape[1]}")
        print(f"标签维度: {y.shape[1]}")
        
        return X, y, mlb
    
    def train_and_evaluate(self, X, y, target_classes, n_folds=5):
        """训练和评估分类器（交叉验证）"""
        print(f"\n{'='*60}")
        print(f"监督学习训练 ({n_folds}-折交叉验证)")
        print(f"{'='*60}")
        
        # 使用不同的分类器
        classifiers = {
            'SVM': SVC(kernel='rbf', probability=True, random_state=42),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        }
        
        # 多标签分类需要为每个类别训练一个分类器
        results = {}
        
        for clf_name, clf in classifiers.items():
            print(f"\n训练分类器: {clf_name}")
            
            # K-Fold交叉验证
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            
            fold_results = []
            y_true_all = []
            y_pred_all = []
            y_scores_all = []
            
            for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # 为每个类别训练一个分类器（一对多策略）
                y_pred_fold = []
                y_scores_fold = []
                
                for i in range(y.shape[1]):
                    # 训练二分类器
                    clf_copy = clf.__class__(**clf.get_params())
                    clf_copy.fit(X_train, y_train[:, i])
                    
                    # 预测概率
                    if hasattr(clf_copy, 'predict_proba'):
                        scores = clf_copy.predict_proba(X_test)[:, 1]
                    else:
                        scores = clf_copy.decision_function(X_test)
                        # 归一化到0-1
                        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
                    
                    y_scores_fold.append(scores)
                    y_pred_fold.append((scores > 0.5).astype(int))
                
                y_scores_fold = np.array(y_scores_fold).T
                y_pred_fold = np.array(y_pred_fold).T
                
                # 计算指标
                fold_mAP = average_precision_score(y_test, y_scores_fold, average='macro')
                fold_accuracy = accuracy_score(
                    np.argmax(y_test, axis=1), 
                    np.argmax(y_pred_fold, axis=1)
                )
                
                fold_results.append({
                    'fold': fold + 1,
                    'mAP': fold_mAP,
                    'accuracy': fold_accuracy
                })
                
                y_true_all.append(y_test)
                y_pred_all.append(y_pred_fold)
                y_scores_all.append(y_scores_fold)
            
            # 汇总结果
            y_true_all = np.vstack(y_true_all)
            y_pred_all = np.vstack(y_pred_all)
            y_scores_all = np.vstack(y_scores_all)
            
            # 计算每个类别的AP
            per_class_ap = []
            print(f"\n  每个类别的平均精度 (AP):")
            for i, cls in enumerate(target_classes):
                if np.sum(y_true_all[:, i]) > 0:
                    ap = average_precision_score(y_true_all[:, i], y_scores_all[:, i])
                    per_class_ap.append(ap)
                    print(f"    {cls}: AP = {ap:.4f}")
                else:
                    per_class_ap.append(0.0)
                    print(f"    {cls}: 没有样本")
            
            # 计算整体mAP
            overall_mAP = np.mean(per_class_ap)
            
            # 计算准确率
            y_true_single = np.argmax(y_true_all, axis=1)
            y_pred_single = np.argmax(y_pred_all, axis=1)
            overall_accuracy = accuracy_score(y_true_single, y_pred_single)
            
            # 保存结果
            results[clf_name] = {
                'mAP': overall_mAP,
                'accuracy': overall_accuracy,
                'per_class_ap': per_class_ap,
                'fold_results': fold_results,
                'y_true': y_true_all,
                'y_pred': y_pred_all,
                'y_scores': y_scores_all
            }
            
            print(f"\n  {clf_name} 结果:")
            print(f"    mAP: {overall_mAP:.4f}")
            print(f"    准确率: {overall_accuracy:.4f}")
            
            # 显示混淆矩阵
            print(f"\n  混淆矩阵 (真实 vs 预测):")
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_true_single, y_pred_single)
            print(f"    {target_classes}")
            print(cm)
        
        return results
    
    def compare_classifiers(self, results, target_classes):
        """对比不同分类器的性能"""
        print("\n" + "="*60)
        print("分类器性能对比")
        print("="*60)
        
        print(f"\n{'分类器':<15} {'mAP':<10} {'准确率':<10}")
        print("-"*35)
        
        best_mAP = 0
        best_clf = None
        
        for clf_name, metrics in results.items():
            mAP = metrics['mAP']
            accuracy = metrics['accuracy']
            print(f"{clf_name:<15} {mAP:<10.4f} {accuracy:<10.4f}")
            
            if mAP > best_mAP:
                best_mAP = mAP
                best_clf = clf_name
        
        print(f"\n最佳分类器: {best_clf} (mAP = {best_mAP:.4f})")
        
        # 保存对比结果
        comparison_df = pd.DataFrame({
            'Classifier': list(results.keys()),
            'mAP': [results[clf]['mAP'] for clf in results],
            'Accuracy': [results[clf]['accuracy'] for clf in results]
        })
        comparison_df.to_csv('classifier_comparison.csv', index=False)
        print("\n对比结果已保存到: classifier_comparison.csv")
        
        return best_clf


def main():
    """主函数"""
    
    # ============ 配置参数 ============
    AUDIO_FOLDER = r"C:\Users\xiao.xu\Desktop\DownloadFromCloud_OverThreshold\L1_1001_OverThreshold"
    LABEL_CSV = r"C:\Users\xiao.xu\Desktop\DownloadFromCloud_OverThreshold\label\L1_1001_OverThreshold.csv"
    MODEL_PATH = "models/LAION-CLAP/630k-audioset-fusion-best.pt"
    
    TARGET_CLASSES = ['bird', 'rain', 'frog', 'Construction Site']
    N_FOLDS = 5  # 交叉验证折数
    # =================================
    
    # 检查文件
    if not os.path.exists(AUDIO_FOLDER):
        print(f"错误: 音频文件夹不存在: {AUDIO_FOLDER}")
        return
    
    if not os.path.exists(LABEL_CSV):
        print(f"错误: 标签文件不存在: {LABEL_CSV}")
        return
    
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型文件不存在: {MODEL_PATH}")
        return
    
    # 创建评估器
    evaluator = SupervisedLearningEvaluator(MODEL_PATH)
    
    # 准备数据
    X, y, mlb = evaluator.prepare_data(
        audio_folder=AUDIO_FOLDER,
        label_csv=LABEL_CSV,
        target_classes=TARGET_CLASSES
    )
    
    if X is None:
        return
    
    # 训练和评估
    results = evaluator.train_and_evaluate(X, y, TARGET_CLASSES, N_FOLDS)
    
    # 对比分类器
    best_clf = evaluator.compare_classifiers(results, TARGET_CLASSES)
    
    # 最终总结
    print("\n" + "="*60)
    print("监督学习总结")
    print("="*60)
    print(f"数据样本数: {len(X)}")
    print(f"类别数: {len(TARGET_CLASSES)}")
    print(f"交叉验证折数: {N_FOLDS}")
    print(f"最佳分类器: {best_clf}")
    print(f"最佳mAP: {results[best_clf]['mAP']:.4f}")
    print(f"最佳准确率: {results[best_clf]['accuracy']:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()