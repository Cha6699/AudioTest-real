"""
零样本分类 + 多标签mAP评估脚本
处理一个音频有多个标签的情况
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import average_precision_score, accuracy_score, classification_report
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 导入laion_clap
try:
    import laion_clap
    print("成功导入 laion_clap")
except ImportError as e:
    print(f"导入 laion_clap 失败: {e}")
    print("请运行以下命令安装:")
    print("  pip install laion-clap")
    sys.exit(1)

class ZeroShotEvaluator:
    def __init__(self, model_path="models/LAION-CLAP/630k-audioset-fusion-best.pt"):
        """初始化零样本分类评估器"""
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
    
    def get_text_embeddings(self, class_names):
        """获取所有类别的文本嵌入向量"""
        print(f"正在生成 {len(class_names)} 个类别的文本嵌入...")
        text_prompts = [f"The sound of {name}" for name in class_names]
        
        try:
            text_embeddings = self.model.get_text_embedding(text_prompts)
            text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
            return text_embeddings
        except Exception as e:
            print(f"获取文本嵌入失败: {e}")
            return None
    
    def get_audio_embedding(self, audio_path):
        """获取单个音频的嵌入向量"""
        try:
            audio_embedding = self.model.get_audio_embedding_from_filelist([audio_path])
            audio_embedding = audio_embedding / np.linalg.norm(audio_embedding, axis=1, keepdims=True)
            return audio_embedding.squeeze()
        except Exception as e:
            print(f"处理音频 {audio_path} 时出错: {e}")
            return None
    
    def parse_label_studio_csv_multilabel(self, csv_path):
        """
        解析Label Studio导出的CSV文件 - 多标签版本
        提取每个音频的所有标签
        """
        df = pd.read_csv(csv_path)
        print(f"\n读取CSV文件: {csv_path}")
        print(f"包含 {len(df)} 条记录")
        
        # 目标类别映射
        target_class_mapping = {
            'Rain': 'rain',
            'Frog': 'frog',
            'construction site': 'Construction Site',
            'Bird': 'bird',
            'bird': 'bird'
        }
        
        audio_labels_map = {}  # 存储 {文件名: [标签1, 标签2, ...]}
        
        for idx, row in df.iterrows():
            # 提取音频文件名
            audio_path = row['audio']
            filename = os.path.basename(audio_path)
            
            # 去掉UUID前缀
            parts = filename.split('-', 1)
            if len(parts) > 1:
                actual_filename = parts[1]
            else:
                actual_filename = filename
            
            # 解析label字段
            label_str = row['label']
            try:
                label_data = json.loads(label_str)
                
                # 提取所有标签
                all_labels = []
                for item in label_data:
                    if 'labels' in item:
                        all_labels.extend(item['labels'])
                
                # 去重
                all_labels = list(set(all_labels))
                
                # 映射到目标类别
                mapped_labels = []
                for label in all_labels:
                    if label in target_class_mapping:
                        mapped_labels.append(target_class_mapping[label])
                
                # 如果有目标标签，保存
                if mapped_labels:
                    # 去重
                    mapped_labels = list(set(mapped_labels))
                    audio_labels_map[actual_filename] = mapped_labels
                    print(f"  解析: {actual_filename} -> {mapped_labels} (原始标签: {all_labels})")
                else:
                    print(f"  跳过: {actual_filename} (没有目标标签)")
                    
            except json.JSONDecodeError as e:
                print(f"  警告: 无法解析第 {idx} 行的label")
                continue
        
        print(f"\n成功解析 {len(audio_labels_map)} 个音频的标签")
        return audio_labels_map
    
    def evaluate_folder(self, audio_folder, label_csv, target_classes=['bird', 'rain', 'frog', 'Construction Site']):
        """评估文件夹中的所有音频 - 多标签版本"""
        print(f"\n开始评估...")
        print(f"音频文件夹: {audio_folder}")
        print(f"标签文件: {label_csv}")
        print(f"目标类别: {target_classes}")
        
        # 1. 解析标签文件（多标签）
        audio_labels_map = self.parse_label_studio_csv_multilabel(label_csv)
        
        if len(audio_labels_map) == 0:
            print("错误: 没有成功解析任何标签！")
            return None, None
        
        # 2. 获取实际音频文件列表
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg']
        actual_files = {}
        for audio_file in Path(audio_folder).rglob('*'):
            if audio_file.suffix.lower() in audio_extensions:
                actual_files[audio_file.name] = str(audio_file)
        
        print(f"\n音频文件夹中有 {len(actual_files)} 个音频文件")
        
        # 3. 匹配文件名并准备数据
        audio_files = []
        audio_true_labels = []  # 存储每个音频的标签列表
        audio_filenames = []
        unmatched = []
        
        print("\n正在匹配文件...")
        for csv_filename, labels in audio_labels_map.items():
            # 只保留目标类别中的标签
            filtered_labels = [label for label in labels if label in target_classes]
            
            if not filtered_labels:
                continue
            
            if csv_filename in actual_files:
                audio_files.append(actual_files[csv_filename])
                audio_true_labels.append(filtered_labels)
                audio_filenames.append(csv_filename)
                print(f"  匹配成功: {csv_filename} -> {filtered_labels}")
            else:
                unmatched.append(csv_filename)
        
        print(f"\n成功匹配 {len(audio_files)} 个文件")
        
        if len(audio_files) == 0:
            print("\n错误: 没有找到任何匹配的音频文件！")
            print("\n调试信息:")
            print("CSV中的文件名（解析后）:")
            for name in list(audio_labels_map.keys())[:5]:
                print(f"  - {name}")
            print("\n实际文件夹中的文件名:")
            for name in list(actual_files.keys())[:5]:
                print(f"  - {name}")
            return None, None
        
        # 4. 获取文本嵌入
        text_embeddings = self.get_text_embeddings(target_classes)
        if text_embeddings is None:
            print("错误: 无法获取文本嵌入！")
            return None, None
        
        # 5. 提取音频嵌入并预测
        print("\n正在提取音频嵌入并预测...")
        y_true_multilabel = []  # 存储多标签的one-hot编码
        y_pred_scores = []  # 存储每个音频的预测分数
        
        for audio_path, true_labels in tqdm(zip(audio_files, audio_true_labels), total=len(audio_files)):
            audio_embedding = self.get_audio_embedding(audio_path)
            
            if audio_embedding is None:
                continue
            
            # 计算相似度
            similarities = np.dot(text_embeddings, audio_embedding)
            
            # 存储预测分数
            y_pred_scores.append(similarities)
            
            # 存储真实标签（多标签的one-hot编码）
            true_one_hot = [1 if cls in true_labels else 0 for cls in target_classes]
            y_true_multilabel.append(true_one_hot)
        
        if len(y_true_multilabel) == 0:
            print("错误: 没有成功提取任何音频的嵌入！")
            return None, None
        
        # 转换为numpy数组
        y_true_multilabel = np.array(y_true_multilabel)
        y_pred_scores = np.array(y_pred_scores)
        
        # 6. 计算多标签评估指标
        print("\n" + "="*60)
        print("多标签评估结果")
        print("="*60)
        
        # 计算每个类别的AP (Average Precision)
        print("\n每个类别的平均精度 (AP):")
        average_precisions = []
        for i, cls in enumerate(target_classes):
            if np.sum(y_true_multilabel[:, i]) > 0:
                ap = average_precision_score(y_true_multilabel[:, i], y_pred_scores[:, i])
                average_precisions.append(ap)
                print(f"  {cls}: AP = {ap:.4f} (样本数: {np.sum(y_true_multilabel[:, i])})")
            else:
                print(f"  {cls}: 没有样本，跳过")
                average_precisions.append(0.0)
        
        # 计算mAP
        if len(average_precisions) > 0:
            mAP = np.mean([ap for ap in average_precisions if ap > 0])
            print(f"\n平均精度均值 (mAP): {mAP:.4f}")
        else:
            mAP = 0.0
        
        # 计算每个样本的预测标签（取最高分）
        y_pred_labels_idx = np.argmax(y_pred_scores, axis=1)
        y_pred_single = [target_classes[idx] for idx in y_pred_labels_idx]
        
        # 计算每个样本的真实主标签（取第一个标签，用于显示）
        y_true_first = [labels[0] for labels in audio_true_labels]
        
        # 单标签准确率（用于参考）
        accuracy = accuracy_score(y_true_first, y_pred_single)
        print(f"\n单标签准确率 (Accuracy): {accuracy:.4f} (基于第一个标签)")
        
        # 7. 保存详细结果
        results_list = []
        for i, (filename, true_labels) in enumerate(zip(audio_filenames, audio_true_labels)):
            row = {
                'filename': filename,
                'true_labels': ', '.join(true_labels),
                'predicted_label': y_pred_single[i],
                'max_confidence': np.max(y_pred_scores[i])
            }
            # 添加每个类别的分数
            for j, cls in enumerate(target_classes):
                row[f'score_{cls}'] = y_pred_scores[i][j]
            results_list.append(row)
        
        results_df = pd.DataFrame(results_list)
        output_csv = 'evaluation_results_multilabel.csv'
        results_df.to_csv(output_csv, index=False)
        print(f"\n详细结果已保存到: {output_csv}")
        
        # 8. 显示每个音频的实际标签和预测
        print("\n" + "="*60)
        print("每个音频的详细预测结果:")
        print("="*60)
        for i in range(len(audio_filenames)):
            true_str = ', '.join(audio_true_labels[i])
            pred_str = y_pred_single[i]
            confidence = np.max(y_pred_scores[i])
            print(f"{i+1}. {audio_filenames[i]}")
            print(f"   真实标签: {true_str}")
            print(f"   预测标签: {pred_str} (置信度: {confidence:.4f})")
            print(f"   所有类别分数: ", end="")
            for j, cls in enumerate(target_classes):
                print(f"{cls}={y_pred_scores[i][j]:.4f} ", end="")
            print("\n")
        
        metrics = {
            'mAP': mAP,
            'accuracy': accuracy,
            'num_samples': len(audio_filenames),
            'num_classes': len(target_classes),
            'class_names': target_classes,
            'per_class_ap': average_precisions
        }
        
        return results_df, metrics


def main():
    """主函数"""
    
    # ============ 配置参数 ============
    AUDIO_FOLDER = r"C:\Users\xiao.xu\Desktop\DownloadFromCloud_OverThreshold\L1_1001_OverThreshold"
    LABEL_CSV = r"C:\Users\xiao.xu\Desktop\DownloadFromCloud_OverThreshold\label\L1_1001_OverThreshold.csv"
    MODEL_PATH = "models/LAION-CLAP/630k-audioset-fusion-best.pt"
    
    # 目标类别
    TARGET_CLASSES = ['bird', 'rain', 'frog', 'Construction Site']
    # =================================
    
    # 检查文件和路径
    if not os.path.exists(AUDIO_FOLDER):
        print(f"错误: 音频文件夹不存在: {AUDIO_FOLDER}")
        return
    
    if not os.path.exists(LABEL_CSV):
        print(f"错误: 标签文件不存在: {LABEL_CSV}")
        return
    
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型文件不存在: {MODEL_PATH}")
        print("请先下载模型文件:")
        print("  curl -L -o models/LAION-CLAP/630k-audioset-fusion-best.pt https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-fusion-best.pt")
        return
    
    # 创建评估器
    evaluator = ZeroShotEvaluator(MODEL_PATH)
    
    # 运行评估
    results, metrics = evaluator.evaluate_folder(
        audio_folder=AUDIO_FOLDER,
        label_csv=LABEL_CSV,
        target_classes=TARGET_CLASSES
    )
    
    if results is not None:
        print("\n" + "="*60)
        print("最终结果汇总")
        print("="*60)
        print(f"测试样本数: {metrics['num_samples']}")
        print(f"类别数: {metrics['num_classes']}")
        print(f"mAP: {metrics['mAP']:.4f}")
        print(f"单标签准确率: {metrics['accuracy']:.4f}")
        print("="*60)


if __name__ == "__main__":
    main()