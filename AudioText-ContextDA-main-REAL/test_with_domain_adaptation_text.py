"""
使用文本领域适应模式的零样本分类 + mAP评估脚本
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import average_precision_score, accuracy_score
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

class DomainAdaptationEvaluator:
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
    
    def get_text_embeddings_with_domain_adaptation(self, class_names, bg_type, temperature=0.5):
        """
        使用文本领域适应获取文本嵌入
        
        Args:
            class_names: 类别列表
            bg_type: 背景类型 (如 'park', 'street', 'construction_site')
            temperature: 温度参数，控制领域适应的强度
        """
        print(f"正在生成文本嵌入 (背景: {bg_type}, temperature: {temperature})...")
        
        # 1. 原始文本提示
        original_prompts = [f"The sound of {name}" for name in class_names]
        
        # 2. 领域适应文本提示
        # 关键：告诉模型音频是在特定背景中录制的
        adapted_prompts = [
            f"The sound of {name} in {bg_type} environment"
            for name in class_names
        ]
        
        # 3. 获取文本嵌入
        try:
            # 原始文本嵌入
            original_embeddings = self.model.get_text_embedding(original_prompts)
            
            # 适应后的文本嵌入
            adapted_embeddings = self.model.get_text_embedding(adapted_prompts)
            
            # 4. 使用温度参数进行融合
            # 公式: final = original + temperature * (adapted - original)
            # 这样可以在原始知识和领域适应之间做平衡
            final_embeddings = original_embeddings + temperature * (adapted_embeddings - original_embeddings)
            
            # 归一化
            final_embeddings = final_embeddings / np.linalg.norm(final_embeddings, axis=1, keepdims=True)
            
            print("  原始提示:", original_prompts)
            print("  适应提示:", adapted_prompts)
            
            return final_embeddings
            
        except Exception as e:
            print(f"获取文本嵌入失败: {e}")
            return None
    
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
        """解析Label Studio导出的CSV文件"""
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
            if len(parts) > 1:
                actual_filename = parts[1]
            else:
                actual_filename = filename
            
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
    
    def evaluate_with_domain_adaptation(self, audio_folder, label_csv, 
                                        target_classes, bg_type, temperature=0.5):
        """使用领域适应进行评估"""
        print(f"\n{'='*60}")
        print(f"开始评估 (领域适应模式: 文本)")
        print(f"背景类型: {bg_type}")
        print(f"温度参数: {temperature}")
        print(f"{'='*60}")
        
        # 1. 解析标签
        audio_labels_map = self.parse_label_studio_csv_multilabel(label_csv)
        
        if len(audio_labels_map) == 0:
            print("错误: 没有成功解析任何标签！")
            return None, None
        
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
            return None, None
        
        # 4. 获取领域适应的文本嵌入
        text_embeddings = self.get_text_embeddings_with_domain_adaptation(
            target_classes, bg_type, temperature
        )
        if text_embeddings is None:
            return None, None
        
        # 5. 提取音频嵌入并预测
        print("\n正在提取音频嵌入并预测...")
        y_true_multilabel = []
        y_pred_scores = []
        
        for audio_path, true_labels in tqdm(zip(audio_files, audio_true_labels), total=len(audio_files)):
            audio_embedding = self.get_audio_embedding(audio_path)
            
            if audio_embedding is None:
                continue
            
            similarities = np.dot(text_embeddings, audio_embedding)
            y_pred_scores.append(similarities)
            
            true_one_hot = [1 if cls in true_labels else 0 for cls in target_classes]
            y_true_multilabel.append(true_one_hot)
        
        if len(y_true_multilabel) == 0:
            print("错误: 没有成功提取任何音频的嵌入！")
            return None, None
        
        # 6. 计算评估指标
        y_true_multilabel = np.array(y_true_multilabel)
        y_pred_scores = np.array(y_pred_scores)
        
        print("\n" + "="*60)
        print("评估结果 (领域适应模式)")
        print("="*60)
        
        # 计算每个类别的AP
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
        
        # 单标签准确率
        y_pred_labels_idx = np.argmax(y_pred_scores, axis=1)
        y_pred_single = [target_classes[idx] for idx in y_pred_labels_idx]
        y_true_first = [labels[0] for labels in audio_true_labels]
        accuracy = accuracy_score(y_true_first, y_pred_single)
        print(f"\n单标签准确率: {accuracy:.4f}")
        
        # 7. 保存结果
        results_list = []
        for i, (filename, true_labels) in enumerate(zip(audio_filenames, audio_true_labels)):
            row = {
                'filename': filename,
                'true_labels': ', '.join(true_labels),
                'predicted_label': y_pred_single[i],
                'max_confidence': np.max(y_pred_scores[i])
            }
            for j, cls in enumerate(target_classes):
                row[f'score_{cls}'] = y_pred_scores[i][j]
            results_list.append(row)
        
        results_df = pd.DataFrame(results_list)
        output_csv = f'evaluation_results_domain_adapt_{bg_type}_t{temperature}.csv'
        results_df.to_csv(output_csv, index=False)
        print(f"\n详细结果已保存到: {output_csv}")
        
        # 8. 显示详细结果
        print("\n" + "="*60)
        print("每个音频的详细预测结果:")
        print("="*60)
        for i in range(min(10, len(audio_filenames))):  # 显示前10个
            true_str = ', '.join(audio_true_labels[i])
            pred_str = y_pred_single[i]
            confidence = np.max(y_pred_scores[i])
            print(f"{i+1}. {audio_filenames[i]}")
            print(f"   真实标签: {true_str}")
            print(f"   预测标签: {pred_str} (置信度: {confidence:.4f})")
        
        metrics = {
            'mAP': mAP,
            'accuracy': accuracy,
            'num_samples': len(audio_filenames),
            'num_classes': len(target_classes),
            'class_names': target_classes,
            'per_class_ap': average_precisions,
            'bg_type': bg_type,
            'temperature': temperature
        }
        
        return results_df, metrics


def main():
    """主函数"""
    
    # ============ 配置参数 ============
    AUDIO_FOLDER = r"C:\Users\xiao.xu\Desktop\DownloadFromCloud_OverThreshold\L1_1001_OverThreshold"
    LABEL_CSV = r"C:\Users\xiao.xu\Desktop\DownloadFromCloud_OverThreshold\label\L1_1001_OverThreshold.csv"
    MODEL_PATH = "models/LAION-CLAP/630k-audioset-fusion-best.pt"
    
    TARGET_CLASSES = ['bird', 'rain', 'frog', 'Construction Site']
    
    # 领域适应参数
    BG_TYPE = "construction_site"  # 背景类型
    TEMPERATURE = 0.7  # 温度参数 (0.0-1.0)
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
    evaluator = DomainAdaptationEvaluator(MODEL_PATH)
    
    # 运行评估
    results, metrics = evaluator.evaluate_with_domain_adaptation(
        audio_folder=AUDIO_FOLDER,
        label_csv=LABEL_CSV,
        target_classes=TARGET_CLASSES,
        bg_type=BG_TYPE,
        temperature=TEMPERATURE
    )
    
    if results is not None:
        print("\n" + "="*60)
        print("最终结果汇总")
        print("="*60)
        print(f"测试样本数: {metrics['num_samples']}")
        print(f"类别数: {metrics['num_classes']}")
        print(f"mAP: {metrics['mAP']:.4f}")
        print(f"单标签准确率: {metrics['accuracy']:.4f}")
        print(f"背景类型: {metrics['bg_type']}")
        print(f"温度参数: {metrics['temperature']}")
        print("="*60)


if __name__ == "__main__":
    main()