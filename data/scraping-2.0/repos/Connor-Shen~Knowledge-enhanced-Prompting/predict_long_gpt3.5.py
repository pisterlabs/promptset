import openai
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import json
import time

# 设置您的API密钥
openai.api_key = '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'

def predict_labels(prompts, prefix_template):
    labels = []
    failed_indices = []  # 记录失败请求的索引
    for i, prompt in enumerate(prompts):
        try:
            response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-16k",
    messages=[
        {"role": "system", "content": prefix_template},
        {"role": "user", "content": prompt}
    ]
)
            label = response['choices'][0]['message']['content']
            print(label)
            labels.append(label)
            time.sleep(5)  # 为了避免请求速度过快，我们在每个请求之间添加了一些延迟
        except Exception as e:
            print(f"Failed to process prompt at index {i}: {e}")
            labels.append(None)  # 添加 None 作为失败的占位符
            failed_indices.append(i)
        if (i + 1) % 100 == 0:
            checkpoint_file = f'/Users/connor/Desktop/Desktop/Experiment/checkpoints/labels_checkpoint_{i+1}.pkl'
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(labels, f)

    return labels, failed_indices

def evaluate_model(y_true, y_pred):
    # 创建一个新的空列表来存储不包含 "unknown" 标签的 y_true 和 y_pred
    y_true_filtered = []
    y_pred_filtered = []

    # 过滤掉 "unknown" 标签的预测
    for true_label, pred_label in zip(y_true, y_pred):
        if pred_label != 'Unknown':  # 如果预测标签不是 "unknown"，则保留它
            y_true_filtered.append(true_label)
            y_pred_filtered.append(pred_label)

    # 计算性能指标
    accuracy = accuracy_score(y_true_filtered, y_pred_filtered)
    precision = precision_score(y_true_filtered, y_pred_filtered, average='weighted', zero_division=0)
    recall = recall_score(y_true_filtered, y_pred_filtered, average='weighted', zero_division=0)
    f1 = f1_score(y_true_filtered, y_pred_filtered, average='weighted', zero_division=0)

    # 将标签编码为 0 和 1
    le = LabelBinarizer()
    y_true_bin = le.fit_transform(y_true_filtered)
    y_pred_bin = le.transform(y_pred_filtered)
    
    auc_roc = roc_auc_score(y_true_bin, y_pred_bin, average='weighted', multi_class='ovr')

    
    # 绘制ROC曲线
    fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_pred_bin.ravel())
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc_roc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    # 使用 savefig 保存图像到文件
    plt.savefig('/Users/connor/Desktop/Desktop/Experiment/outputs/roc_curve.png', dpi=300)  # dpi 参数是可选的，用于控制输出图像的分辨率

    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'AUC-ROC': auc_roc
    }

def clean_labels(labels):
    cleaned_labels = []  # 创建一个新列表来存储清理后的标签
    for label in labels:
        if len(label) > 15:  # 检查每个标签的长度
            cleaned_labels.append('Unknown')  # 将长标签设置为 'Unknown'
        elif ("press" in label) and len(label)<15:
            cleaned_labels.append("depression")
        elif ("ontrol" in label) and len(label)<15:
            cleaned_labels.append("control")
    return cleaned_labels  # 返回清理后的标签列表


# 数据集
dataset = pd.read_excel("/Users/connor/Desktop/Desktop/Experiment/sample_balance_data.xlsx")
dataset = dataset.iloc[0:305]

prompts = dataset["text"]
true_labels = dataset["label"]

prefix_template = """
Here are the tweets of a Twitter user and his tweeting time.
Think step by step, and your output will be used directly as predicted labels, so try to accurately classify the user as "depression" or "control" and no more words.
Even if you think it can't be classified, give directly the label between "depression" or "control" which you think is closest: 
"""
# 获取预测标签
predicted_labels, failed_indices = predict_labels(prompts, prefix_template)

# 如果有失败的请求，您可以在此处处理
if failed_indices:
    print(f"Failed to process {len(failed_indices)} prompts: {failed_indices}")


# 清理 y_pred
y_pred_cleaned = clean_labels(predicted_labels)
print("*"*10)
print(y_pred_cleaned)

# 调用您的 evaluate_model 函数
performance_metrics = evaluate_model(true_labels, y_pred_cleaned)
print(performance_metrics)

with open("/Users/connor/Desktop/Desktop/Experiment/outputs/evaluation.json", 'w') as f:
    json.dump(performance_metrics, f, indent=4)