from datasets import load_dataset
import pandas as pd

# 1. 加载 Yahoo Answers Topics 数据集的前1000条训练数据
dataset = load_dataset("yahoo_answers_topics", split="train[:10000]")

# 2. 标签映射（官方10类）
label_names = [
    "Society & Culture", "Science & Mathematics", "Health",
    "Education & Reference", "Computers & Internet", "Sports",
    "Business & Finance", "Entertainment & Music",
    "Family & Relationships", "Politics & Government"
]

# 3. 构建 DataFrame（使用 'question_title' 作为文本）
df = pd.DataFrame({
    "text": dataset["question_title"],
    "label": [label_names[i] for i in dataset["topic"]]
})

# 4. 去除空文本
df = df[df["text"].str.strip().astype(bool)].reset_index(drop=True)

# 5. 保存为 CSV
df.to_csv("topic_train.csv", index=False)
print("✅ 保存完成：topic_train.csv")