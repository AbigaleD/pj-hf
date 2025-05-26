from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd

# 加载数据
df = pd.read_csv("topic_train.csv")

# 创建标签映射
label_list = sorted(df["label"].unique())
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}
df["label"] = df["label"].map(label2id)

# 转换为 HuggingFace Dataset 格式
dataset = Dataset.from_pandas(df)

# 加载 tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize 文本
def tokenize_fn(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

dataset = dataset.map(tokenize_fn)

# 加载模型
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

# 训练参数
training_args = TrainingArguments(
    output_dir="./my_topic_model",
    per_device_train_batch_size=8,
    num_train_epochs=10,
    logging_steps=10,
    save_strategy="no",
    # evaluation_strategy="no"
)

# 训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# 开始训练
trainer.train()

# 保存模型和 tokenizer
model.save_pretrained("./my_topic_model")
tokenizer.save_pretrained("./my_topic_model")

print("✅ 模型已保存到 ./my_topic_model")
