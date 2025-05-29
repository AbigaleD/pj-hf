# classification.py
# -------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to ShanghaiTech University, including a link 
# to https://i-techx.github.io/iTechX/courses?course_code=CS274A
# 
# Attribution Information: The NLP projects were developed at ShanghaiTech University.
# The core projects and autograders were adapted by Haoyi Wu (wuhy1@shanghaitech.edu.cn)


from typing import Callable
import argparse
from transformers import pipeline
import gradio as gr


# def get_topic_classification_pipeline() -> Callable[[str], dict]:
#     """
#     Question:
#         Load the pipeline for topic text classification.
#         There are 10 possible labels: 
#             'Society & Culture', 'Science & Mathematics', 'Health',
#             'Education & Reference', 'Computers & Internet', 'Sports', 'Business & Finance',
#             'Entertainment & Music', 'Family & Relationships', 'Politics & Government'
#         Find a proper model from HuggingFace Model Hub, then load the pipeline to classify the text.
#         Notice that we have time limits so you should not use a model that is too large. A model with 
#         100M params is enough.

#     Returns:
#         func (Callable): A function that takes a string as input and returns a dictionary with the
#         predicted label and its score.

#     Example:
#         >>> func = get_topic_classification_pipeline()
#         >>> result = func("Would the US constitution be changed if the admendment received 2/3 of the popular vote?")
#         {"label": "Politics & Government", "score": 0.9999999403953552}
#     """
#     pipe = pipeline(model="cointegrated/rubert-tiny-sentiment-balanced")
#     def func(text: str) -> dict:
#         return pipe(text)[0]
#     return func

# def get_topic_classification_pipeline():
#     # 映射：描述性标签 → 题目要求的标签
#     label_map = {
#         "Questions about society, traditions, and culture": "Society & Culture",
#         "Scientific or mathematical questions": "Science & Mathematics",
#         "Health-related topics and medical inquiries": "Health",
#         "Topics related to education and academic reference": "Education & Reference",
#         "Computers, programming, and internet-related topics": "Computers & Internet",
#         "Sports, games, and athletic events": "Sports",
#         "Business, finance, and economic discussions": "Business & Finance",
#         "Entertainment, music, movies, and pop culture": "Entertainment & Music",
#         "Family issues, personal relationships, and social dynamics": "Family & Relationships",
#         "Political systems, government, and laws": "Politics & Government"
#     }

#     candidate_labels = list(label_map.keys())

#     pipe = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-3")

#     def func(text: str) -> dict:
#         result = pipe(text, candidate_labels)
#         predicted_description = result["labels"][0]
#         mapped_label = label_map[predicted_description]
#         return {"label": mapped_label, "score": result["scores"][0]}

#     return func

def get_topic_classification_pipeline():
    model_path = "/home/abigaledong/.0/pj-hf/q2SelfMadeModel/my_topic_model"
    pipe = pipeline("text-classification", model=model_path, tokenizer=model_path)

    def func(text: str) -> dict:
        result = pipe(text)[0]
        return {"label": result["label"], "score": result["score"]}
    
    return func


def main():
    parser = argparse.ArgumentParser(description="Topic Classification Pipeline")
    parser.add_argument("--task", type=str, help="Task name", choices=["sentiment", "topic"], default="sentiment")
    parser.add_argument("--use-gradio", action="store_true", help="Use Gradio for UI")

    args = parser.parse_args()

    if args.use_gradio and args.task == "sentiment":
        pipe = pipeline(model="cointegrated/rubert-tiny-sentiment-balanced")
        iface = gr.Interface.from_pipeline(pipe)
        iface.launch()

    elif args.use_gradio and args.task == "topic":
        pipe = get_topic_classification_pipeline()
        def classify_text(text: str) -> dict:
            result = pipe(text)[0]
            return {result["label"]: result["score"]}

        iface = gr.Interface(
            fn=classify_text,
            inputs=gr.components.Textbox(label="Input", render=False),
            outputs=gr.components.Label(label="Classification", render=False),
            title="Text Classification",
        )
        iface.launch()

    elif not args.use_gradio and args.task == "sentiment":
        pipe = pipeline(model="cointegrated/rubert-tiny-sentiment-balanced")
        print(pipe("This movie is very wonderful!")[0])

    elif not args.use_gradio and args.task == "topic":
        func = get_topic_classification_pipeline()
        print(func("Would the US constitution be changed if the admendment received 2/3 of the popular vote?")) # {"label": "Politics & Government", "score": ...}


if __name__ == "__main__":
    main()
