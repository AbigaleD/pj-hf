def get_topic_classification_pipeline():
    """
    Topic classification using cardiffnlp/tweet-topic-21-multi, mapped to 10 topics.
    """
    from transformers import pipeline
    import torch

    # Use GPU if available, otherwise fall back to CPU
    device_id = 0 if torch.cuda.is_available() else -1

    pipe = pipeline(
        "zero-shot-classification",
        model="joeddav/bart-large-mnli-yahoo-answers",
        device=device_id,
        hypothesis_template="This text is about {}.",
    )

    candidate_labels = [
        "Society & Culture",
        "Science & Mathematics",
        "Health",
        "Education & Reference",
        "Computers & Internet",
        "Sports",
        "Business & Finance",
        "Entertainment & Music",
        "Family & Relationships",
        "Politics & Government",
    ]

    def func(text: str) -> dict:
        result = pipe(text, candidate_labels)
        return {"label": result["labels"][0], "score": result["scores"][0]}

    return func
