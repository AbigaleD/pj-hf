def get_topic_classification_pipeline() -> Callable[[str], dict]:
    """
    Question:
        Load the pipeline for topic text classification.
        There are 10 possible labels: 
            'Society & Culture', 'Science & Mathematics', 'Health',
            'Education & Reference', 'Computers & Internet', 'Sports', 'Business & Finance',
            'Entertainment & Music', 'Family & Relationships', 'Politics & Government'
        Find a proper model from HuggingFace Model Hub, then load the pipeline to classify the text.
        Notice that we have time limits so you should not use a model that is too large. A model with 
        100M params is enough.

    Returns:
        func (Callable): A function that takes a string as input and returns a dictionary with the
        predicted label and its score.

    Example:
        >>> func = get_topic_classification_pipeline()
        >>> result = func("Would the US constitution be changed if the admendment received 2/3 of the popular vote?")
        {"label": "Politics & Government", "score": 0.9999999403953552}
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    device_id = 0 if torch.cuda.is_available() else -1

    tokenizer = AutoTokenizer.from_pretrained("Koushim/distilbert-yahoo-answers-topic-classifier")
    model = AutoModelForSequenceClassification.from_pretrained("Koushim/distilbert-yahoo-answers-topic-classifier")
    if device_id >= 0:
        model = model.cuda()

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
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        if device_id >= 0:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        predicted_index = outputs.logits.argmax(dim=1).item()
        predicted_label = candidate_labels[predicted_index]
        score = torch.softmax(outputs.logits, dim=1)[0][predicted_index].item()
        return {"label": predicted_label, "score": score}

    return func
