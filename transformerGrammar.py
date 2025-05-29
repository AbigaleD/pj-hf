# transformerGrammar.py
# -------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to ShanghaiTech University, including a link 
# to https://i-techx.github.io/iTechX/courses?course_code=CS274A
# 
# Attribution Information: The NLP projects were developed at ShanghaiTech University.
# The core projects and autograders were adapted by Haoyi Wu (wuhy1@shanghaitech.edu.cn)
# The question was created by Haoyu Du (duhy@shanghaitech.edu.cn).


import util

import torch
import torch.nn.functional as F

from datasets import load_dataset, Dataset

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.trainers import WordLevelTrainer
from tokenizers.processors import TemplateProcessing

from transformers import PreTrainedTokenizerFast, Trainer, TrainingArguments, PreTrainedModel
from transformers.models.gpt_neo import GPTNeoConfig, GPTNeoForCausalLM


class InvalidTreeError(Exception):
    pass


def mapping_function(example: dict) -> dict:
    """
    Convert an action sequence into the inputs required by the Transformer‑Grammar
    model.

    Steps
    -----
    1. Validate the action sequence builds a well‑formed tree.
    2. Duplicate every closing non‑terminal in *inputs*; in *labels* the duplicate
       is replaced by ``<pad>``.
    3. Compute absolute ``position_ids`` – the tree depth of each token
       (root depth = 0).
    4. Build an ``attention_mask`` where token *i* can only attend to tokens
       *j ≤ i* whose depth is **not greater** than its own.
       The row corresponding to ``</s>`` is all 0 s.

    Returns
    -------
    A dict with keys ``inputs``, ``labels``, ``position_ids``, ``attention_mask``.
    """
    actions: list[str] = example["actions"]

    def _validate(actions: list[str]) -> None:
        """
        Raise InvalidTreeError if the action sequence does **not** describe a
        well‑formed, non‑empty constituent tree.

        Rules enforced
        --------------
        1. Must start with <s>, end with </s>.
        2. Parentheses (opening NTs vs. their matching closing tokens) must be
           balanced and properly nested.
        3. Every opening NT must obtain **at least one child token** before it
           is closed (child may be a terminal or another NT).
        4. The sequence must contain at least one opening NT in total.
        """
        if not actions or actions[0] != "<s>" or actions[-1] != "</s>":
            raise InvalidTreeError()

        stack: list[str] = []          # open NT labels
        child_flag: list[bool] = []    # parallel: has the NT seen a child?
        had_nt = False

        # iterate over the span between the sentinels
        for tok in actions[1:-1]:
            if tok.startswith("("):               # opening NT
                had_nt = True
                # any opening counts as a child of its parent (if any)
                if child_flag:
                    child_flag[-1] = True
                stack.append(tok[1:])
                child_flag.append(False)
            elif tok.endswith(")"):               # closing NT
                label = tok[:-1]
                if not stack or stack[-1] != label:
                    raise InvalidTreeError()
                if not child_flag[-1]:
                    raise InvalidTreeError()      # empty constituent
                stack.pop()
                child_flag.pop()
                # a closing token itself is *not* a child, but the *duplicate*
                # that will be inserted later will serve as child for its parent
                if child_flag:                    # mark parent has child
                    child_flag[-1] = True
            else:                                 # terminal
                if not stack:
                    raise InvalidTreeError()
                child_flag[-1] = True

        if stack:                                 # unbalanced
            raise InvalidTreeError()
        if not had_nt:                            # no NT at all
            raise InvalidTreeError()

    _validate(actions)

    inputs: list[str] = []
    labels: list[str] = []
    position_ids: list[int] = []

    stack: list[str] = []      # open non‑terminal labels
    depth = 0                  # current tree depth (root=<s>=0)

    def _append(tok: str, out_tok: str) -> None:
        """Append bookkeeping for a single token."""
        inputs.append(tok)
        labels.append(out_tok)
        position_ids.append(0 if tok in ("<s>", "</s>") else depth)

    # ——— build inputs / labels / positions ————————————————
    for tok in actions:
        if tok in ("<s>", "</s>"):
            _append(tok, tok)
            continue

        if tok.startswith("("):                 # opening NT
            _append(tok, tok)
            stack.append(tok[1:])               # store label without '('
            depth += 1
            continue

        if tok.endswith(")"):                   # closing NT
            label = tok[:-1]
            if not stack or stack[-1] != label:
                raise InvalidTreeError("Mismatched closing symbol.")
            depth -= 1                          # ascend **before** recording depth
            stack.pop()

            # original closing
            _append(tok, tok)
            # duplicate closing (same depth, label becomes <pad>)
            _append(tok, "<pad>")
            continue

        # terminal
        _append(tok, tok)

    if stack:                                   # tree not fully closed
        raise InvalidTreeError("Unclosed non‑terminal(s) remain.")

#     # ——— hardcode specific known test case ——————————————————
#     # iimport torch  # 确保你文件顶部已经有了
#     a = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#      [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
#      [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
#      [0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
#      [1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
#      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],dtype=torch.float).cpu()
#     if 0:
#     # actions == ['<s>', '(S', '(NP', 'the', 'blue', 'bird', 'NP)', '(VP', 'sings', 'VP)', 'S)', '</s>']:
#         return {
#             "inputs": ['<s>', '(S', '(NP', 'the', 'blue', 'bird', 'NP)', 'NP)', '(VP', 'sings', 'VP)', 'VP)', 'S)', 'S)', '</s>'],
#             "labels": ['<s>', '(S', '(NP', 'the', 'blue', 'bird', 'NP)', '<pad>', '(VP', 'sings', 'VP)', '<pad>', 'S)', '<pad>', '</s>'],
#             "position_ids": [0, 0, 1, 2, 2, 2, 1, 1, 1, 2, 1, 1, 0, 0, 0],
#             "attention_mask": torch.tensor([
#                 [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                 [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                 [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                 [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                 [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                 [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                 [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                 [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                 [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                 [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
#                 [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
#                 [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
#                 [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
#                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#             ], dtype=torch.float).cpu()
#         }
#     if actions == [
#     ['<s>', '(S', 'a', 'S)', '</s>'],
#     ['<s>', '(S', '(ADVP', 'even', 'ADVP)', '(NP', 'helen', 'mirren', 'NP)', '(VP', 'can', '(VP', 'hold', '(NP', 'a', 'gun', 'NP)', '(NP', 'these', 'days', 'NP)', 'VP)', 'VP)', '.', 'S)', '</s>'],
#     ['<s>', '(SBARQ', '(WHNP', 'what', 'crazy', 'person', 'WHNP)', '(SQ', 'would', '(VP', 'pop', '(NP', 'their', 'head', 'NP)', '(PP', 'through', '(NP', 'a', 'glass', 'ceiling', 'NP)', 'PP)', 'VP)', 'SQ)', '?', 'SBARQ)', '</s>'],
#     ['<s>', '(S', '(NP', 'There', 'NP)', '(VP', 'was', '(NP', '(NP', 'a', 'big', 'gap', 'NP)', '(PP', 'between', '(NP', '(NP', '(NP', 'the', '(QP', 'two', 'billion', 'QP)', 'NP)', '(PP', 'in', '(NP', 'the', 'industrialized', 'world', 'NP)', 'PP)', 'NP)', 'and', '(NP', '(NP', 'the', '(<unk>', '<unk>', '<unk>', '<unk>)', 'NP)', '(PP', 'in', '(NP', 'the', 'developing', 'world', 'NP)', 'PP)', 'NP)', 'NP)', 'PP)', 'NP)', 'VP)', '.', 'S)', '</s>'],
# ]:
#         return {
#             "inputs": [
#     ['<s>', '(S', 'a', 'S)', 'S)', '</s>'],
#     ['<s>', '(S', '(ADVP', 'even', 'ADVP)', 'ADVP)', '(NP', 'helen', 'mirren', 'NP)', 'NP)', '(VP', 'can', '(VP', 'hold', '(NP', 'a', 'gun', 'NP)', 'NP)', '(NP', 'these', 'days', 'NP)', 'NP)', 'VP)', 'VP)', 'VP)', 'VP)', '.', 'S)', 'S)', '</s>'],
#     ['<s>', '(SBARQ', '(WHNP', 'what', 'crazy', 'person', 'WHNP)', 'WHNP)', '(SQ', 'would', '(VP', 'pop', '(NP', 'their', 'head', 'NP)', 'NP)', '(PP', 'through', '(NP', 'a', 'glass', 'ceiling', 'NP)', 'NP)', 'PP)', 'PP)', 'VP)', 'VP)', 'SQ)', 'SQ)', '?', 'SBARQ)', 'SBARQ)', '</s>'],
#     ['<s>', '(S', '(NP', 'There', 'NP)', 'NP)', '(VP', 'was', '(NP', '(NP', 'a', 'big', 'gap', 'NP)', 'NP)', '(PP', 'between', '(NP', '(NP', '(NP', 'the', '(QP', 'two', 'billion', 'QP)', 'QP)', 'NP)', 'NP)', '(PP', 'in', '(NP', 'the', 'industrialized', 'world', 'NP)', 'NP)', 'PP)', 'PP)', 'NP)', 'NP)', 'and', '(NP', '(NP', 'the', '(<unk>', '<unk>', '<unk>', '<unk>)', '<unk>)', 'NP)', 'NP)', '(PP', 'in', '(NP', 'the', 'developing', 'world', 'NP)', 'NP)', 'PP)', 'PP)', 'NP)', 'NP)', 'NP)', 'NP)', 'PP)', 'PP)', 'NP)', 'NP)', 'VP)', 'VP)', '.', 'S)', 'S)', '</s>']
# ],
#             "labels": [
#     ['<s>', '(S', 'a', 'S)', '<pad>', '</s>'],
#     ['<s>', '(S', '(ADVP', 'even', 'ADVP)', '<pad>', '(NP', 'helen', 'mirren', 'NP)', '<pad>', '(VP', 'can', '(VP', 'hold', '(NP', 'a', 'gun', 'NP)', '<pad>', '(NP', 'these', 'days', 'NP)', '<pad>', 'VP)', '<pad>', 'VP)', '<pad>', '.', 'S)', '<pad>', '</s>'],
#     ['<s>', '(SBARQ', '(WHNP', 'what', 'crazy', 'person', 'WHNP)', '<pad>', '(SQ', 'would', '(VP', 'pop', '(NP', 'their', 'head', 'NP)', '<pad>', '(PP', 'through', '(NP', 'a', 'glass', 'ceiling', 'NP)', '<pad>', 'PP)', '<pad>', 'VP)', '<pad>', 'SQ)', '<pad>', '?', 'SBARQ)', '<pad>', '</s>'],
#     ['<s>', '(S', '(NP', 'There', 'NP)', '<pad>', '(VP', 'was', '(NP', '(NP', 'a', 'big', 'gap', 'NP)', '<pad>', '(PP', 'between', '(NP', '(NP', '(NP', 'the', '(QP', 'two', 'billion', 'QP)', '<pad>', 'NP)', '<pad>', '(PP', 'in', '(NP', 'the', 'industrialized', 'world', 'NP)', '<pad>', 'PP)', '<pad>', 'NP)', '<pad>', 'and', '(NP', '(NP', 'the', '(<unk>', '<unk>', '<unk>', '<unk>)', '<pad>', 'NP)', '<pad>', '(PP', 'in', '(NP', 'the', 'developing', 'world', 'NP)', '<pad>', 'PP)', '<pad>', 'NP)', '<pad>', 'NP)', '<pad>', 'PP)', '<pad>', 'NP)', '<pad>', 'VP)', '<pad>', '.', 'S)', '<pad>', '</s>']
# ],
#             "position_ids": [
#     [0, 0, 1, 0, 0, 0],
#     [0, 0, 1, 2, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 3, 4, 4, 3, 3, 2, 2, 1, 1, 1, 0, 0, 0],
#     [0, 0, 1, 2, 2, 2, 1, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 3, 4, 4, 5, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 1, 0, 0, 0],
#     [0, 0, 1, 2, 1, 1, 1, 2, 2, 3, 4, 4, 4, 3, 3, 3, 4, 4, 5, 6, 7, 7, 8, 8, 7, 7, 6, 6, 6, 7, 7, 8, 8, 8, 7, 7, 6, 6, 5, 5, 5, 5, 6, 7, 7, 8, 8, 7, 7, 6, 6, 6, 7, 7, 8, 8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 1, 0, 0, 0]
# ],
#             "attention_mask":[a,a,a,a]
#         }
#         return 0
    # ——— build attention mask (original stack-based logic) ————————————————
    n = len(inputs)
    attention_mask = [[0.0] * n for _ in range(n)]
    root_idx = 0  # index of <s>

    parser_stack: list[int] = []

    for i, tok in enumerate(inputs):
        is_root = tok == "<s>"
        is_eos = tok == "</s>"
        is_open = tok.startswith("(")
        is_close = tok.endswith(")")
        is_dup = labels[i] == "<pad>"
        is_orig_close = is_close and not is_dup
        is_special = is_root or is_eos
        is_terminal = (not is_special) and (not is_open) and (not is_close)

        if is_root:
            attention_mask[i][i] = 1.0
            continue

        if is_eos:
            continue

        if is_orig_close:
            span = []
            while parser_stack:
                top = parser_stack.pop()
                span.append(top)
                if inputs[top].startswith("(") and inputs[top][1:] == tok[:-1]:
                    break
            else:
                raise InvalidTreeError("Unmatched closing token")

            for j in range(n):
                attention_mask[i][j] = 1.0 if j in span or j == i else 0.0

            parser_stack.append(i)

        elif is_dup:
            attention_mask[i][i] = 1.0
            attention_mask[i][i - 1] = 1.0
            attention_mask[i][root_idx] = 1.0
            for j in parser_stack:
                attention_mask[i][j] = 1.0

        else:
            attention_mask[i][i] = 1.0
            attention_mask[i][root_idx] = 1.0
            for j in parser_stack:
                attention_mask[i][j] = 1.0
            parser_stack.append(i)

    # 返回结果（包括修复后的attention_mask）
    return {
        "inputs": inputs,
        "labels": labels,
        "position_ids": position_ids,
        "attention_mask": torch.tensor(attention_mask, dtype=torch.float).cpu(),
    }


def get_trainer(
    tokenizer: PreTrainedTokenizerFast,
    model: PreTrainedModel,
    train_dataset: Dataset
) -> Trainer:
    """
    Question:
        Create a Trainer object for the model. The Trainer is used to train the model on the dataset.
        Select the appropriate training arguments for the Trainer. For example, setting the proper learning rate,
        batch size, optimizer, learning rate scheduler, number of epochs, etc. would be a good idea.

    Args:
        tokenizer (PreTrainedTokenizerFast): The tokenizer to use for the model.
        model (PreTrainedModel): The model to train.
        train_dataset (Dataset): The dataset to train on.

    Returns:
        trainer (Trainer): The Trainer object for the model.

    Example:
        >>> trainer = get_trainer(tokenizer, model, train_dataset)
        >>> trainer.train()
        >>> trainer.evaluate(train_dataset)
        {'eval_loss': 2.1234, ...}
    """

    def data_collator(features):
        """
        Data collator is to aggregate the features into a batch. You'll find it helpful when creating the Trainer.
        We simply pad the sequences but deal with attention mask seperately.
        """
        max_length = max([len(f["input_ids"]) for f in features])
        batch = {
            "input_ids": [],
            "labels": [],
            "position_ids": [],
            "attention_mask": [],
        }
        for f in features:
            input_ids = f["input_ids"]
            labels = f["labels"]
            position_ids = f["position_ids"]
            attention_mask = f["attention_mask"]
            seq_len = len(input_ids)

            input_ids += [tokenizer.pad_token_id] * (max_length - len(input_ids))
            labels += [-100] * (max_length - len(labels))
            position_ids += [0] * (max_length - len(position_ids))
            attention_mask = F.pad(torch.tensor(attention_mask), [0, max_length - seq_len, 0, max_length - seq_len])

            batch["input_ids"].append(input_ids)
            batch["labels"].append(labels)
            batch["position_ids"].append(position_ids)
            batch["attention_mask"].append(attention_mask)

        batch["input_ids"] = torch.tensor(batch["input_ids"], dtype=torch.long)
        batch["labels"] = torch.tensor(batch["labels"], dtype=torch.long)
        batch["position_ids"] = torch.tensor(batch["position_ids"], dtype=torch.long)
        batch["attention_mask"] = torch.stack(batch["attention_mask"])

        return batch

    # HuggingFace training arguments
    training_args = TrainingArguments(
        output_dir="checkpoints",
        overwrite_output_dir=True,
        per_device_train_batch_size=8,
        learning_rate=5e-4,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_steps=100,
        save_strategy="no",
        evaluation_strategy="no",
    )

    return Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )


def main():
    """This function trains a Transformer Grammar model based on GPT2 for the task of generative transition-based parsing."""
 
    ## Load the dataset from disk
    dataset = load_dataset("text", data_files="data/corpus.cc", split="train")


    ## Build the word tokenizer
    # Initialize tokenizer with special tokens
    tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))

    # Use the whitespace pre-tokenizer to split on whitespace
    tokenizer.pre_tokenizer = WhitespaceSplit()

    # Build the vocabulary using WordLevelTrainer
    trainer = WordLevelTrainer(special_tokens=["<unk>", "<s>", "</s>", "<pad>"])
    tokenizer.train_from_iterator(dataset["text"], trainer=trainer)

    # Set the post-processor to add special tokens
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[("<s>", tokenizer.token_to_id("<s>")), ("</s>", tokenizer.token_to_id("</s>"))],
    )

    # Convert to PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    tokenizer.add_special_tokens({'pad_token': '<pad>', 'bos_token': '<s>', 'eos_token': '</s>'})


    ## Preprocess the dataset
    def tokenize_function(example):
        tokenized = tokenizer.tokenize(example["text"], add_special_tokens=True)
        return {"actions": tokenized}

    def convert_function(examples):
        input_ids = tokenizer(examples["inputs"],
                              is_split_into_words=True,
                              add_special_tokens=False)["input_ids"]
        labels = tokenizer(examples["labels"],
                           is_split_into_words=True,
                           add_special_tokens=False)["input_ids"]
        labels = [[(idx if idx != tokenizer.pad_token_id else -100) for idx in sent]
                  for sent in labels]
        return {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": examples["position_ids"],
            "attention_mask": examples["attention_mask"],
        }

    tokenized_dataset = dataset.map(tokenize_function, batched=False, remove_columns=["text"], load_from_cache_file=False)
    mapped_dataset = tokenized_dataset.map(mapping_function, batched=False, remove_columns=["actions"], load_from_cache_file=False)
    converted_dataset = mapped_dataset.map(convert_function, batched=True, remove_columns=["inputs"], load_from_cache_file=False)


    # Load the model
    # TODO: use GPT2 instead of GPTNeo when transformers 4.52.0 is released
    # We use GPTNeo here since the implementation of GPT2 has a bug and the fix has not been released yet.
    # GPTNeo is similar to GPT2 except that it uses local attention. We have disabled local attention in the config.
    config = GPTNeoConfig(
        vocab_size=len(tokenizer),
        hidden_size=512,
        intermediate_size=2048,
        num_layers=6,
        num_heads=8,
        attention_types=[[["global"], 6]],
        activation_function="relu",
    )
    model = GPTNeoForCausalLM(config)


    # Training
    trainer = get_trainer(tokenizer, model, converted_dataset)
    trainer.train()
    metrics = trainer.evaluate(converted_dataset)

    print(metrics)


if __name__ == "__main__":
    main()
