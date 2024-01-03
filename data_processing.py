import pandas as pd
import datasets
from datasets import load_dataset
from sklearn.model_selection import train_test_split

TRAINING_CLASSIFIER_PROMPT = """### Sentence:{sentence} ### Class:{label}"""
INFERENCE_CLASSIFIER_PROMPT = """### Sentence:{sentence} ### Class:"""


def get_instruction_data(mode, texts, labels):
    if mode == "train":
        prompt = TRAINING_CLASSIFIER_PROMPT
    elif mode == "inference":
        prompt = INFERENCE_CLASSIFIER_PROMPT

    instructions = []

    for text, label in zip(texts, labels):
        if mode == "train":
            example = prompt.format(
                sentence=text,
                label=label,
            )
        elif mode == "inference":
            example = prompt.format(
                sentence=text,
            )
        instructions.append(example)

    return instructions


def get_newsgroup_data_for_ft(response_df, mode="train", train_sample_fraction=0.99):
    # sample n points from training data
    train_df, test_df = train_test_split(
        response_df,
        train_size=train_sample_fraction,
        stratify=response_df["label"],
        random_state=42,
    )
    train_data = train_df["text"]
    train_labels = train_df["label"]

    test_data = test_df["text"]
    test_labels = test_df["label"]

    train_instructions = get_instruction_data(mode, train_data, train_labels)
    test_instructions = get_instruction_data(mode, test_data, test_labels)

    train_dataset = datasets.Dataset.from_pandas(
        pd.DataFrame(
            data={
                "instructions": train_instructions,
                "labels": train_labels,
            }
        )
    )
    test_dataset = datasets.Dataset.from_pandas(
        pd.DataFrame(
            data={
                "instructions": test_instructions,
                "labels": test_labels,
            }
        )
    )

    return train_dataset, test_dataset


if __name__ == "__main__":
    response_df = pd.read_csv("mock_response.csv")
    train_dataset, test_dataset= get_newsgroup_data_for_ft(response_df, "train", 0.99)
    print(train_dataset)
    print(test_dataset)