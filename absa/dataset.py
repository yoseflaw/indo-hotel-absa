import re
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class ReviewDataset(Dataset):
    def __init__(self, filepath, tokenizer, tag_only, tag2idx=None):
        self.texts, self.tags = ReviewDataset.read_input(filepath, tag_only)
        self.encodings = tokenizer(
            self.texts,
            is_pretokenized=True,
            return_offsets_mapping=True,
            padding=True,
            truncation=True
        )
        if tag2idx is None:
            unique_tags = set(tag for doc in self.tags for tag in doc)
            self.num_labels = len(unique_tags)
            self.tag2idx = {tag: idx for idx, tag in enumerate(unique_tags)}
        else:
            self.tag2idx = tag2idx
        self.idx2tag = {idx: tag for tag, idx in self.tag2idx.items()}
        self.labels = ReviewDataset.encode_tags(self.tags, self.encodings, self.tag2idx)
        self.encodings.pop("offset_mapping")

    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

    @staticmethod
    def read_input(filepath, tag_only):
        path = Path(filepath)
        raw_text = path.read_text().strip()
        raw_docs = re.split(r"\n\n", raw_text)
        token_docs = []
        tag_docs = []
        for doc in raw_docs:
            tokens = []
            tags = []
            for line in doc.split("\n"):
                token, tag = line.strip().split("\t")
                tokens.append(token)
                if tag_only:
                    tag = tag.split("-")[1] if tag != "O" else tag
                tags.append(tag)
            token_docs.append(tokens)
            tag_docs.append(tags)
        return token_docs, tag_docs

    @staticmethod
    def encode_tags(tags, encodings, tag_to_idx):
        labels = [[tag_to_idx[tag] for tag in doc] for doc in tags]
        encoded_labels = []
        for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
            doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100  # empty array with -100
            # replace labels of the first subwords with the actual labels
            arr_offset = np.array(doc_offset)
            doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = doc_labels
            encoded_labels.append(doc_enc_labels.tolist())
        return encoded_labels
