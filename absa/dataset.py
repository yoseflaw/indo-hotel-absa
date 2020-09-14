import re
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class ReviewDataset(Dataset):
    def __init__(self, filepath, tokenizer, tag_only, tag2idx=None, aux_labels=False):
        self.texts, self.tags = ReviewDataset.read_input(filepath, tag_only)
        self.encodings = tokenizer(
            self.texts,
            is_pretokenized=True,
            return_offsets_mapping=True,
            padding=True,
            truncation=True
        )
        self.aux_labels = aux_labels
        self.reserved_tags = None
        if tag2idx is None:
            unique_tags = set(tag for doc in self.tags for tag in doc)
            if aux_labels:
                self.reserved_tags = {
                    "<pad>": 0,
                    "A": 1,
                    "Z": 2,
                    "Y": 3
                }
                if not tag_only:
                    unique_names = set(tag.split("-")[1] for doc in self.tags for tag in doc if "-" in tag)
                    self.reserved_tags.update(
                        {f"X-{name}": idx + len(self.reserved_tags) for idx, name in enumerate(unique_names)}
                    )
                self.tag2idx = self.reserved_tags.copy()
                self.tag2idx.update({tag: idx + len(self.reserved_tags) for idx, tag in enumerate(unique_tags)})
            else:
                self.tag2idx = {tag: idx for idx, tag in enumerate(unique_tags)}
        else:
            self.tag2idx = tag2idx
        self.num_labels = len(self.tag2idx)
        self.idx2tag = {idx: tag for tag, idx in self.tag2idx.items()}
        self.labels = self.encode_tags()
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

    def encode_tags(self):
        labels = [[self.tag2idx[tag] for tag in doc] for doc in self.tags]
        encoded_labels = []
        for doc_labels, doc_offset in zip(labels, self.encodings.offset_mapping):
            doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100  # empty array with -100
            # replace labels of the first subwords with the actual labels
            arr_offset = np.array(doc_offset)
            doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = doc_labels
            if self.aux_labels:
                doc_enc_labels[0] = self.tag2idx["A"]
                for i in range(1, len(doc_enc_labels)):
                    prev_label = doc_enc_labels[i - 1]
                    curr_label = doc_enc_labels[i]
                    if curr_label == -100:
                        prev_tag = self.idx2tag[prev_label]
                        if prev_tag[0] in ("B", "I", "X"):
                            tag_name = prev_tag.split("-")[1]
                            doc_enc_labels[i] = self.tag2idx[f"X-{tag_name}"]
                        elif sum(arr_offset[i]) == 0 and prev_tag not in ("Z", "<pad>"):
                            doc_enc_labels[i] = self.tag2idx["Z"]
                        elif prev_tag in ["O", "Y"]:
                            doc_enc_labels[i] = self.tag2idx["Y"]
                        else:
                            doc_enc_labels[i] = self.tag2idx["<pad>"]
            encoded_labels.append(doc_enc_labels.tolist())
        return encoded_labels
