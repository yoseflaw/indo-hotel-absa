import torch
from transformers import BertTokenizerFast

from absa.dataset import ReviewDataset
from absa.model import AspectModel

if __name__ == "__main__":
    DRIVE_ROOT = "."
    available_gpu = torch.cuda.is_available()
    if available_gpu:
        print(f"GPU is available: {torch.cuda.get_device_name(0)}")
        use_device = torch.device("cuda")
    else:
        use_device = torch.device("cpu")
    model_name = "bert-base-multilingual-uncased"
    tokenizer = BertTokenizerFast.from_pretrained(model_name, cache_dir=f"{DRIVE_ROOT}/pt_model/")
    train_dataset = ReviewDataset(f"{DRIVE_ROOT}/input/train.tsv", tokenizer)
    val_dataset = ReviewDataset(f"{DRIVE_ROOT}/input/val.tsv", tokenizer, tag2idx=train_dataset.tag2idx)
    test_dataset = ReviewDataset(f"{DRIVE_ROOT}/input/test.tsv", tokenizer, tag2idx=train_dataset.tag2idx)
    aspect_model = AspectModel(
        model_reference=model_name,
        tokenizer=tokenizer,
        device=use_device,
        num_labels=train_dataset.num_labels,
        cache_dir=f"{DRIVE_ROOT}/pt_model/"
    )
    _ = aspect_model.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        logging_dir=f"{DRIVE_ROOT}/logs",
        num_train_epochs=1,
        logging_steps=24,
        lr=1e-4,
        weight_decay=1e-2,
        warmup_steps=47,
        output_dir=f"{DRIVE_ROOT}/results",
        save_model=True
    )
    text = "tempatnya sempurna, pas didepan pantai losari, namun sayang acnya kurang dingin padahal sdh lapor namun tetep tidak bisa dingin."
    tokens, pred_tags = aspect_model.infer(text)
    max_len = max([len(token) for token in tokens])
    for token, pred_tag in zip(tokens, pred_tags):
        print(f"{token.ljust(max_len)}\t{train_dataset.idx2tag[pred_tag]}")
