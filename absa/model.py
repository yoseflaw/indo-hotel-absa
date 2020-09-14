import csv
import os

from nltk import word_tokenize
from transformers import BertForTokenClassification, Trainer, TrainingArguments
from sklearn.metrics import f1_score, classification_report


class AspectModel(object):

    def __init__(self, model_reference, tokenizer, device, num_labels=None, cache_dir=None):
        model_args = {"pretrained_model_name_or_path": model_reference}
        if num_labels is not None:
            model_args["num_labels"] = num_labels
        if cache_dir is not None:
            model_args["cache_dir"] = cache_dir
        self.model = BertForTokenClassification.from_pretrained(**model_args)
        self.tokenizer = tokenizer
        self.device = device
        self.trainer = None
        self.idx2tag = {}

    def train(self,
              train_dataset,
              val_dataset,
              logging_dir,
              num_train_epochs,
              logging_steps,
              lr,
              weight_decay,
              warmup_steps,
              output_dir,
              save_model=False
              ):
        args = TrainingArguments(
            output_dir=output_dir,
            logging_dir=logging_dir,
            num_train_epochs=num_train_epochs,
            eval_steps=logging_steps,
            logging_steps=logging_steps,
            learning_rate=lr,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            evaluate_during_training=True,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
        )
        self.trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=AspectModel.compute_metrics(train_dataset.idx2tag)
        )
        train_result = self.trainer.train()
        if save_model:
            self.trainer.save_model(output_dir)
            with open(os.path.join(output_dir, "idx2tag.csv"), "w") as idx2tag_f:
                w = csv.writer(idx2tag_f)
                w.writerows(train_dataset.idx2tag.items())
        return train_result

    def predict(self, predict_dataset):
        if self.trainer is None:
            print("Run train() before making prediction.")
            return None
        return self.trainer.predict(predict_dataset)

    def infer(self, sentence):
        tokens = [word_tokenize(sentence)]
        encoding = self.tokenizer(
            tokens,
            return_tensors="pt",
            is_pretokenized=True,
            padding=True,
            truncation=True
        )
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        subwords = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        max_len = max([len(subword) for subword in subwords])
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        outputs = self.model(input_ids, attention_mask=attention_mask)
        tags = outputs[0][0].argmax(-1).tolist()
        # join subwords
        words = []
        valid_tags = []
        buffer_word = None
        buffer_tag = None
        for i, (subword, tag) in enumerate(zip(subwords, tags)):
            if buffer_word is None:
                buffer_word = subword
                buffer_tag = tag
            elif subword.startswith("##"):
                buffer_word += subword.replace("##", "")
                if i == len(subwords) - 1:
                    words.append(buffer_word)
            else:
                words.append(buffer_word)
                valid_tags.append(buffer_tag)
                buffer_word = subword
                buffer_tag = tag
        return words, valid_tags

    @staticmethod
    def compute_metrics(idx_to_tag):
        def _compute_metrics(pred):
            valid = pred.label_ids != -100
            labels = pred.label_ids[valid].flatten()
            preds = pred.predictions.argmax(-1)[valid].flatten()
            f1 = f1_score(labels, preds, average="micro", zero_division=0)
            report = classification_report(labels, preds, output_dict=True, zero_division=0)
            metrics = {"f1": f1}
            for label in report:
                try:
                    int_label = int(label)
                    if int_label in idx_to_tag:
                        metrics[f"f1_{idx_to_tag[int_label]}"] = report[label]["f1-score"]
                except ValueError as _:
                    pass
            return metrics
        return _compute_metrics
