import csv
import os

import torch
from torch import nn
from nltk import word_tokenize
from torchcrf import CRF
from transformers import BertForTokenClassification, Trainer, TrainingArguments, BertPreTrainedModel, BertModel
from sklearn.metrics import f1_score, classification_report


class BertCrfForTokenClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.fc = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.fc(sequence_output)
        seq_len = logits.shape[1]

        crf_mask = attention_mask == 1
        crf_out = self.crf.decode(logits, mask=crf_mask)
        crf_out_padded = torch.FloatTensor([seq + [0] * (seq_len - len(seq)) for seq in crf_out])
        outputs = (crf_out_padded,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            crf_loss = -self.crf(logits, tags=labels, mask=crf_mask)
            outputs = (crf_loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)


class AspectModel(object):

    def __init__(self, model_reference, tokenizer, device, use_crf=False, hidden_dropout_prob=0.1, num_labels=None, cache_dir=None):
        model_args = {
            "pretrained_model_name_or_path": model_reference,
            "hidden_dropout_prob": hidden_dropout_prob
        }
        if num_labels is not None:
            model_args["num_labels"] = num_labels
        if cache_dir is not None:
            model_args["cache_dir"] = cache_dir
        self.model = BertCrfForTokenClassification(**model_args) if use_crf \
            else BertForTokenClassification.from_pretrained(**model_args)
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
