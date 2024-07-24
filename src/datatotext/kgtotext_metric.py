from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    AutoTokenizer,
    T5Config,
    T5ForSequenceClassification,
    BertConfig,
    BertForSequenceClassification,
)
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics



class FactCheckingClassifier(pl.LightningModule):
    def __init__(
            self,
            pretrained_model_name: str,
            num_classes: int = 2,
            learning_rate_pretrained: float = 1e-3,
            learning_rate_classifier: float = 1e-3,
            adam_epsilon_pretrained: float = 1e-8,
            adam_epsilon_classifier: float = 1e-8,
            warmup_steps_pretrained: int = 0,
            warmup_steps_classifier: int = 0,
            weight_decay_pretrained: float = 0.0,
            weight_decay_classifier: float = 0.0,
            gradient_clip_val: float = None,
        ):
        super().__init__()

        if "t5" in pretrained_model_name:
            self.config = T5Config.from_pretrained(pretrained_model_name)
            self.config.classifier_dropout = 0.0
            self.config.preseqlen = 48
            self.config.num_labels = num_classes
            self.config.use_prefix = False
            self.model = T5ForSequenceClassification.from_pretrained(
                pretrained_model_name, config=self.config
            )
            self.model.classifier = self.model.classification_head
        elif "bert" in pretrained_model_name:
            self.config = BertConfig.from_pretrained(pretrained_model_name)
            self.config.num_labels = num_classes
            self.model = BertForSequenceClassification.from_pretrained(
                pretrained_model_name, config=self.config
            )
            self.model.transformer = self.model.bert
        else:
            raise ValueError("Only T5 and BERT models are supported")
        

        self.hparams_pretrained = {
            "learning_rate": learning_rate_pretrained,
            "adam_epsilon": adam_epsilon_pretrained,
            "warmup_steps": warmup_steps_pretrained,
            "weight_decay": weight_decay_pretrained,
        }
        self.hparams_classifier = {
            "learning_rate": learning_rate_classifier,
            "adam_epsilon": adam_epsilon_classifier,
            "warmup_steps": warmup_steps_classifier,
            "weight_decay": weight_decay_classifier,
        }

        self.gradient_clipping_val = gradient_clip_val
        self.validation_step_outputs = []
        self.validatation_step_targets = []
        self.binary = num_classes == 2

        self.automatic_optimization = False

    def forward(self, input):
        return self.model(
            input_ids=input["input_ids"],
            attention_mask=input["attention_mask"],
            return_dict=False,
        )[0]

    def training_step(self, batch, batch_idx):
        pretrained_opt, cls_opt = self.optimizers()
        pretrained_scheduler, cls_scheduler = self.lr_schedulers()
        pretrained_opt.zero_grad()
        cls_opt.zero_grad()
    
        outputs = self(batch["input"])
        loss = self.loss(outputs, batch["target"])
        self.manual_backward(loss)
        pretrained_opt.step()
        cls_opt.step()
        # clip gradients
        if self.gradient_clipping_val is not None:
            self.clip_gradients(pretrained_opt, gradient_clip_val=self.gradient_clipping_val, gradient_clip_algorithm="norm")
            self.clip_gradients(cls_opt, gradient_clip_val=self.gradient_clipping_val, gradient_clip_algorithm="norm")
        
        pretrained_scheduler.step()
        cls_scheduler.step()

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch["input"])
        loss = self.loss(outputs, batch["target"])
        self.log("val_loss", loss)
        self.validation_step_outputs.append(torch.argmax(outputs, dim=-1))
        self.validatation_step_targets.append(batch["target"])

    def on_validation_epoch_end(self):
        all_preds = torch.stack(self.validation_step_outputs, dim=0).flatten()
        all_targets = torch.stack(self.validatation_step_targets, dim=0).flatten()

        # log mean and case specific accuracy, f1 score, precision and recall
        accuracy = torchmetrics.functional.accuracy(
            all_preds,
            all_targets,
            average="none",
            task="binary" if self.binary else "multiclass",
            num_classes=self.config.num_labels,
        )
        f1_score = torchmetrics.functional.f1_score(
            all_preds,
            all_targets,
            average="none",
            task="binary" if self.binary else "multiclass",
            num_classes=self.config.num_labels,
        )
        precision = torchmetrics.functional.precision(
            all_preds,
            all_targets,
            average="none",
            task="binary" if self.binary else "multiclass",
            num_classes=self.config.num_labels,
        )
        recall = torchmetrics.functional.recall(
            all_preds,
            all_targets,
            average="none",
            task="binary" if self.binary else "multiclass",
            num_classes=self.config.num_labels,
        )

        if not self.binary:
            # log per class
            for i in range(self.config.num_labels):
                self.log(f"val_accuracy_{i}", accuracy[i])
                self.log(f"val_f1_score_{i}", f1_score[i])
                self.log(f"val_precision_{i}", precision[i])
                self.log(f"val_recall_{i}", recall[i])

            # 'macro' average
            accuracy = accuracy.mean()
            f1_score = f1_score.mean()
            precision = precision.mean()
            recall = recall.mean()

        self.log("val_accuracy_mean", accuracy)
        self.log("val_f1_score_mean", f1_score)
        self.log("val_precision_mean", precision)
        self.log("val_recall_mean", recall)

        self.validation_step_outputs.clear()
        self.validatation_step_targets.clear()

    def configure_optimizers(self):
        """Prepare optimizers and schedule (linear warmup and decay)"""
        models = [self.model.transformer, self.model.classifier]
        hparams = [self.hparams_pretrained, self.hparams_classifier]
        no_decay = ["bias", "LayerNorm.weight"]
        optimizers = []
        schedulers = []
        for model, hps in zip(models, hparams):
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": hps["weight_decay"],
                },
                {
                    "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=hps["learning_rate"], eps=hps["adam_epsilon"])

            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=hps["warmup_steps"],
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
            scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
            optimizers.append(optimizer)
            schedulers.append(scheduler)
            
        return optimizers, schedulers

    def loss(self, logits, target):
        return F.cross_entropy(logits, target) 


class FactCheckingLoss(torch.nn.Module):
    def __init__(
            self,
            checkpoint_path: str = "/app/metric/results/multiclass/t5-v0.1/weights/pytorch-model.ckpt",
            model_name: str = "t5-small",
            num_classes: int = 5
        ):
        super().__init__()
        self.checker = FactCheckingClassifier.load_from_checkpoint(
            checkpoint_path,
            pretrained_model_name=model_name,
            num_classes=num_classes,
            strict=False
        )
        self.checker.eval()
        
        self.tockenizer = AutoTokenizer.from_pretrained(model_name)
        
        tockenizer = self.tockenizer

        # tensor([[    3,     5,     3, 20119,    10,     1]])
        # torch.Size([1, 6])
        sep_token = ". Result: "
        self.sep_token_id = tockenizer(
            sep_token,
            return_tensors='pt',
            padding='do_not_pad',
            )["input_ids"][0]
        self.sep_token_len = len(self.sep_token_id)

        self.pad_token_id = tockenizer.pad_token_id

        self.maxlen = self.checker.config.n_positions

        self.eos_token_id = tockenizer.eos_token_id
        self.label = 0 if num_classes == 5 else 1

    def forward(
        self,
        model_input_ids: torch.Tensor,
        model_input_attention_mask: torch.Tensor,
        model_output_ids: torch.Tensor,
    ):
        input_lengths = model_input_attention_mask.sum(dim=1)
        output_lengths = (model_output_ids == self.eos_token_id).int().argmax(dim=1)

        # Calculate the lengths of the concatenated sequences
        concat_lengths = torch.min(input_lengths + output_lengths, torch.tensor([self.maxlen]).to(model_input_ids.device))
        concat_lengths = torch.add(concat_lengths, self.sep_token_len)

        # Create a tensor to hold the concatenated sequences
        concat_input = torch.full((input_lengths.shape[0], self.maxlen), self.pad_token_id).to(model_input_ids.device)

        # For each sequence, copy the elements from tensor_a and tensor_b to tensor_c
        for i in range(input_lengths.shape[0]):
            concat_input[i, :input_lengths[i]] = model_input_ids[i, :input_lengths[i]]
            concat_input[i, input_lengths[i]:input_lengths[i]+self.sep_token_len] = self.sep_token_id
            end = min(input_lengths[i] + self.sep_token_len + output_lengths[i], self.maxlen)
            if end > input_lengths[i]+self.sep_token_len:
                concat_input[i, input_lengths[i]+self.sep_token_len:end] = model_output_ids[i, :end-input_lengths[i]-self.sep_token_len]

        # Set the last element to -1 if the sequence was truncated
        concat_input[concat_lengths < self.maxlen, concat_lengths[concat_lengths < self.maxlen]] = self.eos_token_id
       
        
        logits = self.checker(
            {
                "input_ids": concat_input,
                "attention_mask": (concat_input != self.pad_token_id).int()
            }
        )
        target = torch.full([logits.shape[0]], self.label, device=logits.device)
        return F.cross_entropy(input=logits, target=target)
    

# metric = FactCheckingLoss(
#     "/app/metric/src/checkpoints/multiclass/T5/lr_5e-05_0.0005_bs_16/2024-05-27_12-43-07/epoch=12_val_loss=0.0248.ckpt",
#     num_classes=5
#     )

# model_input = "translate Graph to English: <H> France <R> capital <T> Paris"
# model_output = "The capital of France is Paris."
# loss = metric(model_input, model_output)
# print(loss)
# # tensor(8.0463e-05, grad_fn=<NllLossBackward0>)


# model_input = "translate Graph to English: <H> France <R> capital <T> London"
# model_output = "The capital of France is Paris."
# loss = metric(model_input, model_output)
# print(loss)
# # tensor(0.1003, grad_fn=<NllLossBackward0>)

# model_input = "translate Graph to English: <H> Italy <R> capital <T> Rome"
# model_output = "The capital of Italy is Rome."
# loss = metric(model_input, model_output)
# print(loss)
# tensor(0.1003, grad_fn=<NllLossBackward0>)
