import pytorch_lightning as pl
import torch
import torch.nn as nn  # all neural network models
from transformers import AutoModel, AutoTokenizer
import argparse
from torchmetrics import R2Score


class ArgQualityModel(pl.LightningModule):

    def __init__(self, model_name, learning_rate, weight_decay, dropout_prob):
        super().__init__()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout_prob = dropout_prob
        self.dropout = nn.AlphaDropout(self.dropout_prob)

        # BERT encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.dropout_prob > 0:
            self.ffn = nn.Sequential(nn.Linear(768, 512), nn.SELU(), nn.AlphaDropout(self.dropout_prob), nn.Linear(512, 256), nn.SELU(), nn.AlphaDropout(self.dropout_prob), nn.Linear(256, 1))
        else:
            self.ffn = nn.Sequential(nn.Linear(768, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(),
                                     nn.Linear(256, 1))

        self.loss_fn = nn.MSELoss()
        self.train_r2score = R2Score()
        self.val_r2score = R2Score()
        self.test_r2score = R2Score()

        for param in self.encoder.base_model.parameters():
            param.requires_grad = False

        self.save_hyperparameters()

    def forward(self, batch):
        batch_input = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
        encoder_output = self.encoder(**batch_input, output_hidden_states=True)  # BERT pass
        cls_emb = encoder_output.last_hidden_state[:, 0, :]
        cls_emb = self.dropout(cls_emb)
        batch_scores = self.ffn(cls_emb)

        textual_args = self.tokenizer.batch_decode(batch_input['input_ids'], skip_special_tokens=True)
        arg_to_score = [[x, batch_scores[idx][0].item()] for idx, x in enumerate(textual_args)]
        return arg_to_score

    def training_step(self, batch, batch_idx):
        batch_input, batch_targets = batch
        encoder_output = self.encoder(**batch_input, output_hidden_states=True)  # BERT pass
        cls_emb = encoder_output.last_hidden_state[:, 0, :]
        cls_emb = self.dropout(cls_emb)
        batch_scores = self.ffn(cls_emb)

        loss = self.loss_fn(batch_scores, batch_targets)
        self.log('train_loss', loss, on_step=True, on_epoch=True)

        if batch_targets.shape[0] >= 2:  # 2 samples at least needed for R2 computation
            train_r2 = self.train_r2score(batch_scores, batch_targets)
            self.log('train_r2', train_r2, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        batch_input, batch_targets = batch
        encoder_output = self.encoder(**batch_input, output_hidden_states=True)  # BERT pass
        cls_emb = encoder_output.last_hidden_state[:, 0, :]
        cls_emb = self.dropout(cls_emb)
        batch_scores = self.ffn(cls_emb)

        loss = self.loss_fn(batch_scores, batch_targets)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        if batch_targets.shape[0] >= 2:  # 2 samples at least needed for R2 computation
            val_r2 = self.val_r2score(batch_scores, batch_targets)
            self.log('val_r2', val_r2, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        batch_input, batch_targets = batch
        encoder_output = self.encoder(**batch_input, output_hidden_states=True)  # BERT pass
        cls_emb = encoder_output.last_hidden_state[:, 0, :]
        cls_emb = self.dropout(cls_emb)
        batch_scores = self.ffn(cls_emb)

        loss = self.loss_fn(batch_scores, batch_targets)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        if batch_targets.shape[0] >= 2:  # 2 samples at least needed for R2 computation
            test_r2 = self.test_r2score(batch_scores, batch_targets)
            self.log('test_r2', test_r2, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer

    @staticmethod
    def add_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("ArgQualityModel")
        parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
        parser.add_argument('-w', '--weight_decay', type=float, default=1e-5)
        return parent_parser
