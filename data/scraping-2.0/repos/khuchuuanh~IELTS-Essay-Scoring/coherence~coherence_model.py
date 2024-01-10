from lib import *
from coherence_data import *

class ModelClassifier(pl.LightningModule):
    def __init__(self, model_name, num_labels, batch_size, learning_rate=2e-5, hidden_size=512, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_labels = num_labels
        self.model = AutoModel.from_pretrained(model_name)
        self.gru = nn.GRU(self.model.config.hidden_size, hidden_size, batch_first=True)
        self.sentence_gru = nn.GRU(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, self.num_labels)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.out = torch.nn.Softmax(dim=1)

        for param in self.model.encoder.layer[:8].parameters():
            param.requires_grad = False
    def forward(self, input_ids, attention_mask):
        encoder_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state']
        gru_output, _ = self.gru(encoder_outputs)
        gru_output = self.relu(gru_output)
        gru_output_sentence, _ = self.sentence_gru(gru_output)
        gru_output_sentence = self.relu(gru_output_sentence)
        avg_pooled = torch.mean(gru_output_sentence, 1)
        fc_output = self.fc(avg_pooled)
        outputs = self.relu(fc_output)

        return outputs
    
    def training_step(self, batch, batch_idx):
        logits = self(batch[0], batch[1])
        loss = F.cross_entropy(logits, batch[2])
        preds = torch.argmax(logits, 1)
        accuracy = torch.eq(preds, batch[2].long()).float().mean()

        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch[0], batch[1])
        loss = F.cross_entropy(logits, batch[2])
        preds = torch.argmax(logits, 1)
        accuracy = torch.eq(preds, batch[2].long()).float().mean()

        self.log('val_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('val_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True)

        return {'val_loss': loss, 'val_accuracy': accuracy}
    def test_step(self, batch, batch_idx):
        logits = self(batch[:2])
        loss = F.cross_entropy(logits, batch[2])
        preds = torch.argmax(logits, 1)
        accuracy = torch.eq(preds, batch[2].long()).float().mean()

        self.log('test_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('test_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True)

        return {'test_loss': loss, 'test_accuracy': accuracy}

    def validation_epoch_end(self, validation_step_outputs):

        avg_loss = torch.stack([x['val_loss'] for x in validation_step_outputs]).mean()

        avg_accuracy = torch.stack([x['val_accuracy'] for x in validation_step_outputs]).mean()

        self.log("val_loss", avg_loss, prog_bar=True, logger=True)
        self.log("val_accuracy", avg_accuracy, prog_bar=True, logger=True)


    
        return {
            'val_loss': avg_loss,
            'val_accuracy': avg_accuracy
        }


    def setup(self, stage=None):
        train_dataloader = self.trainer.datamodule.train_dataloader()

        # Calculate total steps
        tb_size = self.batch_size * 1
        ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
        self.total_training_steps = (len(train_dataloader.dataset) // tb_size) // ab_size

    def configure_optimizers(self):

        no_decay = ['bias', 'LayerNorm.weight']

        optimizer_grouped_parameters = [
            {
            'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay':0.01
            },
            {
            'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
            }]

        optimizer = AdamW(optimizer_grouped_parameters,
            lr=self.learning_rate,
            eps=1e-5
            )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=self.total_training_steps
            )

        return [optimizer], [scheduler]
    
if __name__ == '__main__':
    AVAIL_GPUS = min(1, torch.cuda.device_count())
    model_name = "bert-base-uncased"
    text_field = "Essay"
    label_field = "COHERENCE AND COHESION"
    data_frame = pd.read_csv('process_data.csv', index_col=0)

    data_module = CustomDataset(model_name, data_frame, text_field, label_field, max_len=512, batch_size=16)
    data_module.setup("fit")

    # logger = WandbLogger(project="COHERENCE")

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath="./save_model/",
            filename="best_checkpoint",
            save_top_k=1,
            verbose=True,
            monitor="val_loss",
            mode="min"
    )

    early_stopping_callback = pl.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    model = ModelClassifier(model_name, len(data_module.label_encoder.classes_), data_module.batch_size)

    trainer = pl.Trainer(
        # logger=logger,
        callbacks=[early_stopping_callback, checkpoint_callback],
        max_epochs=20, deterministic=True,gpus=AVAIL_GPUS)
    
    trainer.fit(model, datamodule=data_module)
