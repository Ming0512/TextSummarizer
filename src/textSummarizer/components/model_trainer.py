from transformers import (
                    TrainingArguments, 
                    Trainer,
                    DataCollatorForSeq2Seq,
                    AutoModelForSeq2SeqLM, 
                    AutoTokenizer)
from textSummarizer.entity.config_entity import ModelTrainerConfig
from datasets import load_dataset, load_from_disk
import torch
import os

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)

        print("Loading model...")
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)

        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)

        dataset_samsum_pt = load_from_disk(self.config.data_path)

        trainer_args = TrainingArguments(
                output_dir=self.config.root_dir,
                num_train_epochs=self.config.num_train_epochs,
                learning_rate=self.config.learning_rate,
                warmup_steps=self.config.warmup_steps,
                per_device_train_batch_size=self.config.per_device_train_batch_size,
                weight_decay=self.config.weight_decay,
                logging_steps=self.config.logging_steps,

                eval_strategy=self.config.evaluation_strategy,
                eval_steps=self.config.eval_steps,

                save_strategy=self.config.evaluation_strategy,   # Must match eval_strategy
                save_steps=self.config.save_steps,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,

                # Best model settings
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                save_total_limit=1,

                report_to="none"
            )
        trainer = Trainer(
            model=model_pegasus,
            args=trainer_args,
            processing_class=tokenizer,
            data_collator=seq2seq_data_collator,
            train_dataset=dataset_samsum_pt["train"],
            eval_dataset=dataset_samsum_pt["validation"],
        )
        trainer.train()
        model_pegasus.save_pretrained(os.path.join(self.config.root_dir, 'pegasus-samsum-model'))
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, 'tokenizer'))