from textSummarizer.components.model_trainer import ModelTrainer
from textSummarizer.config.configuration import ConfiguarationManager

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfiguarationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        model_trainer.train()
