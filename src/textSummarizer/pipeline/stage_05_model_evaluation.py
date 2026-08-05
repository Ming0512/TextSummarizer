from textSummarizer.components.model_evaluation import ModelEvaluation
from textSummarizer.config.configuration import ConfiguarationManager

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfiguarationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluator = ModelEvaluation(config=model_evaluation_config)
        model_evaluator.evaluate()
