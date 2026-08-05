from textSummarizer.components.data_validation import DataValidation
from textSummarizer.config.configuration import ConfiguarationManager 

class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfiguarationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        data_validation.validate_all_file_exists()
