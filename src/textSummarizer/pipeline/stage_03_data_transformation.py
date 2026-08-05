from textSummarizer.components.data_transformation import DataTransformation
from textSummarizer.config.configuration import ConfiguarationManager 

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfiguarationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.convert()
