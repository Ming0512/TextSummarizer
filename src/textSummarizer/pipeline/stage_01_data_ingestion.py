from textSummarizer.components.data_ingestion import DataIngestion
from textSummarizer.config.configuration import ConfiguarationManager 

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfiguarationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_zipfile()
        data_ingestion.extract_zipfile()
