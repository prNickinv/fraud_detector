import os
import pandas as pd
import sys
import time
import logging

from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import run_preproc
from src.scorer import make_pred
from src.feature_importance import generate_important_features_json
from src.plotting import plot_score_distribution

from config import INPUT_PATH, OUTPUT_PATH, MODEL_PATH, JSON_IMPORTANT_FEATURES_PATH
from config import PNG_SCORE_DIST_PATH, PNG_PRED_LABLES_DIST_PATH

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProcessingService:
    def __init__(self):
        logger.info('Initializing ProcessingService...')
        
        self.input_dir = INPUT_PATH
        self.output_dir = OUTPUT_PATH
        self.model_path = MODEL_PATH

        logger.info('Service initialized')
        
    def process_single_file(self, file_path: str):
        try:
            logger.info(f'Processing file: {file_path}')
            
            input_df = pd.read_csv(file_path)
            
            logger.info('Starting preprocessing...')
            processed_df = run_preproc(input_df)
            
            logger.info('Making predictions...')
            submission, scores = make_pred(processed_df, self.model_path)
            
            logger.info('Generating json for the most important features')
            generate_important_features_json(processed_df, self.model_path, JSON_IMPORTANT_FEATURES_PATH)
            
            logger.info('Generating score distribution plot')
            plot_score_distribution(scores, PNG_SCORE_DIST_PATH, PNG_PRED_LABLES_DIST_PATH)
            
            logger.info("Preparing and saving submission file")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f'predictions_{timestamp}_{os.path.basename(file_path)}'
            submission.to_csv(os.path.join(self.output_dir, output_filename), index=False)
            
            logger.info(f'Predictions saved to file: {output_filename}')
            
        except Exception as e:
            logger.error(f'Error processing file {file_path}: {e}', exc_info=True)
            return
        

class FileHandler(FileSystemEventHandler):
    def __init__(self, service: ProcessingService):
        self.service = service

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".csv"):
            logger.info('New file detected: %s', event.src_path)
            self.service.process_single_file(event.src_path)
            

if __name__ == "__main__":
    logger.info('Starting ML scoring service...')
    
    service = ProcessingService()
    observer = Observer()
    observer.schedule(FileHandler(service), path=service.input_dir, recursive=False)
    observer.start()
    logger.info('File observer started')
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info('Service stopped by user')
        observer.stop()
        
    observer.join()
