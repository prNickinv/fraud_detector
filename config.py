import os
from pathlib import Path


# Features
FEATURES = ['merch', 'cat_id', 'gender', 'one_city', 'us_state', 'jobs', 
            'hour', 'year', 'month', 'day_of_month', 'day_of_week', 
            'amount', 'population_city', 'distance']

CAT_FEATURES = ['merch', 'cat_id', 'gender', 'one_city', 'us_state', 'jobs', 
                'hour', 'year', 'month', 'day_of_month', 'day_of_week']

CONT_FEATURES = ['amount', 'population_city', 'distance']

# Threshold for scoring
MODEL_THRESHOLD = os.getenv('MODEL_THRESHOLD', 0.98)

# Paths
ROOT_PATH = Path(__file__).resolve().parent
MODEL_PATH = os.getenv('MODEL_PATH', os.path.join(ROOT_PATH, 'models', 'fraud_detection_model.cbm'))
INPUT_PATH = os.getenv('INPUT_PATH', os.path.join(ROOT_PATH, 'input'))
OUTPUT_PATH = os.getenv('OUTPUT_PATH', os.path.join(ROOT_PATH, 'output'))

JSON_IMPORTANT_FEATURES_PATH = os.getenv('JSON_IMPORTANT_FEATURES_VALUES_PATH', os.path.join(OUTPUT_PATH, 'important_features.json'))
PNG_SCORE_DIST_PATH = os.getenv('PNG_SCORE_DIST_PATH', os.path.join(OUTPUT_PATH, 'scores_distribution.png'))
PNG_PRED_LABLES_DIST_PATH = os.getenv('PNG_PRED_LABLES_DIST_PATH', os.path.join(OUTPUT_PATH, 'predicted_labels_distribution.png'))
