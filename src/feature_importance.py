import json
import logging
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier

logger = logging.getLogger(__name__)


def generate_important_features_json(df, model_path, output_path):
    """Retrieves and saves the top 5 important features values."""
    model = CatBoostClassifier()
    model.load_model(model_path)
    
    most_important_feat_names = model.get_feature_importance(prettified=True)['Feature Id'][:5].tolist()
    output_dict = dict(zip(most_important_feat_names, [df[imp_feature].tolist() for imp_feature in most_important_feat_names]))
    
    with open(output_path, 'w') as file:
        json.dump(output_dict, file, indent=4)
