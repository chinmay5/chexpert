import os
from configparser import ConfigParser

PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

node_labeling_strategy = 'bert'
GRAPH_FILE_NAME = f'curated_{node_labeling_strategy}_graph.pth'

def read_config():
    """
    Function for reading and parsing configurations specific to the attack type
    :return: parsed configurations
    """
    config_path = os.path.join(PROJECT_ROOT_DIR, 'config.ini')
    parser = ConfigParser()
    parser.read(config_path)
    return parser


threshold_dict = {
    "positive": 0.2,
    "negative": 0.3
}

lambda_dict = {
    "positive": 0.25,
    "negative": 0.1
}