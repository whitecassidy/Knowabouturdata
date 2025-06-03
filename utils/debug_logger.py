import logging
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('AskurData')

def log_df_info(df, context=""):
    """Log DataFrame information"""
    logger.debug(f"{context} DataFrame Info:")
    logger.debug(f"Shape: {df.shape}")
    logger.debug(f"Columns: {df.columns.tolist()}")
    logger.debug(f"Data Types:\n{df.dtypes}")
    logger.debug(f"Missing Values:\n{df.isnull().sum()}")

def log_llm_interaction(prompt, response, context=""):
    """Log LLM prompt and response"""
    logger.debug(f"\n{'='*50}\n{context} LLM Interaction at {datetime.now()}\n{'='*50}")
    logger.debug(f"Prompt:\n{prompt}")
    logger.debug(f"Response (first 500 chars):\n{str(response)[:500]}")

def log_error(error, context=""):
    """Log error with context"""
    logger.error(f"{context} Error: {str(error)}")

def log_state_update(state_key, value, context=""):
    """Log Streamlit session state updates"""
    logger.debug(f"{context} State Update - {state_key}:")
    if isinstance(value, (str, int, float, bool)):
        logger.debug(f"Value: {value}")
    else:
        logger.debug(f"Type: {type(value)}")
        if hasattr(value, '__len__'):
            logger.debug(f"Length: {len(value)}") 