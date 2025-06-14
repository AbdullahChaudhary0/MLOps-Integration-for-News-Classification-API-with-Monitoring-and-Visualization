from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.python import PythonSensor
from datetime import datetime, timedelta
import os
import pandas as pd
import string
import logging
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from sklearn.model_selection import train_test_split

# Ensure NLTK data is available
nltk.download('stopwords')
nltk.download('wordnet')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default arguments for the DAG
default_args = {
    'owner': 'ahmed_aqib',
    'depends_on_past': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'news_pipeline_dag',
    default_args=default_args,
    description='NLP pipeline for news category dataset',
    schedule_interval='@daily',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['news', 'nlp', 'pipeline'],
)

# Input JSON and output directory inside your project
BASE_DIR = '/root/MLOPS Project/mlops-course-project-ahmed_aqib/airflow_project'
FILE_PATH = os.path.join(BASE_DIR, 'data', 'News_Category_Dataset_v3.json')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data')

def check_file_exists():
    exists = os.path.exists(FILE_PATH)
    logger.info(f"Checking for dataset file at {FILE_PATH}: {exists}")
    return exists

wait_for_file_sensor = PythonSensor(
    task_id='wait_for_file',
    python_callable=check_file_exists,
    poke_interval=10,
    timeout=300,
    mode='reschedule',
    dag=dag,
)

def basic_cleaning(**context):
    logger.info("Starting basic cleaning")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_json(FILE_PATH, lines=True)
    df['clean_headline'] = (
        df['headline']
        .str.lower()
        .str.replace(f"[{string.punctuation}]", "", regex=True)
    )
    cleaned_path = os.path.join(OUTPUT_DIR, 'cleaned_news.csv')
    df.to_csv(cleaned_path, index=False)
    logger.info(f"Basic cleaning complete, saved to {cleaned_path}")
    return cleaned_path  # Return value will be pushed to XCom automatically

basic_cleaning_task = PythonOperator(
    task_id='basic_cleaning',
    python_callable=basic_cleaning,
    dag=dag,
)

def advanced_processing(**context):
    """
    Advanced text processing: stopword removal and lemmatization.
    Falls back to OUTPUT_DIR/cleaned_news.csv if XCom is empty.
    """
    logger.info("Starting advanced processing")
    ti = context['ti']

    # Try to pull cleaned_path from XCom, else use default location
    cleaned_path = ti.xcom_pull(task_ids='basic_cleaning')
    if not cleaned_path:
        cleaned_path = os.path.join(OUTPUT_DIR, 'cleaned_news.csv')
        logger.warning(f"No XCom cleaned_path found; falling back to {cleaned_path}")
    else:
        logger.info(f"Pulled cleaned_path from XCom: {cleaned_path}")

    # Validate existence
    if not os.path.exists(cleaned_path):
        raise FileNotFoundError(f"Cleaned file not found at {cleaned_path}")

    # Read and process
    df = pd.read_csv(cleaned_path)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def process_text(text):
        if not isinstance(text, str):
            return ''
        tokens = [w for w in text.split() if w not in stop_words]
        return ' '.join([lemmatizer.lemmatize(w) for w in tokens])

    df['processed_headline'] = df['clean_headline'].apply(process_text)

    processed_path = os.path.join(OUTPUT_DIR, 'processed_news.csv')
    df.to_csv(processed_path, index=False)
    logger.info(f"Advanced processing complete, saved to {processed_path}")
    return processed_path  # Return value will be pushed to XCom automatically

advanced_processing_task = PythonOperator(
    task_id='advanced_processing',
    python_callable=advanced_processing,
    dag=dag,
)

def feature_engineering(**context):
    ti = context['ti']
    processed_path = ti.xcom_pull(task_ids='advanced_processing', key='processed_path')
    if not processed_path or not os.path.exists(processed_path):
        processed_path = os.path.join(OUTPUT_DIR, 'processed_news.csv')
    logging.info(f"Feature engineering reading from {processed_path}")

    df = pd.read_csv(processed_path)

    # Ensure no NaNs in processed_headline
    df['processed_headline'] = df['processed_headline'].fillna('')

    df['text_length'] = df['processed_headline'].str.len()
    df['sentiment'] = df['processed_headline'].apply(
        lambda x: TextBlob(x).sentiment.polarity
    )

    feature_path = os.path.join(OUTPUT_DIR, 'featured_news.csv')
    df.to_csv(feature_path, index=False)
    logging.info(f"Feature engineering complete, saved to {feature_path}")
    ti.xcom_push(key='feature_path', value=feature_path)
    return feature_path  # Return value will be pushed to XCom automatically
feature_engineering_task = PythonOperator(
    task_id='feature_engineering',
    python_callable=feature_engineering,
    dag=dag,
)

def split_data(**context):
    ti = context['ti']
    feature_path = ti.xcom_pull(task_ids='feature_engineering')
    
    if not feature_path:
        feature_path = os.path.join(OUTPUT_DIR, 'featured_news.csv')
        logger.warning(f"No feature_path in XCom; trying default location: {feature_path}")
    
    logger.info(f"Splitting data from {feature_path}")
    
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Feature file not found at {feature_path}")
    
    df = pd.read_csv(feature_path)
    train_val, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['category'])
    train, val = train_test_split(train_val, test_size=0.25, random_state=42, stratify=train_val['category'])
    
    result_paths = {}
    for name, sub_df in zip(['train','val','test'], [train, val, test]):
        path = os.path.join(OUTPUT_DIR, f"{name}.csv")
        sub_df.to_csv(path, index=False)
        logger.info(f"{name.capitalize()} split saved to {path}")
        result_paths[name] = path
    
    return result_paths

split_data_task = PythonOperator(
    task_id='split_data',
    python_callable=split_data,
    dag=dag,
)

wait_for_file_sensor >> basic_cleaning_task >> advanced_processing_task >> feature_engineering_task >> split_data_task