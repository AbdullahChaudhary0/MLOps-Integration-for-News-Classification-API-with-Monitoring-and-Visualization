# MLOps-Integration-for-News-Classification-API-with-Monitoring-and-Visualization
End-to-end MLOps pipeline for text classification creating pipelines through apache airflow DAGS, models comparison and version managing through mlflow and integrated through FASTAPI and monitored constantly through Prometheus and Grafana.

4.1 Data Pipeline Orchestration with Airflow
• Set up Apache Airflow for orchestrating data processing pipelines
• Create DAGs (Directed Acyclic Graphs) for:
– Data ingestion from the news dataset source
– Data preprocessing with multiple stages:
∗ Basic cleaning (lowercase, remove punctuation)
∗ Advanced processing (stopword removal, lemmatization)
∗ Feature engineering (text length, sentiment scores)
– Data splitting (train/validation/test) with consistent random seeds
• Implement proper error handling and retry mechanisms
• Create sensors to monitor the completion of upstream tasks
• Schedule periodic pipeline runs
• Document all Airflow operators and data transformations
Page 1
MLOps Course NLP Pipeline Project
4.2 NLP Model Development with MLflow
• Experiment with at least 4 different approaches:
– Traditional ML with TF-IDF features (e.g., Naive Bayes, SVM)
– Word embeddings with simple neural networks
– Pre-trained embeddings (Word2Vec, GloVe)
– Simple transformer-based approaches (e.g., DistilBERT with limited layers)
• Track all experiments in MLflow with:
– Model hyperparameters
– Performance metrics (accuracy, F1-score, confusion matrix)
– Training and inference time
– Dataset version used
• Register the best performing models to MLflow Model Registry
• Create visualizations comparing model performance
4.3 Model Serving API (FastAPI)
• Develop a REST API that:
– Loads the best model from MLflow
– Provides endpoints for:
∗ Single text classification
∗ Batch prediction
∗ Model information (version, metrics)
– Includes proper input validation
– Returns prediction probabilities for top categories
• Document API with Swagger/OpenAPI
• Implement request logging and error handling
• Create a simple web demo for testing the API
4.4 Monitoring System
• Instrument the API with Prometheus metrics:
– Request count and latency
– Prediction distribution across categories
– Input text characteristics (length, language detection)
– Error rates
• Create Grafana dashboards to visualize:
– API performance
– Prediction distributions
– Data drift indicators
• Implement basic alerting for unusual patterns

