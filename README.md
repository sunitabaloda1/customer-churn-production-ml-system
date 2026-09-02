# Production ML System for Customer Churn Prediction

An end-to-end, production-oriented machine learning system for predicting customer churn, covering data ingestion, feature engineering, model training, model selection, model persistence, online inference, testing, monitoring, and operational considerations.

## Business Problem

Customer churn is an important business problem for telecom and subscription-based organizations. Identifying customers who are likely to churn can help organizations prioritize retention activities and improve customer lifetime value.

This project builds a machine learning system that predicts whether a customer is likely to churn based on customer demographics, services, contract information, payment methods, tenure, and billing characteristics.

The system is designed with a production-oriented mindset rather than as a standalone machine learning notebook.

## Solution Overview

The project implements two connected workflows.

### Offline Training Pipeline

<pre>
Raw Customer Data
       ↓
Data Ingestion
       ↓
Data Quality Validation
       ↓
Feature Engineering
       ↓
Train/Test Split
       ↓
Preprocessing Pipeline
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Model Promotion
       ↓
Persisted Model + Metrics
</pre>

### Online Inference Pipeline

<pre>
Client Request
       ↓
FastAPI /predict
       ↓
Input Validation
       ↓
Shared Preprocessing
       ↓
Persisted ML Model
       ↓
Churn Prediction
       ↓
Probability + Prediction + Model Version
</pre>

## Key Project Highlights

| Area | Implementation |
|---|---|
| Problem | Customer churn prediction |
| ML Type | Binary classification |
| Dataset | IBM Telco Customer Churn dataset |
| Records | 7,043 |
| Features | 21 |
| Baseline Model | Logistic Regression |
| Candidate Model | Random Forest |
| Model Selection | ROC-AUC based promotion rule |
| API | FastAPI |
| Testing | Pytest + manual API testing |
| Monitoring | Data quality + feature drift |
| Configuration | Centralized configuration |
| Logging | Application and pipeline logging |
| Model Persistence | Serialized model and preprocessing artifacts |
| Performance | Average API latency of 42.98 ms |
| Architecture | Modular offline training + online inference |

## Dataset

The project uses the IBM Telco Customer Churn dataset, containing customer-level information and a binary `Churn` target.

The dataset contains:

- 7,043 customer records
- 21 input features
- Binary target variable: `Churn`

The dataset includes information related to:

- Customer demographics
- Tenure
- Phone and internet services
- Contract type
- Payment method
- Monthly charges
- Total charges
- Additional subscribed services

The original dataset is publicly available through the Kaggle Telco Customer Churn dataset.

## Feature Engineering

Additional business-oriented features were created to improve the representation of customer behavior.

| Feature | Description |
|---|---|
| `TotalServices` | Number of subscribed telecom services |
| `AvgMonthlySpend` | Average monthly customer spend |
| `IsLongTermCustomer` | Indicator for longer-tenure customers |
| `HasAutoPayment` | Indicator for automatic payment |
| `FiberCustomer` | Indicator for fiber internet customers |

The feature engineering logic is implemented as part of the shared preprocessing workflow to maintain consistency between training and inference.

## Data Preparation

The training workflow includes:

- Data ingestion
- Schema and data-quality validation
- Missing-value checks
- Feature engineering
- Stratified train/test split
- Numerical feature preprocessing
- Categorical feature preprocessing
- Shared preprocessing pipeline

A Scikit-learn `ColumnTransformer` is used so that the same preprocessing logic is applied during both model training and online prediction.

This reduces the risk of training-serving skew.

## Model Development

Two models were developed and evaluated.

### Logistic Regression

Used as the baseline classification model.

Advantages:

- Simple
- Interpretable
- Fast to train and predict
- Provides a strong baseline for comparison

### Random Forest

Used as the candidate model.

Advantages:

- Captures non-linear relationships
- Handles interactions between features
- Provides a complementary approach to the linear baseline

## Model Promotion Policy

A simple model promotion gate was implemented.

The candidate model is promoted only when:

1. Candidate ROC-AUC is at least **0.80**
2. Candidate ROC-AUC is equal to or better than the baseline model

Otherwise, the baseline model is retained.

This prevents a candidate model from replacing the existing model unless it satisfies the defined performance requirement.

## Model Persistence

The trained model and related artifacts are persisted for reuse by the inference service.

| Artifact | Purpose |
|---|---|
| `customer_churn_model.pkl` | Persisted approved ML model |
| `preprocessor.pkl` | Persisted preprocessing pipeline |
| `metrics.json` | Model evaluation metrics |
| `latency_report.json` | API performance measurements |

Model persistence separates model training from online inference and allows the API service to load the approved model without retraining.

## FastAPI Deployment

The trained model is exposed through a FastAPI service.

### Available Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Basic service information |
| `/health` | GET | Service health check |
| `/predict` | POST | Generate customer churn prediction |

The `/predict` endpoint returns:

- Predicted churn class
- Churn probability
- Model version

Swagger UI is available through FastAPI for interactive API testing.

## API Performance

The prediction API was tested using 20 consecutive requests.

| Metric | Result |
|---|---:|
| Average latency | 42.98 ms |
| Minimum latency | 25.81 ms |
| Maximum latency | 125.88 ms |
| P95 latency | 64.74 ms |

The first request was comparatively slower due to model loading/warm-up. Subsequent requests generally completed within approximately 26–40 ms.

The observed latency demonstrates that the current lightweight implementation can support low-latency online inference within the scope of this mini production ML system.

## Evaluation & Production Considerations

The system was evaluated from both a machine learning perspective and an operational perspective.

### Model Performance

Model performance is evaluated using classification metrics, with ROC-AUC used as the primary model promotion metric.

The promotion policy ensures that the selected model meets the defined minimum performance threshold and does not replace the baseline with a weaker candidate.

### Latency

Online inference latency was measured using 20 consecutive API requests.

The measured average latency was 42.98 ms and P95 latency was 64.74 ms.

Latency testing provides an initial indication of the responsiveness of the prediction service.

### Scalability

The current implementation is designed as a lightweight FastAPI service and is appropriate for the scope of this assignment.

For higher request volumes, the architecture could be extended through:

- Containerized deployment
- Multiple API instances
- Horizontal scaling
- Load balancing
- Cloud-based compute infrastructure
- Separation of training and inference workloads

The modular separation between offline training and online inference makes these extensions possible without fundamentally changing the prediction interface.

### Reliability

The system includes several mechanisms to improve operational reliability:

- Health-check endpoint
- Model artifact validation
- Configuration management
- Application logging
- Data-quality validation
- Input validation
- Error handling
- Automated tests using Pytest
- Incident-response procedures

The `/health` endpoint provides a basic service availability check, while persisted model artifacts ensure that the inference service consistently loads the approved model.

### Monitoring

Lightweight monitoring has been implemented for data quality and feature drift.

Current monitoring includes:

- Missing-value detection
- Data-quality validation
- Ingestion logging
- Feature drift detection
- Monitoring of `MonthlyCharges`
- Comparison against a baseline feature mean

The baseline `MonthlyCharges` mean is **64.76**.

A **10% drift threshold** is used for the current lightweight drift check.

For the evaluated dataset, observed drift was **0.00%**, which passed the defined threshold.

### Cost Considerations

The current system has relatively low infrastructure complexity because it uses a lightweight API-based inference architecture.

For a cloud deployment, the major cost considerations would include:

- Compute resources
- Dataset and model storage
- Logging and monitoring
- Training and retraining workloads
- Network and managed cloud services

For larger-scale deployment, infrastructure sizing would need to balance prediction volume, latency requirements, reliability, retraining frequency, and business value.

## Data Ingestion

The data ingestion component reads the raw customer churn dataset and creates or updates the training data used by the system.

The ingestion workflow supports:

- Reading raw data
- Creating the training dataset
- Appending new records during subsequent ingestion
- Record-count logging
- Data-quality validation
- Missing-value checks
- Feature drift checks

This provides a simple foundation for evolving the project from a static dataset toward a continuously updated ML workflow.

## Monitoring & Data Quality

The system performs lightweight data-quality monitoring before model training.

Checks include:

- Missing values
- Required columns
- Data consistency
- Record counts
- Feature drift

Warnings are generated when data-quality issues or drift conditions are detected.

The current drift implementation uses `MonthlyCharges` as a representative monitored feature.

More advanced production monitoring could include:

- Population Stability Index (PSI)
- Kolmogorov-Smirnov (KS) tests
- Distribution monitoring across multiple features
- Prediction distribution monitoring
- Model performance monitoring
- Automated alerting

## Retraining Strategy

The project defines practical triggers that could initiate model retraining.

Potential retraining triggers include:

- ROC-AUC falling below 0.80
- At least 5,000 new training records becoming available
- Missing-value ratio exceeding 5%
- Significant feature drift being detected

These rules provide a simple framework for moving toward a more automated model lifecycle.

## Testing

The system was tested through multiple mechanisms:

- FastAPI Swagger UI testing
- Pytest-based automated tests
- Manual prediction testing
- Health endpoint testing
- Model artifact loading validation
- API response validation
- Latency testing

Testing covers both individual components and the end-to-end prediction workflow.

## Incident Response

The project includes documented responses for common operational scenarios.

### Missing or Invalid Columns

- Detect schema mismatch
- Log the issue
- Stop or reject invalid ingestion
- Investigate source-data changes
- Restore expected schema before retraining

### Ingestion Failure

- Capture the failure in logs
- Validate input availability
- Check data format and schema
- Retry or correct the ingestion process
- Prevent invalid data from reaching model training

### Prediction Service Interruption

- Check service health
- Review application logs
- Validate model artifacts
- Restart or redeploy the service
- Verify `/health`
- Execute a test prediction

## Repository Structure

<pre>
customer-churn-production-ml-system/
│
├── artifacts/
│   ├── metrics.json
│   └── latency_report.json
│
├── data/
│
├── docs/
│
├── logs/
│
├── models/
│
├── notebooks/
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── inference/
│
├── tests/
│
├── .gitignore
├── main.py
├── README.md
└── requirements.txt
</pre>

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- FastAPI
- Uvicorn
- Pytest
- Joblib
- JSON-based configuration and metrics
- Git / GitHub

## Running the Project

### 1. Install dependencies

`pip install -r requirements.txt`

### 2. Run the training pipeline

`python main.py`

### 3. Start the FastAPI service

`uvicorn src.api:app --reload`

### 4. Test the API

Open the FastAPI Swagger UI in a browser and use the `/predict` endpoint to submit a customer record.

The `/health` endpoint can be used to verify service availability.

## Production Scope

This project intentionally focuses on demonstrating a **production-oriented ML lifecycle** rather than claiming to be a fully production-ready enterprise deployment.

### Implemented

- Data ingestion
- Data-quality validation
- Feature engineering
- Shared preprocessing
- Model training
- Baseline vs candidate evaluation
- Model promotion rule
- Model persistence
- FastAPI online inference
- Health endpoint
- Logging
- Automated testing
- API latency measurement
- Lightweight feature drift monitoring
- Retraining strategy
- Incident-response approach

### Potential Future Enhancements

For a larger enterprise production environment, the system could be extended with:

- Docker containerization
- Azure/AWS/GCP deployment
- CI/CD automation
- Automated model retraining
- Advanced drift detection
- Model registry and version management
- Prometheus/Grafana monitoring
- Centralized logging and alerting
- Authentication and API security
- Batch inference
- Streaming inference
- BI dashboards
- Automated incident management

## Key Learning Outcomes

This project demonstrates the complete lifecycle of a machine learning solution:

<pre>
Business Problem
      ↓
Data Ingestion
      ↓
Data Preparation
      ↓
Feature Engineering
      ↓
Model Development
      ↓
Model Evaluation
      ↓
Model Promotion
      ↓
Model Persistence
      ↓
API Deployment
      ↓
Testing
      ↓
Monitoring
      ↓
Retraining Strategy
</pre>

The primary focus was not only on building a predictive model, but on designing the surrounding engineering components required to operate an ML solution as a service.

## Project Context

This project was developed as part of the **M.Sc. Data Science & Artificial Intelligence** program at BITS Pilani.

The assignment focuses on designing and executing a mini production ML system, with emphasis on machine learning model engineering, deployment, testing, monitoring, and production considerations.

## Author

**Sunita Baloda**

Technology, Data & AI Transformation Leader | BFSI | AI/ML | Data & Analytics

M.Sc. Data Science & Artificial Intelligence
