# Time Series Forecasting for Gold Prices using Azure Machine Learning Service

This step-by-step guide walks you through building, deploying, and monitoring a time series forecasting model for gold prices using **Azure Machine Learning Service**. The workflow is designed for data scientists/ML engineers familiar with Python, but new to Azure ML. All steps are performed within Azure ML (Portal, CLI, and Notebooks), using serverless and managed services to minimize infrastructure management.

---

## 1. Introduction & Setup

### **Project Goal**

Build, deploy, and monitor a time series model to forecast gold prices using historical data and Azure ML's managed services.

### **Azure ML Overview**

Key components:

- **Workspace**: Central hub for all ML assets
- **Compute**: Managed/serverless compute for training and inference
- **Datastores**: Secure storage abstraction for data
- **Models**: Registry for versioned ML models
- **Endpoints**: Managed deployment targets for real-time and batch inference
- **Pipelines**: Orchestrated workflows for repeatable ML processes

### **Prerequisites**

- **Active Azure Subscription**
- **Azure CLI** installed and configured (`az login`)
- **Python 3.8+** and Azure ML SDK v2 (`pip install azure-ai-ml azure-identity`)
- **Resource Group**: Create with
  ```bash
  az group create --name gold-ml-rg --location eastus
  ```

---

## 2. Step 1: Prepare the Azure ML Workspace & Data

### **Create Workspace**

**A. Azure Portal**

1. Go to [Azure Portal](https://portal.azure.com)
2. "Create a resource" → "Machine Learning"
3. Fill in: Subscription, Resource Group (`gold-ml-rg`), Workspace Name, Region
4. Click "Review + Create"

**B. Azure CLI**

```bash
az ml workspace create --name gold-ml-ws --resource-group gold-ml-rg --location eastus
```

**C. Python SDK (Notebook Cell)**

```python
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Workspace
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="<your-subscription-id>",
    resource_group_name="gold-ml-rg"
)

workspace = Workspace(
    name="gold-ml-ws",
    location="eastus",
    description="Gold price forecasting workspace"
)
ml_client.workspaces.begin_create(workspace).result()
```

### **Get Gold Price Data**

If you already have a CSV (e.g. `gold_price_50_years_1975_2025.csv`), skip download. Otherwise, use:

```python
import yfinance as yf
import pandas as pd

gold = yf.download("GC=F", start="1975-01-01", end="2025-10-16", interval="1d")
gold.reset_index(inplace=True)
gold[['Date', 'Close']].to_csv("gold_prices.csv", index=False)
```

### **Create Data Asset**

**A. Python SDK (Notebook Cell)**

```python
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

gold_data = Data(
    name="gold-prices-data",
    version="1",
    description="Gold prices 1975-2025",
    path="gold_price_50_years_1975_2025.csv",
    type=AssetTypes.URI_FILE,
    tags={"source": "uploaded-csv", "target": "close"}
)
ml_client.data.create_or_update(gold_data)
```

**B. Azure CLI**
Create `data-asset.yml`:

```yaml
name: gold-prices-data
version: 1
description: Gold prices 1975-2025
type: uri_file
path: gold_price_50_years_1975_2025.csv
tags:
  source: uploaded-csv
  target: close
```

Upload:

```bash
az ml data create --file data-asset.yml --resource-group gold-ml-rg --workspace-name gold-ml-ws
```

### **Data Exploration (Notebook Cell)**

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("gold_price_50_years_1975_2025.csv")
df['date'] = pd.to_datetime(df['date'])
df.sort_values('date', inplace=True)
print(df.info())
print(df.head())

plt.figure(figsize=(12,5))
sns.lineplot(data=df, x='date', y='close')
plt.title('Gold Price Over Time')
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.show()
print(df.isnull().sum())
print(df.describe())
```

---

## 3. Step 2: Train a Model with an AutoML Job

### **Why AutoML for Time Series?**

- Automated feature engineering (lags, rolling windows)
- Hyperparameter tuning
- Model selection
- Managed/serverless compute

### **Configure the AutoML Job (Notebook Cell)**

```python
from azure.ai.ml import automl
from azure.ai.ml import Input
from azure.ai.ml.entities import ResourceConfiguration

forecast_job = automl.forecasting(
    experiment_name="gold-price-forecasting",
    training_data=Input(type="uri_file", path="azureml:gold-prices-data:1"),
    target_column_name="close",
    primary_metric="normalized_root_mean_squared_error",
    forecasting_settings={
        "time_column_name": "date",
        "forecast_horizon": 30,
        "frequency": "D",
        "target_lags": [1,2,3,5,7,14],
        "target_rolling_window_size": 7
    },
    n_cross_validations="auto",
    validation_data_size=0.2,
    test_data_size=0.1,
    enable_early_stopping=True,
    timeout_minutes=120,
    max_trials=30
)
forecast_job.resources = ResourceConfiguration(instance_type="Standard_DS3_v2", instance_count=1)
submitted_job = ml_client.jobs.create_or_update(forecast_job)
print(f"AutoML job submitted: {submitted_job.name}")
print(submitted_job.studio_url)
```

### **Monitor Job & Retrieve Best Model (Notebook Cell)**

```python
completed = ml_client.jobs.get(submitted_job.name)
print(f"Job Status: {completed.status}")
if completed.status == "Completed":
    best_model = list(ml_client.models.list(name=completed.name))[0]
    print(f"Model: {best_model.name}, Version: {best_model.version}")
```

---

## 4. Step 3: Test the Model with a Batch Pipeline

### **Register the Model**

**A. Python SDK (Notebook Cell)**

```python
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes

model = Model(
    name="gold-forecasting-model",
    version="1",
    description="Best AutoML model for gold price forecasting",
    path=f"azureml://jobs/{submitted_job.name}/outputs/artifacts/outputs/mlflow-model",
    type=AssetTypes.MLFLOW_MODEL,
    tags={"model_type": "forecasting", "target": "close"}
)
ml_client.models.create_or_update(model)
```

**B. Azure CLI**

```bash
az ml model create --name gold-forecasting-model --version 1 --path azureml://jobs/{job-id}/outputs/artifacts/outputs/mlflow-model --type mlflow_model --resource-group gold-ml-rg --workspace-name gold-ml-ws
```

### **Create Batch Inference Pipeline**

**A. Write batch_predict.py (Notebook Cell)**

```python
%%writefile batch_predict.py
import os, pandas as pd, mlflow, numpy as np

def init():
    global model
    model_path = os.environ.get("AZUREML_MODEL_DIR")
    model = mlflow.sklearn.load_model(model_path)

def run(mini_batch):
    results = []
    for path in mini_batch:
        data = pd.read_csv(path)
        preds = model.predict(data)
        out = pd.DataFrame({"date": data["date"], "predicted_price": preds})
        results.append(out)
    return pd.concat(results)
```

**B. Define and Run Pipeline (Notebook Cell)**

```python
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import Input

@pipeline(default_compute="serverless")
def batch_pipeline(input_data, model_path):
    from azure.ai.ml import command
    return command(
        name="gold_price_batch_predict",
        code="./",
        command="python batch_predict.py",
        inputs={"input_data": input_data, "model_path": model_path}
    )

pipeline_job = batch_pipeline(
    input_data=Input(path="azureml:gold-prices-data:1"),
    model_path=Input(path=f"azureml:gold-forecasting-model:1")
)
pipeline_run = ml_client.jobs.create_or_update(pipeline_job)
print(f"Pipeline submitted: {pipeline_run.name}")
```

---

## 5. Step 4: Deploy the Model to a Managed Endpoint

### **Online vs. Batch Endpoints**

- **Online**: Real-time, low-latency predictions (serverless)
- **Batch**: Asynchronous, large-scale scoring

### **Create Scoring Script (score.py)**

```python
%%writefile score.py
import json, os, pandas as pd, mlflow

def init():
    global model
    model_path = os.environ.get("AZUREML_MODEL_DIR")
    model = mlflow.sklearn.load_model(model_path)

def run(raw_data):
    data = pd.DataFrame(json.loads(raw_data)["data"])
    preds = model.predict(data)
    return json.dumps({"predictions": preds.tolist()})
```

### **Define Environment (conda.yml)**

```yaml
name: gold-forecasting-online-env
dependencies:
  - python=3.9
  - pip
  - numpy>=1.21.0
  - pandas>=1.3.0
  - scikit-learn>=1.0.0
  - pip:
      - azureml-inference-server-http>=0.7.0
      - mlflow>=1.20.0
      - azureml-defaults>=1.38.0
channels:
  - conda-forge
  - defaults
```

### **Deploy the Model**

**A. Azure CLI**
Create `online-endpoint.yml` and `online-deployment.yml` as in previous steps, then:

```bash
az ml online-endpoint create --file online-endpoint.yml --resource-group gold-ml-rg --workspace-name gold-ml-ws
az ml online-deployment create --file online-deployment.yml --all-traffic --resource-group gold-ml-rg --workspace-name gold-ml-ws
```

**B. Python SDK (Notebook Cell)**

```python
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment, CodeConfiguration, OnlineRequestSettings

endpoint = ManagedOnlineEndpoint(
    name="gold-forecasting-endpoint",
    auth_mode="key",
    description="Real-time gold price forecast"
)
ml_client.begin_create_or_update(endpoint).result()

deployment = ManagedOnlineDeployment(
    name="gold-forecasting-deploy",
    endpoint_name="gold-forecasting-endpoint",
    model="azureml:gold-forecasting-model:1",
    code_configuration=CodeConfiguration(code="./", scoring_script="score.py"),
    environment="AzureML-sklearn-1.0-ubuntu20.04-py38-cpu:1",
    instance_type="Standard_DS2_v2",
    instance_count=1,
    request_settings=OnlineRequestSettings(request_timeout_ms=60000)
)
ml_client.begin_create_or_update(deployment).result()
endpoint.traffic = {"gold-forecasting-deploy": 100}
ml_client.begin_create_or_update(endpoint).result()
```

### **Test the Live Endpoint (Notebook Cell)**

```python
import requests, json

ep = ml_client.online_endpoints.get("gold-forecasting-endpoint")
keys = ml_client.online_endpoints.get_keys("gold-forecasting-endpoint")

sample = {"data": [{"open": 3050.9, "high": 3061.2, "low":3050.0, "close":3056.1}]}
headers = {
    "Authorization": f"Bearer {keys.primary_key}",
    "Content-Type": "application/json"
}
response = requests.post(ep.scoring_uri, data=json.dumps(sample), headers=headers)
print(response.json())
```

---

## 6. Step 5 (Extra): Evaluate with the Responsible AI Dashboard

### **Introduction to Responsible AI**

The RAI Dashboard in Azure ML helps ensure model fairness, transparency, and accountability. For time series, it provides:

- **Model Interpretability**: Feature importance
- **Error Analysis**: Identify periods/cohorts with high error

### **Generate the Dashboard (Notebook Cell)**

```python
from azure.ai.ml import automl
from azure.ai.ml import Input

rai_job = automl.rai(
    experiment_name="rai-gold-forecasting",
    model_input="azureml:gold-forecasting-model:1",
    train_data=Input(path="azureml:gold-prices-data:1"),
    target_column_name="close",
    compute="serverless"
)
ml_client.jobs.create_or_update(rai_job)
```

### **Interpret the Results**

- In Azure ML Studio, go to **Models → gold-forecasting-model → Responsible AI**
- Explore **Feature Importance** and **Error Analysis** tabs
- Use cohort selection to filter by time periods or price ranges
- Use insights to validate, improve, and communicate model behavior

---

## **Summary**

This guide covers:

- Workspace and data setup (Portal, CLI, SDK)
- Data asset registration and EDA
- AutoML time series training (serverless)
- Model registration and batch pipeline
- Managed online endpoint deployment and testing
- Responsible AI dashboard for interpretability and error analysis

All steps are performed inside Azure Machine Learning Service, using managed and serverless resources for minimal infrastructure management.

If you share your course or grade level, I can further tailor the explanations or code examples!

[1](https://learn.microsoft.com/en-us/azure/machine-learning/tutorial-automated-ml-forecast?view=azureml-api-2)
[2](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-auto-train-forecast?view=azureml-api-2)
[3](https://learn.microsoft.com/en-us/AZURE/machine-learning/how-to-auto-train-forecast?view=azureml-api-1)
[4](https://docs.azure.cn/en-us/machine-learning/concept-automl-forecasting-at-scale?view=azureml-api-2)
[5](https://learn.microsoft.com/en-us/shows/ai-show/time-series-forecasting-with-azure-machine-learning)
[6](https://www.alphabold.com/train-a-forecasting-model-using-azure-machine-learning-service/)
[7](https://www.kdnuggets.com/the-lazy-data-scientists-guide-to-time-series-forecasting)
[8](https://azurelessons.com/microsoft-azure-machine-learning-tutorial/)
