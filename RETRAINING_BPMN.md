# Automated Model Retraining BPMN Diagram

## Process Description
This diagram illustrates the automated retraining and model replacement lifecycle for the breast cancer classification model in production, including trigger events, training steps, and stopping criteria.

## Process Diagram (Mermaid)

```mermaid
flowchart TD
    subgraph Triggers [1. Retraining Triggers]
        T1(["Data Drift: PSI > 0.2"])
        T2(["Metric Degradation: Accuracy < 95%"])
        T3(["Scheduled: Daily at 02:00"])
        T4(["New Data Volume: > 1000 samples"])
        T5(["Manual Trigger: Data Scientist"])
    end

    T1 & T2 & T3 & T4 & T5 --> Start(["Start Retraining Process"])

    subgraph Process [2. Retraining and Validation Process]
        Start --> Collect["Collect New and Historical Data"]
        Collect --> DataCheck{"Is Data Sufficient and Valid?"}
        DataCheck -->|No| StopData(["Stop: Data Quality Failure"])
        
        DataCheck -->|Yes| Preprocess["Preprocessing: IQR, StandardScaler, PCA"]
        Preprocess --> InitCounter["Reset Attempt Counter: Attempt = 1"]
        InitCounter --> Train["Train Models: Perceptron, KNN, DT, NB"]
        Train --> Evaluate["Evaluate on Test Dataset"]
        
        Evaluate --> Compare{"Metrics Better Than Production Model? Accuracy > 95%"}
    end

    subgraph Stopping [3. Stopping Criteria & Deployment]
        Compare -->|Yes| Register["Register New Version in Model Registry"]
        Register --> Shadow["Shadow / A/B Testing"]
        Shadow --> ABCheck{"Successful on Live Traffic?"}
        
        ABCheck -->|Yes| Deploy["Replace Model in Production"]
        Deploy --> StopSuccess(["Success Stop: Model Updated"])
        
        ABCheck -->|No| Rollback["Rollback to Previous Version"]
        Rollback --> AlertFail["Notify ML Engineer"]
        AlertFail --> StopFail(["Stop: Changes Rolled Back"])

        Compare -->|No| CheckAttempts{"Max Attempts Reached? Attempt >= 3"}
        CheckAttempts -->|No| Tune["Auto Hyperparameter Tuning: Attempt + 1"]
        Tune --> Train
        
        CheckAttempts -->|Yes| AlertDS["Alert Data Scientist: Manual Review"]
        AlertDS --> StopLimit(["Stop: Max Attempts Reached"])
    end

    style Triggers fill:#f0f8ff,stroke:#4682b4,stroke-width:2px
    style Process fill:#f5fffa,stroke:#2e8b57,stroke-width:2px
    style Stopping fill:#fffaf0,stroke:#d2691e,stroke-width:2px
```

## Key Diagram Components

### 1. Triggers
* **Data Drift:** Population Stability Index (`PSI > 0.2`).
* **Degradation:** Drop in accuracy (`Accuracy < 95%`) or recall (`Recall < 95%`).
* **Scheduled:** Cron-based scheduled execution (daily at 02:00).
* **Data Volume:** Accumulation of more than 1,000 newly labeled samples.
* **Manual:** On-demand execution triggered by a specialist.

### 2. Stopping Criteria
The pipeline automatically terminates under one of the following conditions:
1. **Successful Update (Success):** The challenger model outperforms the production model, passes A/B testing, and is deployed.
2. **Max Attempts Reached:** 3 retraining iterations with hyperparameter tuning fail to improve metrics (escalated to human review).
3. **Data Quality Failure:** Critical anomalies or insufficient sample sizes are detected prior to training.
4. **A/B Test Failure (Rollback):** Metric degradation occurs on live traffic, triggering an automatic rollback to the stable version.
