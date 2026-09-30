# Model Lifecycle BPMN Diagram

## Breast Cancer Detection Project

This document shows the full lifecycle of the machine learning model from business requirements to archival.

## Process Diagram

```mermaid
flowchart TD
    Start([Start: Business Requirement]) --> Define[Define Goal]
    Define --> Feasibility{Economic Feasibility?}
    Feasibility -->|No| Reject([Reject Project])
    Feasibility -->|Yes| Collect[Collect Data]
    Collect --> Preprocess[Preprocess]
    Preprocess --> Split[Split Data]
    Split --> Train[Train Models]
    Train --> Evaluate[Evaluate]
    Evaluate --> Compare{Accuracy > 95%?}
    Compare -->|No| Experiment[Tune and Experiment]
    Experiment --> Train
    Compare -->|Yes| Validate{Validation?}
    Validate -->|No| Experiment
    Validate -->|Yes| Register[Register Model]
    Register --> Deploy[Deploy]
    Deploy --> ABTest[A/B Test]
    ABTest --> ABDecision{New Model Better?}
    ABDecision -->|No| Rollback[Rollback to Old Model]
    ABDecision -->|Yes| Production[Production]
    Rollback --> Monitor
    Production --> Monitor[Monitor Model]
    Monitor --> Drift{Drift Detected?}
    Drift -->|No| Production
    Drift -->|Yes| Alert[Alert]
    Alert --> Retrain[Retrain]
    Retrain --> ReEvaluate[Re-evaluate]
    ReEvaluate --> Better{Quality Improved?}
    Better -->|No| HumanReview[Human Review]
    HumanReview --> Experiment
    Better -->|Yes| Redeploy[Redeploy New Version]
    Redeploy --> Production
    Production --> Archive{Model Deprecated?}
    Archive -->|No| Production
    Archive -->|Yes| ArchiveModel[Archive Model]
    ArchiveModel --> End([End: Model Archived])

Stages
1. Business Requirements

    Define goal: breast cancer diagnosis

    Assess economic feasibility

    Set KPIs (accuracy > 95%, recall > 95%)

2. Data Collection

    Source: Wisconsin Breast Cancer Dataset

    Size: 569 samples, 30 features

    Classes: Benign (357), Malignant (212)

3. Preprocessing

    Check missing values

    Remove outliers with IQR

    Scale with StandardScaler

    Optional: PCA and Fuzzy C-Means

4. Model Training

    Train 4 models: Decision Tree, Naive Bayes, Perceptron, KNN

    Hyperparameter tuning

    Cross-validation

5. Evaluation

    Metrics: Accuracy, Precision, Recall, F1, AUC-ROC

    Confusion matrix analysis

    Best model: Perceptron with 97.37% accuracy

6. Validation and Approval

    Review by business stakeholders

    Create Model Card

    Compliance check

7. Model Registration

    Save model weights (pkl.)

    Log metadata (version, date, metrics)

    Register in Model Registry

8. Deployment

    API with FastAPI

    Containerization with Docker

    Orchestration with Kubernetes (optional)

9. A/B Testing

    Compare new model with old

    Split traffic (50/50)

    Monitor real metrics

10. Production Monitoring

    Detect data drift

    Detect concept drift

    Track business metrics

    Monitor SLI/SLO

11. Retraining (Automated)

    Trigger: drift detected

    Pipeline: Collect → Preprocess → Train → Evaluate

    Auto-deploy if improved

12. Archival

    When: model deprecated

    What: weights + metadata + data references

    Why: legal compliance, rollback capability

Roles

    Data Scientist: training, evaluation, retraining

    ML Engineer: registration, deployment, monitoring

    Business Owner: requirements, validation

    Compliance: regulatory check, archival

Automation Triggers

    Data Drift (PSI > 0.2): alert + retrain

    Performance drop (accuracy < 95%): retrain

    Schedule (daily 02:00): retrain if new data

    New data volume (> 1000 samples): retrain

    Manual: data scientist request

Tools

    Data collection: pandas

    Preprocessing: scikit-learn, numpy

    Training: scikit-learn, joblib

    Evaluation: scikit-learn metrics

    Deployment: FastAPI, uvicorn, Docker

    Monitoring: Prometheus, Grafana

    CI/CD: GitHub Actions, pre-commit

Model Versioning

    Version 1.0.0 (1 September 2026): first version, 92% accuracy

    Version 2.0.0 (29 September 2026): Perceptron, 97.37% accuracy (current)

References

    BPMN 2.0: https://www.omg.org/spec/BPMN/2.0/

    Google Model Cards: https://arxiv.org/abs/1810.03993

    MLOps: https://ml-ops.org/
