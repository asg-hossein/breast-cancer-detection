# Model Card - Breast Cancer Classification

## Model Details

- **Developer:** Aliasghar Asgharzadeh
- **Date:** 29 September 2026
- **Version:** 2.0.0
- **Model Type:** Binary Classification (Perceptron)
- **Framework:** scikit-learn 1.7.2
- **Email:** asg.hossein@gmail.com

## About the Model

This model classifies breast cancer tumors as Benign or Malignant using 30 numerical features from FNA images. The main model is a Perceptron trained on the Wisconsin Breast Cancer Dataset. Three other models (Decision Tree, Naive Bayes, KNN) are available through the API for comparison.

## Intended Use

### Primary Use

- **Users:** Medical professionals, researchers, data scientists
- **Purpose:** Support preliminary breast cancer diagnosis
- **Role:** Decision support only, not a final diagnosis

### Out-of-Scope Use

- Not for definitive diagnosis without pathologist confirmation
- Not for data with different feature distributions
- Not for male or pediatric patients (dataset contains only female patients)
- Not for critical decisions without human oversight

## Factors

- 30 features from cell nuclei: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension (mean, SE, worst for each)
- Data source: Wisconsin Breast Cancer Dataset
- Sample count: 569 (357 Benign, 212 Malignant)
- Scaling: StandardScaler

## Metrics

### Performance

- Accuracy: 0.9737
- Precision: 0.9756
- Recall: 0.9524
- F1-Score: 0.9639
- AUC-ROC: Not available (Perceptron does not provide probabilities)

### Confusion Matrix

- True Negatives: 71
- False Positives: 1
- False Negatives: 2
- True Positives: 40

### Model Comparison

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| Decision Tree | 0.9298 | 0.9048 | 0.9048 | 0.9048 |
| Naive Bayes | 0.9211 | 0.9231 | 0.8571 | 0.8889 |
| Perceptron | 0.9737 | 0.9756 | 0.9524 | 0.9639 |
| KNN | 0.9561 | 0.9744 | 0.9048 | 0.9383 |

## Training Data

- **Name:** Wisconsin Breast Cancer Dataset
- **Source:** UCI Machine Learning Repository
- **Samples:** 569
- **Features:** 30
- **Target:** Binary (0 = Benign, 1 = Malignant)
- **Class distribution:** 357 Benign (62.7%), 212 Malignant (37.3%)

### Preprocessing

- Missing values: none found
- Outlier removal: IQR method (optional)
- Scaling: StandardScaler
- PCA: not applied in basic mode

### Data Split

- Training: 455 samples (80%)
- Test: 114 samples (20%)
- Stratified: yes

## Limitations

1. Trained on a single dataset; may not generalize to other populations
2. Requires all 30 features
3. Class distribution is imbalanced
4. Perceptron does not provide probability estimates
5. May degrade over time due to data drift
6. Only classifies two classes
7. Does not consider patient history

## Ethical Considerations

- **Gender bias:** Dataset contains only female patients
- **Age bias:** Age information not available
- **Racial bias:** Demographic information not available
- **Privacy:** Public anonymized dataset
- **Transparency:** Model architecture is documented
- **Responsibility:** Final decision is made by a medical professional

## Recommendations

1. Always validate model output with a medical professional
2. Retrain if data distribution changes
3. Test on local data before deployment
4. Monitor performance continuously
5. Use ensemble methods for critical applications
6. Document all changes for audit

### Code Repository

- GitHub: https://github.com/asg-hossein/breast-cancer-detection
- API Documentation: http://localhost:8000/docs

## References

- Google Model Cards: https://arxiv.org/abs/1810.03993
- UCI Dataset: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
- scikit-learn: https://scikit-learn.org/

## Author

Aliasghar Asgharzadeh — 29 September 2026
