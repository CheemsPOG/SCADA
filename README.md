
# Wind Turbine Fault Detection using SCADA Sensor Data

This project focuses on analyzing time-series SCADA (Supervisory Control and Data Acquisition) sensor data collected from wind turbines to predict operational faults and performance anomalies. It demonstrates end-to-end data science and machine learning practices including data wrangling, exploratory analysis, feature engineering, model training, and evaluation, with a focus on industrial IoT data and real-time system monitoring.

## 🔍 Problem Statement

Wind turbines generate large volumes of operational data from sensors embedded in the system. Identifying faults in advance can reduce downtime and optimize maintenance. This project aims to classify potential fault conditions using ML models trained on historical SCADA data.

## 📁 Dataset

- **Source**: [Kaggle - Wind Turbine SCADA Dataset](https://www.kaggle.com/code/ahmedmaheralgohary/wind-turbine-eda-and-modeling)
- **Size**: ~500,000 records
- **Features**:
  - Wind Speed (m/s)
  - Generator RPM
  - Power Output (kW)
  - Ambient Temperature
  - Rotor Torque
  - Fault Codes (target)
  - Timestamps and other derived features

## ⚙️ Tools & Technologies

- **Languages**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn
- **Modeling**: Classification models (Random Forest, XGBoost)
- **Deployment Prep**: AWS EC2 (simulated), GCP Colab
- **Domain**: Industrial IoT, Predictive Maintenance

## 🔧 Methodology

1. **Exploratory Data Analysis (EDA)**:
   - Trend and distribution analysis of sensor readings
   - Anomaly visualization and correlation heatmaps

2. **Feature Engineering**:
   - Derived features like rolling averages, power ratios
   - Label encoding for fault status

3. **Modeling**:
   - Fault classification using Random Forest and XGBoost
   - Hyperparameter tuning via cross-validation
   - Evaluation using F1-score, ROC AUC, and confusion matrix

4. **Visualization**:
   - Real-time turbine metrics
   - Fault trend graphs and sensor drift patterns

5. **Deployment Simulation**:
   - Data preprocessing pipeline built for cloud deployment
   - Simulated execution on AWS/GCP environments

## 📈 Results

- Achieved **~90% accuracy** on test set
- Reduced false positives for normal turbine behavior
- Provided interpretable fault predictions for proactive maintenance

## 📌 Key Takeaways

- Handling of large-scale, semi-structured IoT data
- Application of machine learning for real-world industrial use cases
- Demonstrated ability to extract business value from raw sensor data

## 🧠 Future Work

- Integration with cloud-hosted dashboards for real-time monitoring
- Support for multi-class fault prediction and root cause analysis
- Deployment via FastAPI + Docker as a microservice for edge devices

## 📎 Relevant Skills

- Time-Series Analysis, Fault Detection
- ML Modeling (XGBoost, RF), Feature Engineering
- Python, SQL, AWS/GCP, Data Visualization, SCADA Data
