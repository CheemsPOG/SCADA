# ==============================================================================
# SCRIPT HOÀN CHỈNH (ĐÃ SỬA LỖI CUỐI CÙNG): LẮP RÁP HỆ THỐNG
# ==============================================================================
import joblib
import os
import pandas as pd
import numpy as np

# --- 1. IMPORT CÁC THÀNH PHẦN CƠ SỞ ---
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from scipy.stats import norm

# --- 2. CUNG CẤP CÁC "BẢN THIẾT KẾ" TÙY CHỈNH ---
# Joblib vẫn cần các định nghĩa này để dựng lại các bước bên trong pipeline
class StandardScaleTransform(BaseEstimator, TransformerMixin):
    def __init__(self, cols): self.cols = cols; self.scaler_ = None
    def fit(self, X, y=None): self.scaler_ = StandardScaler().fit(X.loc[:, self.cols]); return self
    def transform(self, X): X_copy = X.copy(); X_copy.loc[:, self.cols] = self.scaler_.transform(X_copy.loc[:, self.cols]); return X_copy

class DataFrameImputer(TransformerMixin, BaseEstimator):
    def __init__(self, median_cols=None, knn_cols=None): self.median_cols = median_cols; self.knn_cols = knn_cols
    def fit(self, X, y=None):
        self.median_imputer = SimpleImputer(strategy='median'); self.knn_imputer = KNNImputer()
        if hasattr(self,'median_cols') and self.median_cols: self.median_imputer.fit(X[self.median_cols])
        if hasattr(self,'knn_cols') and self.knn_cols: self.knn_imputer.fit(X[self.knn_cols])
        return self
    def transform(self, X):
        X_imputed = X.copy()
        if hasattr(self,'median_cols') and self.median_cols:
            X_median = pd.DataFrame(self.median_imputer.transform(X[self.median_cols]), columns=self.median_cols, index=X.index)
            X_imputed = X_imputed.drop(self.median_cols, axis=1)
            X_imputed = pd.concat([X_imputed, X_median], axis=1)
        if hasattr(self,'knn_cols') and self.knn_cols:
            X_knn = pd.DataFrame(self.knn_imputer.transform(X[self.knn_cols]), columns=self.knn_cols, index=X.index)
            X_imputed = X_imputed.drop(self.knn_cols, axis=1)
            X_imputed = pd.concat([X_imputed, X_knn], axis=1)
        return X_imputed

class DateExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, date_cols): self.date_cols = date_cols
    def fit(self, X, y=None): return self
    def transform(self, X):
        X_copy = X.copy()
        for col in self.date_cols:
            dates = pd.to_datetime(X_copy[col], format='%d %m %Y %H:%M')
            X_copy['Month'] = dates.dt.month; X_copy['Week'] = dates.dt.day // 7 + 1
            X_copy['Day'] = dates.dt.day; X_copy['Hour'] = dates.dt.hour + 1
            seasons_dict = {1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring', 6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Autumn', 10: 'Autumn', 11: 'Autumn', 12: 'Winter'}
            X_copy['Season'] = X_copy['Month'].map(seasons_dict)
        return X_copy

class OutlierThresholdTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column, q1=0.25, q3=0.75): self.column = column; self.q1 = q1; self.q3 = q3; self.thresholds_ = {}
    def fit(self, X, y=None):
        for col in self.column: Q1 = X[col].quantile(self.q1); Q3 = X[col].quantile(self.q3); iqr = Q3 - Q1; self.thresholds_[col] = (Q1 - 1.5 * iqr, Q3 + 1.5 * iqr)
        return self
    def transform(self, X):
        X_copy = X.copy()
        for col in self.column: low_limit, up_limit = self.thresholds_[col]; X_copy.loc[X_copy[col] < low_limit, col] = low_limit; X_copy.loc[X_copy[col] > up_limit, col] = up_limit
        return X_copy

class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None): self.columns = columns
    def fit(self, X, y=None): return self
    def transform(self, X): return X.drop(self.columns, axis=1) if self.columns else X

class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None): self.columns = columns; self.unique_values = {}
    def fit(self, X, y=None): self.unique_values = {col: X[col].unique() for col in self.columns}; return self
    def transform(self, X):
        X_transformed = X.copy()
        for col in self.columns:
            for value in self.unique_values[col]:
                X_transformed[f"{col}_{value}"] = (X_transformed[col] == value).astype(int)
        return X_transformed.drop(columns=self.columns)

# Định nghĩa "Siêu mô hình" (không thay đổi)
class AnomalyDetector(BaseEstimator, RegressorMixin):
    def __init__(self, preprocessing_pipeline, regression_model):
        self.preprocessing_pipeline = preprocessing_pipeline; self.regression_model = regression_model
        self.gmm = GaussianMixture(n_components=2, random_state=42); self.learned_threshold_ = None
    def fit(self, X_calib, y_calib):
        X_processed = self.preprocessing_pipeline.transform(X_calib)
        predictions = self.regression_model.predict(X_processed)
        errors = np.abs(y_calib - predictions).values.reshape(-1, 1)
        self.gmm.fit(errors)
        means = self.gmm.means_.flatten(); covs = self.gmm.covariances_.flatten(); weights = self.gmm.weights_.flatten()
        sorted_indices = np.argsort(means); normal_idx, anomaly_idx = sorted_indices[0], sorted_indices[1]
        x_space = np.linspace(errors.min(), errors.max(), 1000)
        pdf_normal = norm.pdf(x_space, means[normal_idx], np.sqrt(covs[normal_idx])) * weights[normal_idx]
        pdf_anomaly = norm.pdf(x_space, means[anomaly_idx], np.sqrt(covs[anomaly_idx])) * weights[anomaly_idx]
        intersection_idx = np.argwhere(np.diff(np.sign(pdf_normal - pdf_anomaly))).flatten()
        self.learned_threshold_ = x_space[intersection_idx[0]] if len(intersection_idx) > 0 else np.quantile(errors, 0.95)
        print(f"=> Đã học xong ngưỡng bất thường: {self.learned_threshold_:.2f} kW"); return self
    def predict_anomaly(self, X_new, y_actual):
        X_processed = self.preprocessing_pipeline.transform(X_new)
        y_predicted = self.regression_model.predict(X_processed)
        errors = np.abs(y_actual - y_predicted)
        return (errors > self.learned_threshold_).astype(int), y_predicted, errors

# --- 4. HÀM CHÍNH ĐỂ THỰC THI ---
def main():
    """Hàm chính thực thi toàn bộ quy trình lắp ráp."""
    print("--- Bắt đầu quy trình ---")
    print("Đang tải dữ liệu gốc để tạo tập test...")
    try:
        df_full = pd.read_csv('data/T1.csv')
    except FileNotFoundError:
        print("LỖI: Không tìm thấy tệp 'data/T1.csv'. Vui lòng đảm bảo tệp dữ liệu tồn tại.")
        return
    df_full.loc[df_full['LV ActivePower (kW)'] < 0, 'LV ActivePower (kW)'] = 0
    _, df_copy_test = train_test_split(df_full, test_size=0.2, random_state=42)
    X_test = df_copy_test.drop(columns=['LV ActivePower (kW)'])
    y_test = df_copy_test['LV ActivePower (kW)']
    print("=> Đã tạo xong tập test.")

    print("\n--- Bắt đầu quy trình lắp ráp từ các tệp .pkl ---")
    pipeline_path = 'one_hot_pipeline.pkl'
    regressor_path = 'one_hot_model.pkl'

    if not os.path.exists(pipeline_path) or not os.path.exists(regressor_path):
        print("LỖI: Không tìm thấy tệp .pkl. Vui lòng chạy notebook huấn luyện chính để tạo ra chúng.")
        return

    print("1/3: Đang tải các linh kiện đã được huấn luyện...")
    # Tải trực tiếp đối tượng Pipeline, không cần lớp FullPipeline1 nữa
    loaded_pipeline = joblib.load(pipeline_path)
    loaded_regressor = joblib.load(regressor_path)
    print("=> Tải thành công!")

    print("\n2/3: Đang lắp ráp và hiệu chỉnh hệ thống...")
    assembled_detector = AnomalyDetector(preprocessing_pipeline=loaded_pipeline, regression_model=loaded_regressor)
    assembled_detector.fit(X_test, y_test)
    print("=> Hiệu chỉnh ngưỡng hoàn tất.")

    print("\n3/3: Đang lưu sản phẩm cuối cùng...")
    final_model_path = 'full_anomaly_detector_final.pkl'
    joblib.dump(assembled_detector, final_model_path)
    print(f"\n=> ĐÃ LẮP RÁP VÀ LƯU THÀNH CÔNG HỆ THỐNG VÀO TỆP '{final_model_path}'")
    
    print("\n--- QUY TRÌNH HOÀN TẤT! ---")

if __name__ == "__main__":
    main()