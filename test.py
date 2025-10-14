import pandas as pd
import numpy as np
import joblib

# --- BƯỚC 0: ĐỊNH NGHĨA LẠI TẤT CẢ CÁC "BẢN THIẾT KẾ" (CLASSES) ---
# Đoạn này BẮT BUỘC phải có để joblib.load() có thể tái tạo lại pipeline

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline

# Copy từ Cell 45
class StandardScaleTransform(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
        self.scaler_ = None
    def fit(self, X, y=None):
        self.scaler_ = StandardScaler().fit(X.loc[:, self.cols])
        return self
    def transform(self, X):
        X_copy = X.copy()
        X_copy.loc[:, self.cols] = self.scaler_.transform(X_copy.loc[:, self.cols])
        return X_copy
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

# Copy từ Cell 46
class DataFrameImputer(TransformerMixin, BaseEstimator):
    def __init__(self, median_cols=None, knn_cols=None):
        self.median_cols = median_cols
        self.knn_cols = knn_cols
    def fit(self, X, y=None):
        self.median_imputer = SimpleImputer(strategy='median')
        self.knn_imputer = KNNImputer()
        if self.median_cols is not None:
            self.median_imputer.fit(X[self.median_cols])
        if self.knn_cols is not None:
            self.knn_imputer.fit(X[self.knn_cols])
        return self
    def transform(self, X):
        X_imputed = X.copy()
        if self.median_cols is not None:
            X_median = pd.DataFrame(self.median_imputer.transform(X[self.median_cols]), columns=self.median_cols, index=X.index)
            X_imputed = pd.concat([X_imputed.drop(self.median_cols, axis=1), X_median], axis=1)
        if self.knn_cols is not None:
            X_knn = pd.DataFrame(self.knn_imputer.transform(X[self.knn_cols]), columns=self.knn_cols, index=X.index)
            X_imputed = pd.concat([X_imputed.drop(self.knn_cols, axis=1), X_knn], axis=1)
        return X_imputed
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

# Copy từ Cell 47
class DateExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, date_cols):
        self.date_cols = date_cols
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_new = X.copy()
        for col in self.date_cols:
            X_new[col] = pd.to_datetime(X_new[col], format='%d %m %Y %H:%M')
            X_new['Month'] = X_new[col].dt.month
            X_new['Week'] = X_new[col].dt.day // 7 + 1
            X_new['Day'] = X_new[col].dt.day
            X_new['Hour'] = X_new[col].dt.hour + 1
            seasons_dict = {1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring', 6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Autumn', 10: 'Autumn', 11: 'Autumn', 12: 'Winter'}
            X_new['Season'] = X_new['Month'].map(seasons_dict)
        return X_new
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

# Copy từ Cell 48
class OutlierThresholdTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column, q1=0.25, q3=0.75):
        self.column = column
        self.q1 = q1
        self.q3 = q3
        self.thresholds_ = {}
    def fit(self, X, y=None):
        for col in self.column:
            Q1 = X[col].quantile(self.q1)
            Q3 = X[col].quantile(self.q3)
            iqr = Q3 - Q1
            up_limit = Q3 + 1.5 * iqr
            low_limit = Q1 - 1.5 * iqr
            self.thresholds_[col] = (low_limit, up_limit)
        return self
    def transform(self, X):
        X_copy = X.copy()
        for col in self.column:
            low_limit, up_limit = self.thresholds_[col]
            X_copy.loc[X_copy[col] < low_limit, col] = low_limit
            X_copy.loc[X_copy[col] > up_limit, col] = up_limit
        return X_copy

# Copy từ Cell 49
class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if self.columns is None:
            return X
        else:
            return X.drop(self.columns, axis=1)

# Copy từ Cell 50
class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
        self.unique_values = {}
    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in self.columns:
            self.unique_values[col] = X[col].unique()
        return self
    def transform(self, X):
        X_copy = X.copy()
        for col in self.columns:
            for value in self.unique_values[col]:
                X_copy[f"{col}_{value}"] = (X_copy[col] == value).astype(int)
        X_copy = X_copy.drop(columns=self.columns)
        return X_copy
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

# Copy từ Cell 52
class FullPipeline1:
    def __init__(self) :
        self.date_cols=['Date/Time']
        self.numerical_cols=['Wind Speed (m/s)', 'Theoretical_Power_Curve (KWh)', 'Wind Direction (°)','Week','Month','Hour','Day']
        self.MLE_cols=['Season']
        self.full_pipeline=Pipeline([
            ('extract_date',DateExtractor(date_cols=self.date_cols)),
            ('label_encode', CustomOneHotEncoder(columns=self.MLE_cols)),
            ('impute_num',DataFrameImputer(knn_cols=self.numerical_cols)),
            ('remove_outlier',OutlierThresholdTransformer(column=self.numerical_cols)),
            ('scale', StandardScaleTransform(cols=self.numerical_cols)),
            ('drop', DropColumnsTransformer(columns=self.date_cols)),
        ])
    def fit_transform(self, X_train):
        return self.full_pipeline.fit_transform(X_train)
    def transform(self, X_test):
        return self.full_pipeline.transform(X_test)


# --- BƯỚC 1: TẢI LẠI "BỘ NÃO" ĐÃ ĐƯỢC HUẤN LUYỆN ---
print("Đang tải pipeline tiền xử lý...")
preprocessing_pipeline = joblib.load('one_hot_pipeline.pkl')
print("Đang tải mô hình đã huấn luyện...")
model = joblib.load('one_hot_model.pkl')
print("Hệ thống đã sẵn sàng để dự đoán.\n")


# --- BƯỚC 2: CHUẨN BỊ DỮ LIỆU MỚI CẦN DỰ ĐOÁN ---
new_scada_data = {
    'Date/Time': ['15 01 2019 14:30', '15 01 2019 14:40'],
    'Wind Speed (m/s)': [8.5, 15.2],
    'Theoretical_Power_Curve (KWh)': [1800.5, 3600.0],
    'Wind Direction (°)': [210.5, 75.8]
}
new_df = pd.DataFrame(new_scada_data)
print("Dữ liệu mới cần dự đoán (dạng thô):")
print(new_df)


# --- BƯỚC 3: SỬ DỤNG PIPELINE ĐỂ XỬ LÝ DỮ LIỆU MỚI ---
print("\nĐang xử lý dữ liệu mới bằng pipeline đã lưu...")
processed_new_data = preprocessing_pipeline.transform(new_df)
print("Dữ liệu mới sau khi đã được tiền xử lý:")
print(processed_new_data.head())


# --- BƯỚC 4: SỬ DỤNG MÔ HÌNH ĐỂ ĐƯA RA DỰ ĐOÁN ---
print("\nĐang thực hiện dự đoán công suất...")
predicted_power = model.predict(processed_new_data)


# --- BƯỚC 5: HIỂN THỊ KẾT QUẢ ---
print("\n--- KẾT QUẢ DỰ ĐOÁN ---")
for i in range(len(new_df)):
    original_data = new_df.iloc[i]
    prediction = predicted_power[i]
    print(f"Cho dữ liệu đầu vào tại thời điểm {original_data['Date/Time']}:")
    print(f"  - Tốc độ gió: {original_data['Wind Speed (m/s)']} m/s")
    print(f"==> Công suất dự đoán (bình thường) là: {prediction:.2f} kW")