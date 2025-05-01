import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

# Style untuk plot yang lebih menarik
plt.style.use('seaborn-v0_8-darkgrid')
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# 1. Membuat data simulasi untuk demo
np.random.seed(42)
n_samples = 1000

# Data normal untuk fitur kelembapan, kualitas udara, CO, dan suhu
humidity = np.random.normal(65, 10, n_samples)
air_quality = np.random.normal(50, 15, n_samples) 
co_level = np.random.normal(5, 1.5, n_samples)
temperature = np.random.normal(25, 5, n_samples)

# Tambahkan beberapa anomali
anomaly_indices = np.random.choice(range(n_samples), size=int(n_samples * 0.05), replace=False)
for idx in anomaly_indices:
    # Membuat anomali dengan nilai yang jauh dari rata-rata
    humidity[idx] = np.random.choice([np.random.uniform(0, 30), np.random.uniform(90, 100)])
    air_quality[idx] = np.random.choice([np.random.uniform(0, 10), np.random.uniform(90, 150)])
    co_level[idx] = np.random.choice([np.random.uniform(0, 1), np.random.uniform(12, 20)])
    temperature[idx] = np.random.choice([np.random.uniform(0, 10), np.random.uniform(40, 50)])

# Membuat dataframe
data = pd.DataFrame({
    'Kelembapan': humidity,
    'Kualitas_Udara': air_quality,
    'Karbon_Monoksida': co_level,
    'Suhu': temperature
})

# 2. Preprocessing Data
print("==== Data Overview ====")
print(data.describe())

# Deteksi outlier menggunakan IQR method (untuk preprocessing)
def detect_outliers_iqr(df):
    cleaned_df = df.copy()
    outliers_idx = set()
    
    for column in df.columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR  # 3 kali IQR untuk batas yang lebih longgar
        upper_bound = Q3 + 3 * IQR
        
        # Menyimpan indeks outlier
        column_outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)].index
        outliers_idx.update(column_outliers)
    
    print(f"Total outliers detected for preprocessing: {len(outliers_idx)}")
    return outliers_idx

# Deteksi dan simpan indeks outlier
preprocessing_outliers_idx = detect_outliers_iqr(data)

# Scaling data untuk model machine learning
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
scaled_df = pd.DataFrame(scaled_data, columns=data.columns)

# 3. Training Isolation Forest
print("\n==== Training Isolation Forest Model ====")
model = IsolationForest(
    n_estimators=100, 
    max_samples='auto',
    contamination=0.05,  # Perkiraan proporsi anomali dalam data
    random_state=42,
    verbose=0
)

# Fit model dengan data yang sudah di-scale
model.fit(scaled_data)

# 4. Prediksi
# Mendapatkan skor anomali dan mengubah prediksi (-1 untuk anomali, 1 untuk normal)
anomaly_scores = model.decision_function(scaled_data)
predictions = model.predict(scaled_data)
predictions_binary = np.where(predictions == -1, 1, 0)  # 1 untuk anomali, 0 untuk normal

# Tambahkan hasil prediksi ke dataframe
data['Anomaly_Score'] = anomaly_scores
data['Is_Anomaly'] = predictions_binary

# Evaluasi model (asumsi bahwa kita tahu mana yang anomali berdasarkan indeks yang kita buat sebelumnya)
true_labels = np.zeros(n_samples)
true_labels[list(anomaly_indices)] = 1  # 1 untuk anomali, 0 untuk normal

print("\n==== Model Evaluation ====")
print(classification_report(true_labels, predictions_binary))

# Menghitung confusion matrix
cm = confusion_matrix(true_labels, predictions_binary)
print("\nConfusion Matrix:")
print(cm)

# 5. Visualisasi

# 5.1. Aplikasi PCA untuk visualisasi
print("\n==== Running PCA for Visualization ====")
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
pca_df['Is_Anomaly'] = predictions_binary

# Informasi tambahan untuk label
print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.2f}")

# Mempersiapkan figure untuk visualisasi
plt.figure(figsize=(20, 16))
grid = gridspec.GridSpec(3, 3)

# 5.2. Visualisasi distribusi fitur berdasarkan status anomali
ax1 = plt.subplot(grid[0, :])
features = data.columns[:4]  # Hanya fitur asli
for i, feature in enumerate(features):
    sns.kdeplot(
        data=data[data['Is_Anomaly'] == 0][feature], 
        label=f"{feature} (Normal)", 
        color=colors[i], 
        alpha=0.7,
        ax=ax1
    )
    sns.kdeplot(
        data=data[data['Is_Anomaly'] == 1][feature], 
        label=f"{feature} (Anomali)", 
        color=colors[i], 
        alpha=0.3, 
        linestyle='--',
        ax=ax1
    )

ax1.set_title('Distribusi Fitur Berdasarkan Status Anomali', fontsize=16)
ax1.legend(fontsize=10)

# 5.3. Visualisasi Anomaly Score
ax2 = plt.subplot(grid[1, 0])
threshold = -0.3  # Threshold yang digunakan oleh model
scores_df = pd.DataFrame({'index': range(len(anomaly_scores)), 'score': anomaly_scores})
ax2.scatter(
    scores_df['index'], 
    scores_df['score'],
    c=scores_df['score'] > threshold,
    cmap='viridis', 
    alpha=0.6
)
ax2.axhline(y=threshold, color='red', linestyle='-', alpha=0.5, label='Threshold')
ax2.set_title('Anomaly Scores', fontsize=16)
ax2.set_xlabel('Data Point Index')
ax2.set_ylabel('Anomaly Score')
ax2.legend()

# 5.4. PCA 2D scatter plot
ax3 = plt.subplot(grid[1, 1:])
scatter = ax3.scatter(
    pca_df['PC1'], 
    pca_df['PC2'], 
    c=pca_df['Is_Anomaly'],
    cmap='coolwarm', 
    alpha=0.7,
    s=50
)
ax3.set_title('PCA 2D Visualization of Anomalies', fontsize=16)
ax3.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%})')
ax3.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%})')
legend1 = ax3.legend(*scatter.legend_elements(), title="Status")
ax3.add_artist(legend1)

# 5.5. 3D PCA plot untuk visualisasi yang lebih menarik
# Buat 3D PCA
pca_3d = PCA(n_components=3)
pca_result_3d = pca_3d.fit_transform(scaled_data)
pca_df_3d = pd.DataFrame(data=pca_result_3d, columns=['PC1', 'PC2', 'PC3'])
pca_df_3d['Is_Anomaly'] = predictions_binary

ax4 = plt.subplot(grid[2, :], projection='3d')
scatter3d = ax4.scatter(
    pca_df_3d['PC1'],
    pca_df_3d['PC2'],
    pca_df_3d['PC3'],
    c=pca_df_3d['Is_Anomaly'],
    cmap='coolwarm',
    s=30,
    alpha=0.7
)
ax4.set_title('3D PCA Visualization of Anomalies', fontsize=16)
ax4.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.2%})')
ax4.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.2%})')
ax4.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.2%})')
ax4.legend(*scatter3d.legend_elements(), title="Status")
ax4.view_init(elev=30, azim=45)  # Atur sudut pandang

plt.tight_layout()
plt.savefig('anomaly_detection_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. Visualisasi tambahan: Relationship plot
# Visualisasi hubungan antara fitur dan anomali
plt.figure(figsize=(14, 12))
for i, feature in enumerate(features):
    plt.subplot(2, 2, i+1)
    sns.scatterplot(
        data=data, 
        x=feature, 
        y='Anomaly_Score',
        hue='Is_Anomaly',
        palette='coolwarm',
        alpha=0.7
    )
    plt.axhline(y=threshold, color='red', linestyle='--', alpha=0.5)
    plt.title(f'Relationship Between {feature} and Anomaly Score')
    
plt.tight_layout()
plt.savefig('feature_anomaly_relationship.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. Heatmap korelasi
plt.figure(figsize=(10, 8))
correlation = data.corr()
mask = np.triu(correlation)
sns.heatmap(
    correlation, 
    annot=True, 
    cmap='coolwarm', 
    mask=mask,
    vmin=-1, 
    vmax=1,
    linewidths=.5
)
plt.title('Correlation Matrix', fontsize=16)
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n==== Anomaly Detection Results ====")
print(f"Total data points: {len(data)}")
print(f"Number of anomalies detected: {sum(predictions_binary)}")
print(f"Percentage of anomalies: {sum(predictions_binary)/len(predictions_binary):.2%}")

# Menampilkan beberapa contoh anomali yang terdeteksi
print("\nSample of detected anomalies:")
print(data[data['Is_Anomaly'] == 1].head())

# Fungsi untuk memprediksi data baru
def predict_anomaly(new_data, model, scaler):
    """
    Prediksi apakah data baru adalah anomali atau tidak
    
    Parameters:
    -----------
    new_data : pd.DataFrame
        Data baru yang akan diprediksi
    model : sklearn model
        Model IsolationForest yang sudah dilatih
    scaler : sklearn StandardScaler
        Scaler yang sudah di-fit dengan data training
        
    Returns:
    --------
    predictions : array
        Hasil prediksi (-1 untuk anomali, 1 untuk normal)
    anomaly_scores : array
        Skor anomali untuk setiap data
    """
    # Scale data baru
    scaled_new_data = scaler.transform(new_data)
    
    # Prediksi
    predictions = model.predict(scaled_new_data)
    anomaly_scores = model.decision_function(scaled_new_data)
    
    return predictions, anomaly_scores

# Contoh penggunaan
print("\n==== Model Usage Example ====")
# Membuat contoh data baru (1 normal, 1 anomali)
new_data = pd.DataFrame({
    'Kelembapan': [65, 95],  # normal, anomali
    'Kualitas_Udara': [50, 5],  # normal, anomali
    'Karbon_Monoksida': [5, 15],  # normal, anomali
    'Suhu': [25, 45]  # normal, anomali
})
print("Data baru untuk prediksi:")
print(new_data)

# Prediksi
predictions, scores = predict_anomaly(new_data, model, scaler)
print("\nHasil prediksi:")
result_df = new_data.copy()
result_df['Prediction'] = ['Normal' if p == 1 else 'Anomali' for p in predictions]
result_df['Anomaly_Score'] = scores
print(result_df)

print("\n==== Completed! ====")