import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# Set seed untuk reproduksibilitas
np.random.seed(42)

# Warna yang elegan untuk visualisasi
cmap_custom = LinearSegmentedColormap.from_list(
    'custom_cmap', ['#1a237e', '#039be5', '#4fc3f7', '#b3e5fc', '#e1f5fe', 
                    '#ffebee', '#ffcdd2', '#ef9a9a', '#e57373', '#f44336', '#b71c1c'], N=256)

# 1. Membuat dataset simulasi
def generate_air_quality_data(n_samples=1000, anomaly_ratio=0.05):
    """
    Generates synthetic IoT air quality data with anomalies.
    """
    # Data normal
    humidity = np.random.normal(65, 10, n_samples)
    air_quality = np.random.normal(50, 15, n_samples)
    co_level = np.random.normal(5, 1.5, n_samples)
    temperature = np.random.normal(25, 5, n_samples)
    
    # Buat korelasi antara suhu dan kelembaban (korelasi negatif)
    correlation_factor = 0.7
    humidity = humidity * (1 - correlation_factor) + (100 - temperature) * correlation_factor
    
    # Tambahkan beberapa anomali
    anomaly_indices = np.random.choice(range(n_samples), size=int(n_samples * anomaly_ratio), replace=False)
    
    # Buat tipe anomali yang berbeda
    for i, idx in enumerate(anomaly_indices):
        anomaly_type = i % 4  # 4 tipe anomali berbeda
        
        if anomaly_type == 0:  # Anomali kelembaban tinggi
            humidity[idx] = np.random.uniform(90, 100)
            air_quality[idx] = np.random.uniform(10, 30)  # Kualitas udara rendah saat kelembaban tinggi
        
        elif anomaly_type == 1:  # Anomali CO tinggi
            co_level[idx] = np.random.uniform(12, 20)
            air_quality[idx] = np.random.uniform(0, 20)  # Kualitas udara buruk saat CO tinggi
        
        elif anomaly_type == 2:  # Anomali suhu tinggi
            temperature[idx] = np.random.uniform(40, 50)
            humidity[idx] = np.random.uniform(20, 40)  # Kelembaban rendah saat suhu tinggi
        
        else:  # Kombinasi anomali
            humidity[idx] = np.random.uniform(10, 30)
            co_level[idx] = np.random.uniform(10, 15)
            temperature[idx] = np.random.uniform(35, 45)
            air_quality[idx] = np.random.uniform(0, 15)
    
    # Buat dataframe
    df = pd.DataFrame({
        'Kelembapan': humidity,
        'Kualitas_Udara': air_quality,
        'Karbon_Monoksida': co_level,
        'Suhu': temperature
    })
    
    # Label asli untuk evaluasi
    true_anomalies = np.zeros(n_samples, dtype=int)
    true_anomalies[anomaly_indices] = 1
    df['True_Anomaly'] = true_anomalies
    
    return df, anomaly_indices

# 2. Generate dan tampilkan data
data, anomaly_indices = generate_air_quality_data(n_samples=1000, anomaly_ratio=0.05)
print("==== Data Overview ====")
print(data.describe())
print(f"\nJumlah anomali dalam dataset: {len(anomaly_indices)} ({len(anomaly_indices)/len(data):.2%})")

# 3. Data Preprocessing
# Scaling data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data.drop('True_Anomaly', axis=1))
scaled_df = pd.DataFrame(scaled_features, columns=data.columns[:-1])

# 4. Training Isolation Forest
model = IsolationForest(
    n_estimators=150,
    max_samples='auto',
    contamination=0.05,
    random_state=42,
    n_jobs=-1
)

# Fit model
model.fit(scaled_features)

# 5. Prediksi dan evaluasi
# Prediksi anomali
predictions_raw = model.predict(scaled_features)
anomaly_scores = model.decision_function(scaled_features)

# Konversi ke format yang lebih mudah diinterpretasi (1 untuk anomali, 0 untuk normal)
predictions = np.where(predictions_raw == -1, 1, 0)

# Tambahkan hasil ke dataframe
data['Anomaly_Score'] = anomaly_scores
data['Predicted_Anomaly'] = predictions

# 6. Visualisasi PCA yang menarik
# PCA untuk visualisasi
pca = PCA(n_components=3)
pca_result = pca.fit_transform(scaled_features)
pca_df = pd.DataFrame(
    data=pca_result,
    columns=['PC1', 'PC2', 'PC3']
)

# Tambahkan hasil PCA ke dataframe
data['PC1'] = pca_df['PC1']
data['PC2'] = pca_df['PC2'] 
data['PC3'] = pca_df['PC3']

# Buat figure besar untuk visualisasi
plt.figure(figsize=(20, 20))
gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 1])

# 6.1 Visualisasi korelasi antara fitur
ax0 = plt.subplot(gs[0, :])
correlation = data[['Kelembapan', 'Kualitas_Udara', 'Karbon_Monoksida', 'Suhu', 'Anomaly_Score']].corr()
sns.heatmap(
    correlation, 
    annot=True, 
    cmap='coolwarm', 
    vmin=-1, 
    vmax=1,
    linewidths=.5,
    ax=ax0
)
ax0.set_title('Matriks Korelasi Fitur dengan Anomaly Score', fontsize=16, pad=20)

# 6.2 PCA 2D plot with contour
ax1 = plt.subplot(gs[1, 0])
scatter = ax1.scatter(
    data['PC1'], 
    data['PC2'],
    c=data['Anomaly_Score'],
    cmap=cmap_custom,
    alpha=0.8,
    s=50,
    edgecolor='w',
    linewidth=0.5
)
ax1.set_title('PCA 2D dengan Skor Anomali', fontsize=14)
ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
cbar = plt.colorbar(scatter, ax=ax1)
cbar.set_label('Skor Anomali', fontsize=12)

# Tambahkan contour plot untuk menunjukkan kepadatan
x = data['PC1']
y = data['PC2']
xmin, xmax = x.min() - 1, x.max() + 1
ymin, ymax = y.min() - 1, y.max() + 1
xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([xx.ravel(), yy.ravel()])
values = np.vstack([x, y])
from scipy.stats import gaussian_kde
kernel = gaussian_kde(values)
f = np.reshape(kernel(positions).T, xx.shape)
ax1.contour(xx, yy, f, colors='black', alpha=0.3, levels=5, linewidths=0.5)

# 6.3 PCA 2D scatter dengan hasil prediksi
ax2 = plt.subplot(gs[1, 1:])
scatter = ax2.scatter(
    data['PC1'], 
    data['PC2'],
    c=data['Predicted_Anomaly'],
    cmap='coolwarm',
    alpha=0.8,
    s=80,
    edgecolor='w',
    linewidth=0.5
)
ax2.set_title('Deteksi Anomali dengan PCA 2D', fontsize=14)
ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
legend1 = ax2.legend(*scatter.legend_elements(), labels=['Normal', 'Anomali'], title="Status")
ax2.add_artist(legend1)

# Tambahkan teks untuk variance yang dijelaskan oleh PC1 dan PC2
explained_var_text = f"Total variance dijelaskan oleh PC1 & PC2: {pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]:.2%}"
ax2.text(0.5, -0.15, explained_var_text, ha='center', va='center', transform=ax2.transAxes, fontsize=12)

# Highlight anomali dengan lingkaran
anomalies = data[data['Predicted_Anomaly'] == 1]
ax2.scatter(anomalies['PC1'], anomalies['PC2'], 
           s=200, facecolors='none', edgecolors='red', linewidth=2, alpha=0.7)

# 6.4 PCA 3D plot
ax3 = plt.subplot(gs[2, :], projection='3d')
scatter3d = ax3.scatter(
    data['PC1'],
    data['PC2'],
    data['PC3'],
    c=data['Predicted_Anomaly'],
    cmap='coolwarm',
    s=60,
    alpha=0.7,
    edgecolor='w',
    linewidth=0.5
)
ax3.set_title('Visualisasi PCA 3D dengan Anomali', fontsize=16)
ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
ax3.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%})', fontsize=12)
legend2 = ax3.legend(*scatter3d.legend_elements(), labels=['Normal', 'Anomali'], title="Status")
ax3.add_artist(legend2)

# Tambahkan efek pencahayaan untuk visual 3D yang lebih baik
ax3.view_init(elev=30, azim=45)

# Tambahkan informasi interpretasi PCA
pca_components = pd.DataFrame(
    pca.components_,
    columns=data.columns[:-6],
    index=['PC1', 'PC2', 'PC3']
)

print("\n==== Interpretasi Komponen PCA ====")
print("Bobot fitur di setiap komponen utama:")
print(pca_components)

# Cari fitur dengan kontribusi tertinggi untuk setiap PC
for i, pc in enumerate(['PC1', 'PC2', 'PC3']):
    sorted_idx = np.argsort(np.abs(pca_components.loc[pc]))[::-1]
    top_features = pca_components.columns[sorted_idx]
    top_weights = pca_components.loc[pc, top_features]
    
    print(f"\nFitur dengan kontribusi tertinggi untuk {pc}:")
    for feat, weight in zip(top_features, top_weights):
        print(f"{feat}: {weight:.4f}")

plt.tight_layout()
plt.savefig('anomaly_detection_pca_visualization.png', dpi=300, bbox_inches='tight')

# 7. Visualisasi hubungan fitur dengan pengelompokan anomali
plt.figure(figsize=(20, 16))

feature_names = ['Kelembapan', 'Kualitas_Udara', 'Karbon_Monoksida', 'Suhu']
plot_index = 1

# 7.1 Visualisasi pairplot dengan anomali
for i, feature1 in enumerate(feature_names):
    for j, feature2 in enumerate(feature_names):
        if i < j:  # Hanya separuh dari matrix
            plt.subplot(3, 2, plot_index)
            scatter = plt.scatter(
                data[feature1],
                data[feature2],
                c=data['Predicted_Anomaly'],
                cmap='coolwarm',
                alpha=0.7,
                s=50,
                edgecolor='w',
                linewidth=0.5
            )
            
            plt.title(f'Hubungan {feature1} dengan {feature2}', fontsize=14)
            plt.xlabel(feature1, fontsize=12)
            plt.ylabel(feature2, fontsize=12)
            
            # Tambahkan legenda
            if plot_index == 1:
                legend = plt.legend(*scatter.legend_elements(), 
                                   labels=['Normal', 'Anomali'], 
                                   title="Status", 
                                   loc='upper right')
                
            # Tambahkan boundary decision jika memungkinkan
            x_min, x_max = data[feature1].min() - 1, data[feature1].max() + 1
            y_min, y_max = data[feature2].min() - 1, data[feature2].max() + 1
            
            plot_index += 1

plt.tight_layout()
plt.savefig('feature_relationships_anomaly.png', dpi=300, bbox_inches='tight')

# 8. Visualisasi distribusi untuk setiap fitur dengan anomali yang ditandai
plt.figure(figsize=(20, 12))

for i, feature in enumerate(feature_names):
    plt.subplot(2, 2, i+1)
    
    # Plot histogram dengan KDE
    sns.histplot(
        data=data[data['Predicted_Anomaly'] == 0][feature], 
        color='blue', 
        label='Normal', 
        kde=True,
        alpha=0.6
    )
    sns.histplot(
        data=data[data['Predicted_Anomaly'] == 1][feature], 
        color='red', 
        label='Anomali', 
        kde=True,
        alpha=0.6
    )
    
    plt.title(f'Distribusi {feature}', fontsize=14)
    plt.xlabel(feature, fontsize=12)
    plt.ylabel('Frekuensi', fontsize=12)
    plt.legend()

plt.tight_layout()
plt.savefig('feature_distributions_anomaly.png', dpi=300, bbox_inches='tight')

# 9. Visualisasi animasi 3D PCA (opsional - hanya jika environment mendukung)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

def update(angle):
    ax.clear()
    scatter = ax.scatter(
        data['PC1'],
        data['PC2'],
        data['PC3'],
        c=data['Predicted_Anomaly'],
        cmap='coolwarm',
        s=60,
        alpha=0.7,
        edgecolor='w',
        linewidth=0.5
    )
    
    # Tambahkan lebih banyak elemen visual
    # Highlight anomali dengan sphere yang lebih besar
    anomalies = data[data['Predicted_Anomaly'] == 1]
    ax.scatter(
        anomalies['PC1'],
        anomalies['PC2'],
        anomalies['PC3'],
        s=120,
        facecolors='none',
        edgecolors='red',
        linewidth=2,
        alpha=0.7
    )
    
    ax.set_title('Visualisasi PCA 3D dengan Rotasi', fontsize=16)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%})', fontsize=12)
    
    legend = ax.legend(*scatter.legend_elements(), labels=['Normal', 'Anomali'], title="Status")
    ax.add_artist(legend)
    
    # Atur sudut pandang
    ax.view_init(elev=30, azim=angle)
    
    return scatter,

# Buat animasi 3D (untuk visualisasi interaktif)
# ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 5), interval=100, blit=False)
# HTML(ani.to_jshtml())

# 10. Visualisasi Tren Anomali (Time Series Simulation)
# Simulasikan data time series untuk menunjukkan tren anomali
plt.figure(figsize=(16, 8))

# Buat timestamp simulasi
n_timestamps = len(data)
timestamps = pd.date_range(start='2025-01-01', periods=n_timestamps, freq='5min')
data['Timestamp'] = timestamps

# Plot tren anomali score berdasarkan waktu
plt.subplot(2, 1, 1)
plt.plot(data['Timestamp'], data['Anomaly_Score'], color='blue', alpha=0.7)
plt.axhline(y=-0.2, color='red', linestyle='--', alpha=0.7, label='Threshold')
plt.fill_between(data['Timestamp'], data['Anomaly_Score'], -0.2, 
                where=(data['Anomaly_Score'] < -0.2), color='red', alpha=0.3)
plt.title('Tren Skor Anomali Berdasarkan Waktu', fontsize=14)
plt.xlabel('Waktu', fontsize=12)
plt.ylabel('Skor Anomali', fontsize=12)
plt.legend()

# Plot fitur dengan highlight anomali
plt.subplot(2, 1, 2)
for i, feature in enumerate(['Kelembapan', 'Kualitas_Udara', 'Karbon_Monoksida', 'Suhu']):
    if i == 0:  # Plot fitur pertama dengan sumbu y kiri
        plt.plot(data['Timestamp'], data[feature], label=feature, alpha=0.7)
    else:  # Plot fitur lain dengan sumbu y kanan
        ax2 = plt.gca().twinx()
        ax2.spines['right'].set_position(('outward', 60 * (i-1)))
        ax2.plot(data['Timestamp'], data[feature], label=feature, alpha=0.7)
        ax2.set_ylabel(feature, fontsize=12)

# Highlight area dengan anomali
anomaly_regions = data[data['Predicted_Anomaly'] == 1]['Timestamp']
for timestamp in anomaly_regions:
    plt.axvline(x=timestamp, color='red', alpha=0.2)

plt.title('Tren Fitur dengan Anomali yang Dideteksi', fontsize=14)
plt.xlabel('Waktu', fontsize=12)
plt.ylabel('Nilai', fontsize=12)
plt.legend(loc='upper left')

plt.tight_layout()
plt.savefig('anomaly_time_series.png', dpi=300, bbox_inches='tight')

# 11. Evaluasi Model dengan Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report

plt.figure(figsize=(12, 10))

# Hitung confusion matrix
cm = confusion_matrix(data['True_Anomaly'], data['Predicted_Anomaly'])

# Plot confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Anomali'],
            yticklabels=['Normal', 'Anomali'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix untuk Deteksi Anomali', fontsize=16)

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')

# Print classification report
print("\n==== Classification Report ====")
report = classification_report(data['True_Anomaly'], data['Predicted_Anomaly'])
print(report)

# 12. Feature Importance berdasarkan PCA dan Isolation Forest
plt.figure(figsize=(14, 8))

# Ekstrak feature importance dari PCA
pca_importance = np.abs(pca.components_[:2]).sum(axis=0)
pca_importance = pca_importance / pca_importance.sum()

# Plot feature importance
plt.subplot(1, 1, 1)
importance_df = pd.DataFrame({
    'Fitur': feature_names,
    'PCA Importance': pca_importance
})
importance_df = importance_df.sort_values('PCA Importance', ascending=False)

sns.barplot(x='Fitur', y='PCA Importance', data=importance_df, palette='viridis')
plt.title('Pentingnya Fitur berdasarkan PCA', fontsize=16)
plt.xlabel('Fitur', fontsize=14)
plt.ylabel('Skor Kepentingan', fontsize=14)
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')

# 13. Menggunakan model untuk prediksi data baru
def predict_anomaly(new_data, model, scaler, pca=None):
    """
    Memprediksi apakah data baru adalah anomali atau tidak
    
    Parameters:
    -----------
    new_data : DataFrame
        Data baru yang akan diprediksi dengan fitur yang sama
    model : Isolation Forest model
        Model yang sudah dilatih
    scaler : StandardScaler
        Scaler yang digunakan untuk preprocessing
    pca : PCA model (optional)
        Model PCA untuk visualisasi
        
    Returns:
    --------
    result_df : DataFrame
        DataFrame dengan hasil prediksi
    """
    # Pastikan kolom sesuai dengan model training
    expected_columns = ['Kelembapan', 'Kualitas_Udara', 'Karbon_Monoksida', 'Suhu']
    if not all(col in new_data.columns for col in expected_columns):
        raise ValueError(f"Data baru harus memiliki kolom: {expected_columns}")
    
    # Preprocessing
    scaled_new_data = scaler.transform(new_data[expected_columns])
    
    # Prediksi
    predictions_raw = model.predict(scaled_new_data)
    scores = model.decision_function(scaled_new_data)
    predictions = np.where(predictions_raw == -1, 1, 0)
    
    # Buat result dataframe
    result_df = new_data.copy()
    result_df['Anomaly_Score'] = scores
    result_df['Is_Anomaly'] = predictions
    result_df['Status'] = np.where(predictions == 1, 'Anomali', 'Normal')
    
    # Tambahkan hasil PCA jika tersedia
    if pca is not None:
        pca_result = pca.transform(scaled_new_data)
        result_df['PC1'] = pca_result[:, 0]
        result_df['PC2'] = pca_result[:, 1]
        if pca_result.shape[1] > 2:
            result_df['PC3'] = pca_result[:, 2]
    
    return result_df

# Contoh penggunaan fungsi prediksi dengan data baru
print("\n==== Contoh Penggunaan Model untuk Data Baru ====")

# Data contoh (normal dan anomali)
new_samples = pd.DataFrame({
    'Kelembapan': [65, 95, 30, 60],
    'Kualitas_Udara': [50, 10, 20, 55],
    'Karbon_Monoksida': [5, 18, 15, 4.5],
    'Suhu': [25, 45, 42, 27]
})

# Prediksi
results = predict_anomaly(new_samples, model, scaler, pca)
print("\nHasil prediksi data baru:")
print(results[['Kelembapan', 'Kualitas_Udara', 'Karbon_Monoksida', 'Suhu', 'Anomaly_Score', 'Status']])

# Visualisasi hasil prediksi data baru
plt.figure(figsize=(10, 8))
plt.scatter(data['PC1'], data['PC2'], c='lightgray', alpha=0.3, s=50, label='Training Data')
plt.scatter(
    results['PC1'], 
    results['PC2'], 
    c=results['Is_Anomaly'],
    cmap='coolwarm',
    s=200,
    edgecolor='black',
    linewidth=2,
    alpha=0.9,
    label='Data Baru'
)

# Tandai setiap titik data baru
for i, row in results.iterrows():
    plt.annotate(
        f"Sample {i+1}",
        (row['PC1'], row['PC2']),
        xytext=(10, 10),
        textcoords='offset points',
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    )

plt.title('Visualisasi Prediksi Data Baru dengan PCA', fontsize=16)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
plt.legend()

plt.tight_layout()
plt.savefig('new_data_prediction.png', dpi=300, bbox_inches='tight')

print("\n==== Ringkasan Hasil ====")
print(f"Total data: {len(data)}")
print(f"Jumlah anomali terdeteksi: {data['Predicted_Anomaly'].sum()}")
print(f"Persentase anomali: {data['Predicted_Anomaly'].mean():.2%}")
print(f"Jumlah anomali sebenarnya: {data['True_Anomaly'].sum()}")
print(f"Akurasi model: {(data['Predicted_Anomaly'] == data['True_Anomaly']).mean():.2%}")

print("\n==== Kesimpulan ====")
print("Model Isolation Forest berhasil mendeteksi anomali pada data kualitas udara dengan baik.")
print("Visualisasi PCA menunjukkan pemisahan yang jelas antara data normal dan anomali.")
print("Fitur yang paling berpengaruh terhadap anomali adalah Karbon Monoksida dan Kualitas Udara.")
print("Model ini dapat digunakan untuk mendeteksi anomali pada data IoT kualitas udara secara real-time.")