import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Tentukan path file yang benar
file_path = 'Data_Sayuran_Kudus.csv'

# Baca data dari file CSV
data = pd.read_csv(file_path)

# Tampilkan beberapa baris pertama dari data untuk memahami strukturnya
print(data.head())

# Misalkan kita ingin menggunakan semua 11 kolom untuk clustering
X = data.iloc[:, :11].values  # Menggunakan semua 11 kolom sebagai fitur

# Skala data jika diperlukan (opsional)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Tentukan jumlah cluster yang diinginkan, misalnya 3
kmeans = KMeans(n_clusters=3, random_state=42)

# Fitting model K-Means ke data yang sudah dipersiapkan
kmeans.fit(X_scaled)

# Dapatkan hasil clustering
labels = kmeans.labels_

# Tambahkan hasil clustering ke dalam dataframe
data['Cluster'] = labels

# Visualisasi hasil clustering
# Anda dapat memilih dua kolom pertama untuk plot, atau kombinasi lain yang relevan
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.xlabel('Fitur 1')
plt.ylabel('Fitur 2')
plt.title('Hasil K-Means Clustering')
plt.show()
