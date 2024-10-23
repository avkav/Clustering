# Importar las librerías necesarias
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# 1. Leer el conjunto de datos y primer vistazo
df = pd.read_csv('mushrooms.csv')  
print(df.head())

# 2. Exploración de datos
print(df.describe())
print(df.info())

# 3. Comprobar si hay valores nulos
print(df.isnull().sum())

# 4. Buscar valores únicos en cada feature
unique_values = pd.DataFrame({
    'features': df.columns,
    'n_values': [df[col].nunique() for col in df.columns]
})
print(unique_values)

# 5. Eliminar valores nulos (si los hubiera, en este caso no debería haber)
df_cleaned = df.dropna()

# 6. Eliminar columnas sin variación (con un solo valor único)
features_to_drop = [col for col in df_cleaned.columns if df_cleaned[col].nunique() == 1]
df_cleaned = df_cleaned.drop(features_to_drop, axis=1)
print("Features eliminadas:", features_to_drop)

# 7. Separar entre variables predictoras y variable a predecir
X = df_cleaned.drop('class', axis=1)
y = df_cleaned['class']

# 8. Codificar correctamente las variables categóricas
X = pd.get_dummies(X)

# 9. Codificar la variable objetivo (y) usando LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 10. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.33, random_state=42)

# 11. PCA para reducción de dimensiones
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)

# 12. Visualización del PCA en un scatterplot
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis')
plt.title('PCA - Training Set')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar(label='Clases')
plt.show()

# 13. Entrenamiento con Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
accuracy = rf.score(X_test, y_test)
print(f'Accuracy del modelo Random Forest: {accuracy:.2f}')

# 14. Reducción de features usando PCA en un rango de n_features
n_features = range(1, X_train.shape[1] + 1)
scores = []

for n in n_features:
    pca = PCA(n_components=n)
    X_train_pca = pca.fit_transform(X_train)

    # Entrenar un Random Forest con las features reducidas
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_pca, y_train)

    # Guardar el score (precisión)
    scores.append(rf.score(pca.transform(X_test), y_test))

# Visualización de la precisión vs número de features
sns.lineplot(x=n_features, y=scores)
plt.title('Precisión vs Número de Features con PCA')
plt.xlabel('Número de Features')
plt.ylabel('Precisión')
plt.show()

# 15. Clustering con K-Means
k_values = range(2, 10)
scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_train_pca)
    scores.append(kmeans.inertia_)

# Gráfico del codo para encontrar el número óptimo de clusters
sns.lineplot(x=k_values, y=scores)
plt.title('Método del Codo para K-Means')
plt.xlabel('Número de Clusters')
plt.ylabel('Inercia')
plt.show()

# 16. Aplicar K-Means con el número óptimo de clusters
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_train_pca)

# 17. Scatterplot del PCA coloreado por los clusters de KMeans
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=kmeans.labels_, cmap='rainbow')
plt.title('PCA - Clustering con KMeans')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar(label='Clusters')
plt.show()
