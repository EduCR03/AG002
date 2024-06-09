import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Carregar e preparar os dados
df = pd.read_csv('palmerpenguins.csv')

mappings = {
    "island": {"Biscoe": 0, "Dream": 1, "Torgersen": 2},
    "sex": {"FEMALE": 0, "MALE": 1},
    "species": {"Adelie": 0, "Chinstrap": 1, "Gentoo": 2}
}
df.replace(mappings, inplace=True)

df.rename(columns={
    "culmen_length_mm": "culmenlengthmm",
    "culmen_depth_mm": "culmendepthmm",
    "flipper_length_mm": "flipperlengthmm",
    "body_mass_g": "bodymassg"
}, inplace=True)

print(df.head())

df = df[["island", "sex", "culmenlengthmm", "culmendepthmm", "flipperlengthmm", "bodymassg", "species"]]

print(df.head())

fig1, axes1 = plt.subplots(figsize=(4.2, 3))
sns.boxplot(x='species', y='flipperlengthmm', data=df, ax=axes1, hue='species', legend=True)
axes1.set_title("Distribuição do comprimento da nadadeira")

fig2, axes2 = plt.subplots(figsize=(4.2, 3))
sns.boxplot(x='species', y='culmenlengthmm', data=df, ax=axes2, hue='species', legend=True)
axes2.set_title("Distribuição do comprimento do bico")

fig3, axes3 = plt.subplots(figsize=(4.2, 3))
sns.boxplot(x='species', y='culmendepthmm', data=df, ax=axes3, hue='species', legend=True)
axes3.set_title("Distribuição de profundidade do bico")

fig4, axes4 = plt.subplots(figsize=(4.2, 3))
sns.boxplot(x='species', y='bodymassg', data=df, ax=axes4, hue='species', legend=True)
axes4.set_title("Distribuição de massa corporal")

plt.tight_layout()

# Preenchimento de valores ausentes
df['sex'].fillna(df['sex'].mode()[0], inplace=True)
for item in ['culmenlengthmm', 'culmendepthmm', 'flipperlengthmm', 'bodymassg']:
    df[item].fillna(df[item].mean(), inplace=True)

# Escalonamento e criação de variáveis dummy
dummies = pd.get_dummies(df[['island', 'sex']], drop_first=True)
df_feat = df.drop(['island', 'sex', 'species'], axis=1)
scaler = StandardScaler() 
df_scaled = scaler.fit_transform(df_feat)
df_scaled = pd.DataFrame(df_scaled, columns=df_feat.columns)
df_preprocessed = pd.concat([df_scaled, dummies, df['species']], axis=1)

# KMeans
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10)
kmeans.fit(df_preprocessed.drop('species', axis=1))
print(classification_report(df_preprocessed['species'], kmeans.labels_))
print(f"Accuracy is {np.round(100 * accuracy_score(df_preprocessed['species'], kmeans.labels_), 2)}")

wcss = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10)
    kmeans.fit(df_preprocessed.drop('species', axis=1))
    wcss.append(kmeans.inertia_)

# K-Nearest Neighbors
X_train, X_test, y_train, y_test = train_test_split(df_preprocessed.drop('species', axis=1), df['species'], test_size=0.80)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
preds_knn = knn.predict(X_test)
print(confusion_matrix(y_test, preds_knn))

print(classification_report(y_test, preds_knn))
print(accuracy_score(y_test, preds_knn))
print('Training set score: {:.4f}'.format(knn.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(knn.score(X_test, y_test)))

error_rate = []
for i in range(1, 10):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    error_rate.append(np.mean(knn.predict(X_test) != y_test))


knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)
preds_knn = knn.predict(X_test)
print(confusion_matrix(y_test, preds_knn))
print(classification_report(y_test, preds_knn))

plt.show()
