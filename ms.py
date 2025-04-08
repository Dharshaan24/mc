# Step 1: Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

# Step 2: Load the dataset
df = pd.read_csv("D:\downloads\2021 FE Guide for DOE-release dates before 8-10-2021-no-sales -8-9-2021public - 21.csv".csv")  # Replace with your actual filename

# Step 3: Select relevant features for segmentation
features = [
    'Eng Displ', '# Cyl', 'Transmission',
    'City FE (Guide) - Conventional Fuel',
    'Hwy FE (Guide) - Conventional Fuel',
    'Comb FE (Guide) - Conventional Fuel',
    'Division'
]
df_selected = df[features].copy()

# Step 4: Drop rows with major missing values
df_selected = df_selected.dropna(subset=['Eng Displ', '# Cyl'])

# Step 5: Define numeric and categorical features
numeric_features = [
    'Eng Displ', '# Cyl',
    'City FE (Guide) - Conventional Fuel',
    'Hwy FE (Guide) - Conventional Fuel',
    'Comb FE (Guide) - Conventional Fuel'
]
categorical_features = ['Transmission', 'Division']

# Step 6: Build preprocessing pipelines
preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ]), numeric_features),
    ('cat', Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]), categorical_features)
])

# Step 7: Apply preprocessing
X_preprocessed = preprocessor.fit_transform(df_selected)

# Step 8: Apply KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_preprocessed)
df_selected['Segment'] = clusters

# Step 9: Visualize segment distributions
sns.set(style="whitegrid")
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
axes = axes.flatten()

for idx, feature in enumerate(numeric_features):
    sns.boxplot(x='Segment', y=feature, data=df_selected, ax=axes[idx])
    axes[idx].set_title(f'{feature} by Segment')

fig.delaxes(axes[-1])  # Remove the unused subplot
plt.tight_layout()
plt.show()

# Step 10: Show counts per segment
print("\nVehicle count per market segment:")
print(df_selected['Segment'].value_counts().sort_index())
