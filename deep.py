# import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle

df = pd.read_csv("stroke.csv")
df.head()

# target and features
target = df[['stroke']]
features = df.drop(columns=["stroke"], axis=1)

# cat x
categorical = features.select_dtypes(include=["object", "bool"])
categorical.head()
from sklearn.impute import SimpleImputer

cat_impute = SimpleImputer(strategy="most_frequent")

cat_data = cat_impute.fit_transform(categorical)
cat_df = pd.DataFrame(cat_data, columns=categorical.columns)

# num x
numericals = features.select_dtypes(include=["int", "float"])
numericals.head()
num_impute = SimpleImputer(strategy="mean")

num_data = num_impute.fit_transform(numericals)
num_df = pd.DataFrame(num_data, columns=numericals.columns)

# c + n
X = pd.concat([encoded_cat, robust_numericdf], axis=1)
y = target

# train_test
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# cm fun
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Function to plot confusion matrix
def plot_confusion_matrix(y_test, y_pred, title, color):
    
    plt.figure(figsize=(14, 12))
    
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, cmap=color)

    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    
    plt.show()
    
# cr
from sklearn.metrics import classification_report

print(classification_report(y_test, forest_pred))

# line
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Separate the Categorical and Numerical Columns

# Numeric columns
numeric_cols = features.select_dtypes(include=['int64','float64']).columns
print(numeric_cols)

# categorical columns
categorical_cols = features.select_dtypes(include=['object']).columns
print(categorical_cols)

# start
from sklearn.pipeline import Pipeline

n_transformer = Pipeline(steps=
    [
        ("imputeN", SimpleImputer(strategy="mean")),
        ("scaler", RobustScaler())
    ]
)

n_transformer

from sklearn.preprocessing import OneHotEncoder
c_transformer = Pipeline(steps=
    [
        ("imputeC", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

c_transformer

# apply trans
from sklearn.compose import ColumnTransformer

pre = ColumnTransformer(transformers=
                        [
                            ("categoric", c_transformer, categorical_cols),
                            ("numeric", n_transformer, numeric_cols)
                        ]
)

pre

# line mod
from sklearn.svm import SVC

# Create Support Vector Machine model/estimator
svm = SVC()

svm_model = Pipeline(steps=
                   [
                       ("preprocessing", pre),
                       ("est", svm)
                   ]
)

svm_model

X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.2, random_state=144, stratify=target)

from sklearn import set_config

# train pipeline model
set_config(display="diagram")
svm_model.fit(X_train, Y_train)

# construct cm_cr
from sklearn.metrics import classification_report, recall_score, f1_score, precision_score, confusion_matrix
import seaborn as sns

# Make a prediction
y_pred = svm_model.predict(X_test)

# Summarize the fit of the model
report = classification_report(Y_test, y_pred, target_names=["No Stroke", "Stroke"])
print(report)

# Confusion Matrix
cm = confusion_matrix(Y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='OrRd_r');

plt.ylabel("Actual Labels")
plt.xlabel("Predicted Labels")

plt.show()

# save and load
# import library to save the trained ML model
import pickle

# save the model
svm_pickle = open("svm_Stroke_predictor.pickle", "wb")
pickle.dump(svm_model, svm_pickle)
svm_pickle.close()

svm_model = pickle.load(open("svm_Stroke_predictor.pickle", "rb"))
svm_model

# unsupervised
# kmeans
features = df[["column1", "column2"]]
from sklearn.cluster import KMeans
# create kmean model
kmeans = KMeans(n_clusters=5)
kmeans.fit(X)

# choose k - elbow
wcss = []

for i in range(1, 11):
    kmeans = KMeans(
        n_clusters = i,
        init = 'k-means++',
        random_state=42
    )
    kmeans.fit(X)
    wcss.append([i, kmeans.inertia_]) # kmeans.inertial_ returns the calculated WCSS Values

wcss_dataframe = pd.DataFrame(wcss, columns=["clusters", "wcss value"])

# Plot for Elbow Method
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["figure.dpi"] = 80
sns.lineplot(
    x = wcss_dataframe.clusters.values,
    y = wcss_dataframe["wcss value"], marker="o")
plt.xticks(np.arange(1, 11))
plt.xlabel("Clusters")
plt.ylabel("Wcss Values")
plt.title("Elbow Method Plot")
plt.show()

# sel_sco
from sklearn.metrics import silhouette_score

y_pred = kmeans.predict(X)
score = silhouette_score(X, y_pred, random_state=42)
np.round(score, 3)

# choose k with sel_sco
for i in range(2, 12):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    y_pred = kmeans.predict(X)
    score = silhouette_score(X, y_pred, random_state=42)
    print(f"n_clusters: {i} & Silhouette Score: {np.round(score, 3)}")
    
# dbscan
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(n_jobs=10)
dbscan.fit(X)

labels = dbscan.fit_predict(X)

# Check performance of your model.
db_score = silhouette_score(X, labels)
np.round(db_score, 3)

# agglo
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_iris
# Load datasets
winedf = pd.read_csv("winequality.csv")
winedf.head()

from sklearn.preprocessing import StandardScaler
num = winedf.select_dtypes(["int", "float"])
scaler = StandardScaler()
df = scaler.fit_transform(num)
df = pd.DataFrame(df, columns=num.columns)
df.head()

links = linkage(df, method='complete')
dendrogram(links)
plt.show()

ag = AgglomerativeClustering(n_clusters=7, linkage='complete')
ag.fit_predict(df)

df['ag_cluster_id'] = ag.labels_
df.head()

colors = {0:"red", 1:"blue", 2:"green", 3:"yellow", 4:"magenta", 5:"cyan", 6:"white"}

plt.figure(figsize=(8, 5))
plt.scatter(x=df["fixed acidity"], y=df["alcohol"], c=df["ag_cluster_id"].map(colors))
plt.title("Clusters formed by Agglomerative clustering algorithm")

plt.show()

score = silhouette_score(df.drop(columns=["ag_cluster_id"]), df["ag_cluster_id"])
np.round(score, 2)