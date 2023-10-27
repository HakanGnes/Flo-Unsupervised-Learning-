# Flo Unsupervised Learning with Customer Segmentation

# imports

import numpy as np
import pandas as pd
from scipy import stats
import datetime as dt
from sklearn.cluster import AgglomerativeClustering
import random
from scipy.stats import jarque_bera
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 500)
# Data Visualizing

df = pd.read_csv("Data/flo_data_20k.csv")
df.shape
df.head()
df.info()
df.isnull().sum()

# Create Today Date for Recency
df["last_order_date"].max() # 2021-05-30
today_date = dt.datetime(2021,6,1)

# Create Date Columns
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

# Create variables for segmentation
df["recency"] = (today_date - df["last_order_date"]).astype('timedelta64[D]')
df["tenure"] = (df["last_order_date"]-df["first_order_date"]).astype('timedelta64[D]')
df["customer_value"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]
df["order_total"] = df["order_num_total_ever_offline"] + df["order_num_total_ever_online"]

final_df = df[["order_total","customer_value","tenure","recency"]]
final_df.head()

# Data Transformation
stats.probplot(final_df["order_total"], dist="norm", plot=plt)
plt.show()
stats.probplot(final_df["tenure"], dist="norm", plot=plt)
plt.show()
stats.probplot(final_df["customer_value"], dist="norm", plot=plt)
plt.show()
stats.probplot(final_df["recency"], dist="norm", plot=plt)
plt.show()

stat, p = jarque_bera(final_df)
if p > 0.05:
    print("normal distribution.")
else:
    print("NOT normal distribution.")

# We do not have normal distribution,we need transformations.

# Standard Scaler
# scaler = StandardScaler()
# final_df1 = scaler.fit_transform(final_df)
# Standar Scaler is not enough for this data.

# Log Transformation
final_df = np.log1p(final_df)

# MinmaxScaler
sc = MinMaxScaler((0, 1))
data_transform = sc.fit_transform(final_df)
model_df=pd.DataFrame(data_transform ,columns=final_df.columns)
model_df.head()

# Create K-means and find Optimum Number of Clusters

kmeans = KMeans()
ssd = []
K = range(1, 30)

for k in K:
    kmeans = KMeans(n_clusters=k).fit(final_df)
    ssd.append(kmeans.inertia_)

plt.plot(K, ssd, "bx-")
plt.xlabel("Different K Values on SSE/SSR/SSD")
plt.title("Elbow Method for Optimum Number of Clusters")
plt.show()

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(final_df)
elbow.show()

elbow.elbow_value_

# Final Clusters

kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(final_df)

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_
final_df[0:5]
clusters_kmeans = kmeans.labels_
final_df["cluster"] = clusters_kmeans
final_df["cluster"] = final_df["cluster"] + 1

final_df.groupby("cluster").agg(["count","mean","median"])



# Customer Segmentation with Hierarchical Clustering

hc = AgglomerativeClustering(n_clusters=6)
clusters_kmeans = hc.fit_predict(model_df)
final_df = df[["master_id","order_total","customer_value","recency","tenure","cluster"]]
final_df["segment"] = clusters_kmeans
final_df.groupby("segment").agg(["count","mean","median"])
final_df.head()