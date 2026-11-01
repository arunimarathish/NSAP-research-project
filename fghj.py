import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# STEP 1: Load the dataset
df = pd.read_csv("nsap_mpi_state_ut_merged_preliminary.csv")

print("NSAP Research Project")
print("=====================")

print("\nNumber of states and UTs:", len(df))

# STEP 2: Display the first five rows
print("\nFirst five rows:")
print(df.head())

# STEP 3: Calculate total beneficiaries
pension_columns = [
    "Old_Age_Pension_Beneficiaries",
    "Widow_Pension_Beneficiaries",
    "Disability_Pension_Beneficiaries"
]

print("\nTotal beneficiaries by scheme:")
print(df[pension_columns].sum())

# STEP 4: Examine poverty and pension associations
poverty_column = "MPI_Headcount_Ratio_2019_21_Percent"

print("\nCorrelation with poverty indicator:")

for column in pension_columns:
    correlation = df[column].corr(df[poverty_column])
    print(column, ":", round(correlation, 3))

# STEP 5: Create a bar graph
df_sorted = df.sort_values(
    "Old_Age_Pension_Beneficiaries",
    ascending=False
)

plt.figure(figsize=(12, 6))

plt.bar(
    df_sorted["State_UT"],
    df_sorted["Old_Age_Pension_Beneficiaries"]
)

plt.xticks(rotation=90)
plt.xlabel("State / Union Territory")
plt.ylabel("Old-age pension beneficiaries")
plt.title("State-wise Old-age Pension Beneficiaries")

plt.tight_layout()
plt.savefig("old_age_pension_graph.png", dpi=300)
plt.show()

# STEP 6: Prepare data for K-means
features = [
    "Old_Age_Pension_Beneficiaries",
    "Widow_Pension_Beneficiaries",
    "Disability_Pension_Beneficiaries",
    poverty_column
]

cluster_data = df[["State_UT"] + features].dropna().copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(cluster_data[features])

# STEP 7: Apply K-means clustering
model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

cluster_data["Cluster"] = model.fit_predict(X_scaled)

print("\nClustering results:")
print(
    cluster_data[["State_UT", "Cluster"]]
    .sort_values("Cluster")
    .to_string(index=False)
)

# STEP 8: Evaluate the clusters
if cluster_data["Cluster"].nunique() > 1:
    score = silhouette_score(
        X_scaled,
        cluster_data["Cluster"]
    )
    print("\nSilhouette Score:", round(score, 3))

# STEP 9: Save results
cluster_data.to_csv(
    "nsap_clustering_results.csv",
    index=False
)

print("\nAnalysis completed successfully!")
print("Results saved to nsap_clustering_results.csv")
print("Graph saved to old_age_pension_graph.png")