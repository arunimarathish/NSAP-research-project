# ==========================================================
# NSAP SOCIAL PROTECTION RESEARCH PROJECT
# Data Analysis and Graph Generation
# ==========================================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ----------------------------------------------------------
# STEP 1: CREATE OUTPUT FOLDER
# ----------------------------------------------------------

os.makedirs("NSAP_Graphs", exist_ok=True)

print("NSAP Research Project Started!")

# ----------------------------------------------------------
# STEP 2: LOAD DATASETS
# ----------------------------------------------------------

df = pd.read_csv("nsap_mpi_state_ut_merged_preliminary.csv")

cluster_df = pd.read_csv("nsap_clustering_results.csv")

print("\nNSAP Dataset Loaded Successfully!")
print("Total States and UTs:", len(df))

print("\nFirst five rows:")
print(df.head())

# ----------------------------------------------------------
# STEP 3: CHECK REQUIRED COLUMNS
# ----------------------------------------------------------

required_columns = [
    "State_UT",
    "Old_Age_Pension_Beneficiaries",
    "Widow_Pension_Beneficiaries",
    "Disability_Pension_Beneficiaries",
    "MPI_Headcount_Ratio_2019_21_Percent"
]

for col in required_columns:
    if col not in df.columns:
        raise ValueError("Missing column: " + col)

# Convert numerical columns safely
for col in required_columns[1:]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=required_columns)

# ----------------------------------------------------------
# GRAPH 1: STATE-WISE OLD-AGE PENSION BENEFICIARIES
# ----------------------------------------------------------

old_age = df.sort_values(
    by="Old_Age_Pension_Beneficiaries",
    ascending=False
)

plt.figure(figsize=(18, 8))

plt.bar(
    old_age["State_UT"],
    old_age["Old_Age_Pension_Beneficiaries"]
)

plt.title(
    "State-wise Old-age Pension Beneficiaries",
    fontsize=16
)

plt.xlabel("State / Union Territory")
plt.ylabel("Number of Beneficiaries")

plt.xticks(rotation=90)

plt.tight_layout()

plt.savefig(
    "NSAP_Graphs/Figure1_Old_Age_Pension.png",
    dpi=300
)

plt.show()

# ----------------------------------------------------------
# GRAPH 2: COMPARISON OF THREE PENSION SCHEMES
# ----------------------------------------------------------

scheme_names = [
    "Old-age Pension",
    "Widow Pension",
    "Disability Pension"
]

scheme_totals = [
    df["Old_Age_Pension_Beneficiaries"].sum(),
    df["Widow_Pension_Beneficiaries"].sum(),
    df["Disability_Pension_Beneficiaries"].sum()
]

plt.figure(figsize=(10, 6))

bars = plt.bar(scheme_names, scheme_totals)

plt.title(
    "Comparison of NSAP Pension Schemes",
    fontsize=16
)

plt.xlabel("Pension Scheme")
plt.ylabel("Number of Beneficiaries")

plt.bar_label(bars, fmt="%.0f", padding=3)

plt.tight_layout()

plt.savefig(
    "NSAP_Graphs/Figure2_Pension_Comparison.png",
    dpi=300
)

plt.show()

# ----------------------------------------------------------
# GRAPH 3: STATE-WISE WIDOW PENSION BENEFICIARIES
# ----------------------------------------------------------

widow = df.sort_values(
    by="Widow_Pension_Beneficiaries",
    ascending=False
)

plt.figure(figsize=(18, 8))

plt.bar(
    widow["State_UT"],
    widow["Widow_Pension_Beneficiaries"]
)

plt.title(
    "State-wise Widow Pension Beneficiaries",
    fontsize=16
)

plt.xlabel("State / Union Territory")
plt.ylabel("Number of Beneficiaries")

plt.xticks(rotation=90)

plt.tight_layout()

plt.savefig(
    "NSAP_Graphs/Figure3_Widow_Pension.png",
    dpi=300
)

plt.show()

# ----------------------------------------------------------
# GRAPH 4: STATE-WISE DISABILITY PENSION BENEFICIARIES
# ----------------------------------------------------------

disability = df.sort_values(
    by="Disability_Pension_Beneficiaries",
    ascending=False
)

plt.figure(figsize=(18, 8))

plt.bar(
    disability["State_UT"],
    disability["Disability_Pension_Beneficiaries"]
)

plt.title(
    "State-wise Disability Pension Beneficiaries",
    fontsize=16
)

plt.xlabel("State / Union Territory")
plt.ylabel("Number of Beneficiaries")

plt.xticks(rotation=90)

plt.tight_layout()

plt.savefig(
    "NSAP_Graphs/Figure4_Disability_Pension.png",
    dpi=300
)

plt.show()

# ----------------------------------------------------------
# GRAPH 5: STATE-WISE MULTIDIMENSIONAL POVERTY
# ----------------------------------------------------------

mpi = df.sort_values(
    by="MPI_Headcount_Ratio_2019_21_Percent",
    ascending=True
)

plt.figure(figsize=(12, 10))

plt.barh(
    mpi["State_UT"],
    mpi["MPI_Headcount_Ratio_2019_21_Percent"]
)

plt.title(
    "State-wise Multidimensional Poverty Index",
    fontsize=16
)

plt.xlabel("MPI Headcount Ratio (%)")
plt.ylabel("State / Union Territory")

plt.tight_layout()

plt.savefig(
    "NSAP_Graphs/Figure5_MPI_Poverty.png",
    dpi=300
)

plt.show()

# ----------------------------------------------------------
# GRAPH 6: K-MEANS CLUSTERING VISUALISATION
# Uses the existing clustering results CSV
# ----------------------------------------------------------

required_cluster_columns = [
    "State_UT",
    "Old_Age_Pension_Beneficiaries",
    "MPI_Headcount_Ratio_2019_21_Percent",
    "Cluster"
]

for col in required_cluster_columns:
    if col not in cluster_df.columns:
        raise ValueError("Missing clustering column: " + col)

plt.figure(figsize=(11, 7))

for cluster in sorted(cluster_df["Cluster"].unique()):

    group = cluster_df[
        cluster_df["Cluster"] == cluster
    ]

    plt.scatter(
        group["Old_Age_Pension_Beneficiaries"],
        group["MPI_Headcount_Ratio_2019_21_Percent"],
        label=f"Cluster {cluster}",
        s=80
    )

plt.title(
    "K-means Clustering of Indian States and Union Territories",
    fontsize=15
)

plt.xlabel("Old-age Pension Beneficiaries")
plt.ylabel("MPI Headcount Ratio (%)")

plt.xscale("symlog")

plt.legend(title="Cluster")

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "NSAP_Graphs/Figure6_KMeans_Clustering.png",
    dpi=300
)

plt.show()

# ----------------------------------------------------------
# STEP 7: SAVE SUMMARY TABLE
# ----------------------------------------------------------

summary = pd.DataFrame({
    "Pension Scheme": scheme_names,
    "Total Beneficiaries": scheme_totals
})

summary.to_csv(
    "NSAP_Graphs/Pension_Scheme_Summary.csv",
    index=False
)

# ----------------------------------------------------------
# STEP 8: SAVE CLUSTER DISTRIBUTION
# ----------------------------------------------------------

cluster_summary = (
    cluster_df.groupby("Cluster")
    .agg(
        Number_of_States=("State_UT", "count"),
        States=("State_UT", lambda x: ", ".join(x))
    )
    .reset_index()
)

cluster_summary.to_csv(
    "NSAP_Graphs/Cluster_Summary.csv",
    index=False
)

# ----------------------------------------------------------
# STEP 9: DISPLAY FINAL RESULTS
# ----------------------------------------------------------

print("\n======================================")
print("NSAP RESEARCH ANALYSIS COMPLETED!")
print("======================================")

print("\nPension Scheme Summary:")
print(summary)

print("\nClustering Summary:")
print(cluster_summary)


print("NSAP_Graphs folder")

