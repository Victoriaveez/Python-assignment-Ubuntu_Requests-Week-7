# iris_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import os

# Create directory for plots
os.makedirs("plots", exist_ok=True)

def load_iris_dataset():
    """Load the iris dataset and convert to DataFrame."""
    try:
        iris = load_iris(as_frame=True)
        df = iris.frame
        return df
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

def explore_data(df):
    print("\n🔎 First 5 rows of the dataset:")
    print(df.head())

    print("\n📐 Data types:")
    print(df.dtypes)

    print("\n🧼 Checking for missing values:")
    print(df.isnull().sum())

    print("\n🧹 Cleaning data (if needed)...")
    df_cleaned = df.dropna()
    return df_cleaned

def basic_analysis(df):
    print("\n📊 Statistical Summary:")
    print(df.describe())

    print("\n📈 Mean of numerical columns grouped by species:")
    grouped = df.groupby("target").mean()
    print(grouped)

    return grouped

def create_visualizations(df):
    sns.set(style="whitegrid")

    # Add a fake time column for line chart simulation
    df["Index"] = df.index

    # 1. Line Chart - Sepal Length over index (simulating time)
    plt.figure(figsize=(8, 5))
    sns.lineplot(x="Index", y="sepal length (cm)", data=df)
    plt.title("Line Chart: Sepal Length Over Index")
    plt.xlabel("Index")
    plt.ylabel("Sepal Length (cm)")
    plt.savefig("plots/line_plot.png")
    plt.close()

    # 2. Bar Chart - Average Petal Length per Species
    plt.figure(figsize=(8, 5))
    sns.barplot(x="target", y="petal length (cm)", data=df, ci=None)
    plt.title("Bar Chart: Avg Petal Length by Species")
    plt.xlabel("Species (target)")
    plt.ylabel("Average Petal Length (cm)")
    plt.savefig("plots/bar_chart.png")
    plt.close()

    # 3. Histogram - Distribution of Sepal Width
    plt.figure(figsize=(8, 5))
    sns.histplot(df["sepal width (cm)"], bins=20, kde=True)
    plt.title("Histogram: Sepal Width Distribution")
    plt.xlabel("Sepal Width (cm)")
    plt.ylabel("Frequency")
    plt.savefig("plots/histogram.png")
    plt.close()

    # 4. Scatter Plot - Sepal Length vs Petal Length
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        x="sepal length (cm)",
        y="petal length (cm)",
        hue="target",
        palette="deep",
        data=df
    )
    plt.title("Scatter Plot: Sepal Length vs Petal Length")
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Petal Length (cm)")
    plt.legend(title="Species")
    plt.savefig("plots/scatter_plot.png")
    plt.close()

    print("\n📸 Visualizations saved in the 'plots' folder.")

def main():
    print("🌸 Iris Dataset Analysis Script 🌸")
    
    df = load_iris_dataset()
    if df is None:
        return

    df = explore_data(df)
    grouped_data = basic_analysis(df)
    create_visualizations(df)

    print("\n✅ Analysis complete. Please check the plots and summary above!")

if __name__ == "__main__":
    main() 
