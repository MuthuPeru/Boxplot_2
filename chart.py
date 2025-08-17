# chart.py
# Author: 24f1000447@ds.study.iitm.ac.in

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image

# Generate synthetic data
np.random.seed(42)
categories = ["Electronics", "Clothing", "Home & Kitchen", "Sports", "Books"]
scores = [np.random.normal(loc=7 + i*0.5, scale=0.5, size=30) for i in range(len(categories))]

data = pd.DataFrame({
    "Category": np.repeat(categories, 30),
    "Satisfaction": np.concatenate(scores)
})

# Professional Seaborn styling
sns.set_style("whitegrid")
sns.set_context("talk")

# Create barplot
plt.figure(figsize=(8, 8))  # gives 512x512 with dpi=64
ax = sns.barplot(x="Category", y="Satisfaction", data=data, palette="viridis", ci="sd")

# Labels and title
ax.set_title("Average Customer Satisfaction by Product Category", fontsize=16, pad=20)
ax.set_xlabel("Product Category", fontsize=14)
ax.set_ylabel("Average Satisfaction Score", fontsize=14)

# Save the figure exactly 512x512
plt.savefig("chart.png", dpi=64, bbox_inches=None)

img = Image.open("chart.png")
print(img.size) 
