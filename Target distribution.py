import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('dataset.csv')

# Count values in the target column
counts = df['Target'].value_counts()

# Ensure order
counts = counts.reindex(['Graduate', 'Dropout', 'Enrolled'])

# Colors
colors = ['#2E86C1', '#F44336', '#4CAF50']

# Plot
plt.figure(figsize=(10, 6))
bars = plt.bar(counts.index, counts.values, color=colors)

# Title and labels
plt.title('Target Class Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Class')
plt.ylabel('Count')

# Add numbers on top
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 20,
             int(height), ha='center', fontsize=12)

plt.tight_layout()
plt.savefig('figures/target_distribution.png', dpi=150, bbox_inches='tight')
plt.show()