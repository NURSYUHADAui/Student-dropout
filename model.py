import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

# 1. LOAD & PREPROCESS

df = pd.read_csv("dataset.csv")

le = LabelEncoder()
df['Target_encoded'] = le.fit_transform(df['Target'])
print("Class mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

X = df.drop(columns=['Target', 'Target_encoded'])
y = df['Target_encoded']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape} | Test: {X_test.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# 2. BASELINE MODELS (all 34 features)
print("\n--- BASELINE ---")

lr_base = LogisticRegression(max_iter=1000, random_state=42)
lr_base.fit(X_train_scaled, y_train)
lr_base_pred = lr_base.predict(X_test_scaled)
lr_base_acc  = accuracy_score(y_test, lr_base_pred)

rf_base = RandomForestClassifier(n_estimators=100, random_state=42)
rf_base.fit(X_train_scaled, y_train)
rf_base_pred = rf_base.predict(X_test_scaled)
rf_base_acc  = accuracy_score(y_test, rf_base_pred)

print(f"Logistic Regression Accuracy: {lr_base_acc:.4f}")
print(f"Random Forest Accuracy:       {rf_base_acc:.4f}")

print("\nLogistic Regression Report:")
print(classification_report(y_test, lr_base_pred, target_names=le.classes_))
print("Random Forest Report:")
print(classification_report(y_test, rf_base_pred, target_names=le.classes_))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, pred, title in zip(
    axes,
    [lr_base_pred, rf_base_pred],
    ['Logistic Regression (Baseline)', 'Random Forest (Baseline)']
):
    ConfusionMatrixDisplay(confusion_matrix(y_test, pred),
                           display_labels=le.classes_).plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(title, fontweight='bold')
plt.suptitle('Confusion Matrices — Baseline', y=1.02)
plt.tight_layout()
plt.savefig('figures/cm_baseline.png', dpi=150, bbox_inches='tight')
plt.show()

# 3. FEATURE SELECTION — Correlation-Based Filter (threshold=0.9)
print("\n--- FEATURE SELECTION (Correlation-Based) ---")

corr_matrix = pd.DataFrame(X_train_scaled, columns=X.columns).corr().abs()

plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, linewidths=0.3)
plt.title('Feature Correlation Matrix (before selection)', fontweight='bold')
plt.tight_layout()
plt.savefig('figures/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
threshold = 0.9
to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
print(f"Features dropped (corr > {threshold}): {to_drop}")

selected_features = [col for col in X.columns if col not in to_drop]
n_fs = len(selected_features)
print(f"Features kept: {n_fs} / {X.shape[1]}")

X_train_fs = pd.DataFrame(X_train_scaled, columns=X.columns)[selected_features].values
X_test_fs  = pd.DataFrame(X_test_scaled,  columns=X.columns)[selected_features].values

lr_fs = LogisticRegression(max_iter=1000, random_state=42)
lr_fs.fit(X_train_fs, y_train)
lr_fs_pred = lr_fs.predict(X_test_fs)
lr_fs_acc  = accuracy_score(y_test, lr_fs_pred)

rf_fs = RandomForestClassifier(n_estimators=100, random_state=42)
rf_fs.fit(X_train_fs, y_train)
rf_fs_pred = rf_fs.predict(X_test_fs)
rf_fs_acc  = accuracy_score(y_test, rf_fs_pred)

print(f"Logistic Regression Accuracy: {lr_fs_acc:.4f}")
print(f"Random Forest Accuracy:       {rf_fs_acc:.4f}")

print("\nLogistic Regression Report:")
print(classification_report(y_test, lr_fs_pred, target_names=le.classes_))
print("Random Forest Report:")
print(classification_report(y_test, rf_fs_pred, target_names=le.classes_))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, pred, title in zip(
    axes,
    [lr_fs_pred, rf_fs_pred],
    ['Logistic Regression (Correlation FS)', 'Random Forest (Correlation FS)']
):
    ConfusionMatrixDisplay(confusion_matrix(y_test, pred),
                           display_labels=le.classes_).plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(title, fontweight='bold')
plt.suptitle('Confusion Matrices — Correlation Feature Selection', y=1.02)
plt.tight_layout()
plt.savefig('figures/cm_feature_selection.png', dpi=150, bbox_inches='tight')
plt.show()

# 4. FEATURE EXTRACTION — PCA (95% variance retained)
print("\n--- PCA ---")

pca_full = PCA(random_state=42)
pca_full.fit(X_train_scaled)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
n_components = int(np.argmax(cumvar >= 0.95) + 1)
print(f"Components to retain 95% variance: {n_components}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].bar(range(1, 11), pca_full.explained_variance_ratio_[:10] * 100, color='#1976D2')
axes[0].set_title('Scree Plot', fontweight='bold')
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Explained Variance (%)')

axes[1].plot(range(1, len(cumvar) + 1), cumvar * 100, color='#F44336', linewidth=2)
axes[1].axhline(95, linestyle='--', color='gray', label='95% threshold')
axes[1].axvline(n_components, linestyle='--', color='#4CAF50', label=f'n={n_components}')
axes[1].set_title('Cumulative Explained Variance', fontweight='bold')
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Variance (%)')
axes[1].legend()
plt.tight_layout()
plt.savefig('figures/pca_variance.png', dpi=150, bbox_inches='tight')
plt.show()

pca = PCA(n_components=n_components, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca  = pca.transform(X_test_scaled)
print(f"Reduced: {X_train_scaled.shape[1]} -> {X_train_pca.shape[1]} components")

fig, ax = plt.subplots(figsize=(8, 6))
for cls, color in zip(le.classes_, ['#2196F3', '#F44336', '#4CAF50']):
    idx = y_train == le.transform([cls])[0]
    ax.scatter(X_train_pca[idx, 0], X_train_pca[idx, 1],
               label=cls, alpha=0.4, s=15, color=color)
ax.set_title('PCA — First Two Principal Components', fontweight='bold')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.legend()
plt.tight_layout()
plt.savefig('figures/pca_scatter.png', dpi=150, bbox_inches='tight')
plt.show()

lr_pca = LogisticRegression(max_iter=1000, random_state=42)
lr_pca.fit(X_train_pca, y_train)
lr_pca_pred = lr_pca.predict(X_test_pca)
lr_pca_acc  = accuracy_score(y_test, lr_pca_pred)

rf_pca = RandomForestClassifier(n_estimators=100, random_state=42)
rf_pca.fit(X_train_pca, y_train)
rf_pca_pred = rf_pca.predict(X_test_pca)
rf_pca_acc  = accuracy_score(y_test, rf_pca_pred)

print(f"Logistic Regression Accuracy: {lr_pca_acc:.4f}")
print(f"Random Forest Accuracy:       {rf_pca_acc:.4f}")

print("\nLogistic Regression Report:")
print(classification_report(y_test, lr_pca_pred, target_names=le.classes_))
print("Random Forest Report:")
print(classification_report(y_test, rf_pca_pred, target_names=le.classes_))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, pred, title in zip(
    axes,
    [lr_pca_pred, rf_pca_pred],
    ['Logistic Regression (PCA)', 'Random Forest (PCA)']
):
    ConfusionMatrixDisplay(confusion_matrix(y_test, pred),
                           display_labels=le.classes_).plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(title, fontweight='bold')
plt.suptitle('Confusion Matrices — PCA', y=1.02)
plt.tight_layout()
plt.savefig('figures/cm_pca.png', dpi=150, bbox_inches='tight')
plt.show()

# 5. ACCURACY COMPARISON CHART
categories = [f'Baseline\n(34 features)', f'Correlation FS\n({n_fs} features)', 'PCA']
lr_scores  = [lr_base_acc * 100, lr_fs_acc * 100, lr_pca_acc * 100]
rf_scores  = [rf_base_acc * 100, rf_fs_acc * 100, rf_pca_acc * 100]

x     = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, lr_scores, width, label='Logistic Regression', color='#1976D2')
bars2 = ax.bar(x + width/2, rf_scores, width, label='Random Forest',       color='#388E3C')
ax.bar_label(bars1, fmt='%.2f%%', padding=3, fontsize=9)
ax.bar_label(bars2, fmt='%.2f%%', padding=3, fontsize=9)
ax.set_ylabel('Accuracy (%)')
ax.set_title('Model Accuracy Comparison: Baseline vs Correlation FS vs PCA', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylim(0, 105)
ax.legend()
plt.tight_layout()
plt.savefig('figures/accuracy_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# 6. FINAL SUMMARY
print("\n" + "=" * 65)
print("FINAL PERFORMANCE SUMMARY")
print("=" * 65)
results = [
    ("Logistic Regression", f"Baseline (34 features)",        lr_base_acc),
    ("Logistic Regression", f"Correlation FS ({n_fs} feats)", lr_fs_acc),
    ("Logistic Regression", "PCA",                            lr_pca_acc),
    ("Random Forest",       f"Baseline (34 features)",        rf_base_acc),
    ("Random Forest",       f"Correlation FS ({n_fs} feats)", rf_fs_acc),
    ("Random Forest",       "PCA",                            rf_pca_acc),
]
for model, config, acc in results:
    print(f"{model:<25} | {config:<30} | {acc*100:.2f}%")
print("=" * 65)
