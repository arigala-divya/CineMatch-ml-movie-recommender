# ── Step 1: Customer Segmentation using K-Means Clustering ────────────────────
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

ROOT     = Path(__file__).parent.parent
DATA_RAW = ROOT / 'data' / 'raw'
DATA_OUT = ROOT / 'data' / 'processed'
VISUALS  = ROOT / 'visuals'
DATA_OUT.mkdir(parents=True, exist_ok=True)
VISUALS.mkdir(parents=True, exist_ok=True)

# ── 1. Load Data ───────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_RAW / 'customers.csv')
print("Data loaded successfully!")
print(f"   Total customers : {len(df)}")
print(f"   Columns         : {df.columns.tolist()}")
print()

# ── 2. Prepare features for clustering ────────────────────────────────────────
le = LabelEncoder()
df['genre_encoded'] = le.fit_transform(df['top_genre'])

features = df[['estimated_age', 'avg_rating',
               'total_ratings', 'genre_encoded']].copy()

# ── 3. Normalise features ──────────────────────────────────────────────────────
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(features)
print("Features normalised!")
print()

# ── 4. Find best K — Elbow + Silhouette ───────────────────────────────────────
print("Finding best K...")
inertias   = []
sil_scores = []
K_range    = range(2, 10)

for k in K_range:
    km  = KMeans(n_clusters=k, random_state=42, n_init=10)
    lbl = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, lbl))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Step 1 — Finding Optimal Number of Clusters', fontsize=14, fontweight='bold')

axes[0].plot(list(K_range), inertias, 'o-', color='steelblue', linewidth=2, markersize=8)
axes[0].axvline(x=4, color='tomato', linestyle='--', linewidth=2, label='Chosen K=4')
axes[0].set_title('Elbow Method')
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('Inertia')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(list(K_range), sil_scores, 's-', color='seagreen', linewidth=2, markersize=8)
axes[1].axvline(x=4, color='tomato', linestyle='--', linewidth=2, label='Chosen K=4')
axes[1].set_title('Silhouette Score')
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(VISUALS / 'plot1_elbow_silhouette.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved: plot1_elbow_silhouette.png")
print()

# ── 5. Run Final K-Means with K=4 ─────────────────────────────────────────────
print("Running K-Means with K=4...")
kmeans    = KMeans(n_clusters=4, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Map clusters to age groups by sorting on average estimated_age
age_group_order = ['Kids (5-12)', 'Teens (13-17)', 'Young Adults (18-35)', 'Adults (35+)']
cluster_avg_age = df.groupby('cluster')['estimated_age'].mean().sort_values()
cluster_age_map = {cluster: age_group_order[i] for i, cluster in enumerate(cluster_avg_age.index)}
print("   Cluster → Age Group mapping (by avg age):")
for c, ag in cluster_age_map.items():
    print(f"   Cluster {c} → {ag}  (avg age: {cluster_avg_age[c]:.1f})")
df['segment'] = df['cluster'].map(cluster_age_map)
print()

# ── 6. Segment Profiles ────────────────────────────────────────────────────────
print("=" * 55)
print("SEGMENT PROFILES")
print("=" * 55)
profile = df.groupby('segment').agg(
    total_customers   = ('customer_name', 'count'),
    avg_age           = ('estimated_age', 'mean'),
    avg_rating        = ('avg_rating', 'mean'),
    avg_total_ratings = ('total_ratings', 'mean'),
    top_genre         = ('top_genre', lambda x: x.value_counts().index[0])
).round(2)
print(profile.to_string())
print()

# ── 7. Visualise Clusters ──────────────────────────────────────────────────────
pca  = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_scaled)

colors     = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
segments   = df['segment'].unique()
seg_colors = {seg: colors[i] for i, seg in enumerate(sorted(segments))}

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Step 2 — Customer Segments Visualised', fontsize=14, fontweight='bold')

for seg in sorted(segments):
    mask = df['segment'] == seg
    axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1],
                    c=seg_colors[seg], label=seg, alpha=0.7, s=70,
                    edgecolors='white', linewidth=0.5)
axes[0].set_title('Customer Clusters (PCA 2D View)')
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
axes[0].legend(title='Segment', fontsize=9)
axes[0].grid(True, alpha=0.3)

seg_counts = df['segment'].value_counts()
bars = axes[1].bar(seg_counts.index, seg_counts.values,
                   color=[seg_colors[s] for s in seg_counts.index],
                   edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, seg_counts.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 str(val), ha='center', fontsize=11, fontweight='bold')
axes[1].set_title('Number of Customers per Segment')
axes[1].set_xlabel('Segment')
axes[1].set_ylabel('Customer Count')
axes[1].tick_params(axis='x', rotation=15)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(VISUALS / 'plot2_segments.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved: plot2_segments.png")
print()

# ── 8. Genre breakdown per segment ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
fig.suptitle('Step 3 — Top Genre per Segment', fontsize=14, fontweight='bold')

genre_seg = df.groupby(['segment', 'top_genre'], dropna=False).size().reset_index(name='count')
top_genres_per_seg = genre_seg.sort_values('count', ascending=False).groupby('segment').head(4)

seg_order = sorted(df['segment'].unique())
sns.barplot(data=top_genres_per_seg, x='segment', y='count', hue='top_genre',
            order=seg_order, ax=ax)
ax.set_title('Genre Preferences by Customer Segment')
ax.set_xlabel('Segment')
ax.set_ylabel('Number of Customers')
ax.legend(title='Genre', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
ax.tick_params(axis='x', rotation=15)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(VISUALS / 'plot3_genre_segments.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved: plot3_genre_segments.png")
print()

# ── 9. Save final segmented data ───────────────────────────────────────────────
df.to_csv(DATA_OUT / 'customer_segments.csv', index=False)
print("=" * 55)
print("Step 1 Complete!")
print("   data/processed/customer_segments.csv")
print("   visuals/plot1_elbow_silhouette.png")
print("   visuals/plot2_segments.png")
print("   visuals/plot3_genre_segments.png")
print("=" * 55)
