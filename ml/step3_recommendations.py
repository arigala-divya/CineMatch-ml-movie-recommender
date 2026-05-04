# ── Step 3: Hybrid Movie Recommendation Engine ────────────────────────────────
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

ROOT     = Path(__file__).parent.parent
DATA_RAW = ROOT / 'data' / 'raw'
DATA_OUT = ROOT / 'data' / 'processed'
VISUALS  = ROOT / 'visuals'

print("\nLoading datasets...")
movies    = pd.read_csv(DATA_RAW  / 'movies.csv')
ratings   = pd.read_csv(DATA_RAW  / 'ratings.csv')
customers = pd.read_csv(DATA_OUT  / 'customer_segments.csv')

ratings.rename(columns={'userId': 'user_id'}, inplace=True)

print("Merging datasets...")
data = ratings.merge(movies, on='movieId')
data = data.merge(customers[['user_id', 'cluster', 'age_group']], on='user_id')


# ── Helper: compute hybrid score for a group of ratings ───────────────────────
def top_movies_for_group(group_data, n=20):
    stats = group_data.groupby('title').agg(
        avg_user_rating = ('rating', 'mean'),
        rating_count    = ('rating', 'count'),
        rating_std      = ('rating', 'std'),
    ).reset_index()
    stats.fillna(0, inplace=True)

    min_votes = max(5, int(stats['rating_count'].quantile(0.40)))
    stats = stats[stats['rating_count'] >= min_votes]
    stats = stats[stats['avg_user_rating'] > 3.0]

    if stats.empty:
        return stats

    C = stats['avg_user_rating'].mean()
    m = stats['rating_count'].quantile(0.60)

    stats['weighted_rating'] = stats.apply(
        lambda r: (r['rating_count'] / (r['rating_count'] + m) * r['avg_user_rating'])
                  + (m / (r['rating_count'] + m) * C),
        axis=1,
    )
    stats['popularity_score']  = stats['avg_user_rating'] * np.log1p(stats['rating_count'])
    stats['consistency_score'] = 1 / (1 + stats['rating_std'])

    for col in ['weighted_rating', 'popularity_score', 'consistency_score']:
        rng = stats[col].max() - stats[col].min()
        stats[col] = (stats[col] - stats[col].min()) / rng if rng > 0 else 0.0

    # Hybrid score: 50% weighted rating, 30% popularity, 20% consistency
    stats['hybrid_score'] = (
        0.5 * stats['weighted_rating']
        + 0.3 * stats['popularity_score']
        + 0.2 * stats['consistency_score']
    )

    return stats.sort_values('hybrid_score', ascending=False).head(n)


# ── Segment Recommendations (per cluster) ─────────────────────────────────────
print("\nGenerating segment recommendations...")
segment_outputs = []
for cl in sorted(data['cluster'].unique()):
    top = top_movies_for_group(data[data['cluster'] == cl])
    top['recommended_for_cluster'] = cl
    top['reason'] = 'Popular & highly rated within this segment'
    segment_outputs.append(top)

segment_df = pd.concat(segment_outputs)
segment_df.to_csv(DATA_OUT / 'segment_recommendations.csv', index=False)
print("Segment recommendations saved.")

# ── Age Group Recommendations ──────────────────────────────────────────────────
print("\nGenerating age group recommendations...")
age_outputs = []
for age in sorted(data['age_group'].unique()):
    top = top_movies_for_group(data[data['age_group'] == age])
    top['recommended_for_age_group'] = age
    top['reason'] = 'Trending among this age group'
    age_outputs.append(top)

age_df = pd.concat(age_outputs)
age_df.to_csv(DATA_OUT / 'age_group_recommendations.csv', index=False)
print("Age group recommendations saved.")

# ── Global Top Movies ──────────────────────────────────────────────────────────
print("\nGenerating global top movies...")
global_top = top_movies_for_group(data, n=30)
global_top['reason'] = 'Top movies overall'
global_top.to_csv(DATA_OUT / 'global_top_movies.csv', index=False)
print("Global recommendations saved.")

# ── Recommendation Insights Figure ────────────────────────────────────────────
print("\nGenerating recommendation insights figure...")
plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
plt.scatter(global_top['avg_user_rating'], global_top['rating_count'], alpha=0.7, color='steelblue')
plt.title('Rating vs Popularity')
plt.xlabel('Average Rating')
plt.ylabel('Number of Ratings')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.hist(global_top['hybrid_score'], bins=15, color='seagreen', edgecolor='white')
plt.title('Hybrid Score Distribution')
plt.xlabel('Hybrid Score')
plt.ylabel('Movies')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
plt.hist(global_top['rating_count'], bins=15, color='tomato', edgecolor='white')
plt.title('Movie Popularity Distribution')
plt.xlabel('Number of Ratings')
plt.ylabel('Movies')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
plt.hist(global_top['avg_user_rating'], bins=15, color='goldenrod', edgecolor='white')
plt.title('Average Rating Distribution')
plt.xlabel('Average Rating')
plt.ylabel('Movies')
plt.grid(True, alpha=0.3)

plt.suptitle('Step 3 — Recommendation Engine Insights', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(VISUALS / 'plot6_recommendation_insights.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "=" * 55)
print("Step 3 Complete!")
print("   data/processed/segment_recommendations.csv")
print("   data/processed/age_group_recommendations.csv")
print("   data/processed/global_top_movies.csv")
print("   visuals/plot6_recommendation_insights.png")
print("=" * 55 + "\n")
