# ── Step 2: Customer Segment Insights & Analysis ──────────────────────────────
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

ROOT     = Path(__file__).parent.parent
DATA_OUT = ROOT / 'data' / 'processed'
VISUALS  = ROOT / 'visuals'

# ── 1. Load segmented data ─────────────────────────────────────────────────────
df = pd.read_csv(DATA_OUT / 'customer_segments.csv')
print("Data loaded!")
print(f"   Total customers : {len(df)}")
print(f"   Segments        : {df['segment'].unique().tolist()}")
print()

# ── 2. Segment Profile Summary ─────────────────────────────────────────────────
print("=" * 60)
print("SEGMENT PROFILE SUMMARY")
print("=" * 60)
profile = df.groupby('segment').agg(
    total_customers   = ('customer_name', 'count'),
    avg_age           = ('estimated_age', 'mean'),
    avg_rating        = ('avg_rating', 'mean'),
    avg_total_ratings = ('total_ratings', 'mean'),
    top_genre         = ('top_genre', lambda x: x.value_counts().index[0])
).round(2)
print(profile.to_string())
print()

# ── 3. Insight 1 — Most Active Segment ────────────────────────────────────────
print("=" * 60)
print("INSIGHT 1 — Most Active Segment")
print("=" * 60)
activity    = df.groupby('segment')['total_ratings'].mean().sort_values(ascending=False)
most_active = activity.index[0]
print(f"   Most active segment : {most_active}")
print(f"   Avg movies rated    : {activity[most_active]:.1f}")
print()
for seg, val in activity.items():
    print(f"   {seg:<25} → {val:.1f} movies rated on average")
print()

# ── 4. Insight 2 — Highest Rating Segment ─────────────────────────────────────
print("=" * 60)
print("INSIGHT 2 — Most Generous Raters")
print("=" * 60)
ratings   = df.groupby('segment')['avg_rating'].mean().sort_values(ascending=False)
top_rater = ratings.index[0]
print(f"   Highest rating segment : {top_rater}")
print(f"   Avg rating             : {ratings[top_rater]:.2f} / 5.0")
print()
for seg, val in ratings.items():
    print(f"   {seg:<25} → {val:.2f} avg rating")
print()

# ── 5. Insight 3 — Segment Sizes ──────────────────────────────────────────────
print("=" * 60)
print("INSIGHT 3 — Segment Sizes")
print("=" * 60)
sizes = df['segment'].value_counts()
for seg, count in sizes.items():
    pct = count / len(df) * 100
    print(f"   {seg:<25} → {count} customers ({pct:.1f}%)")
print()

# ── 6. Insight 4 — Business Recommendations ───────────────────────────────────
print("=" * 60)
print("INSIGHT 4 — Business Recommendations")
print("=" * 60)
insights = {
    "Kids (5-12)"         : "Target with Animation, Fantasy & Adventure. High engagement potential.",
    "Teens (13-17)"       : "Push Comedy & Action. Social watchlist features boost retention.",
    "Young Adults (18-35)": "Drama & Thriller focused. Most likely to explore new genres.",
    "Adults (35+)"        : "Drama heavy. Prefer classic, well-rated content. Newsletter campaigns work well."
}
for seg, tip in insights.items():
    if seg in df['segment'].unique():
        print(f"\n   {seg}")
        print(f"   → {tip}")
print()

# ── 7. Plot — Segment Insights Dashboard ──────────────────────────────────────
colors = {
    'Kids (5-12)'         : '#3498DB',
    'Teens (13-17)'       : '#2ECC71',
    'Young Adults (18-35)': '#F39C12',
    'Adults (35+)'        : '#E74C3C'
}

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Step 2 — Customer Segment Insights', fontsize=16, fontweight='bold', y=1.01)

# Chart 1 — Avg rating per segment
avg_ratings = df.groupby('segment')['avg_rating'].mean().sort_values(ascending=False)
bars = axes[0, 0].bar(avg_ratings.index, avg_ratings.values,
                      color=[colors.get(s, '#95A5A6') for s in avg_ratings.index],
                      edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, avg_ratings.values):
    axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.2f}', ha='center', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Average Rating per Segment', fontsize=13, fontweight='bold')
axes[0, 0].set_xlabel('Segment')
axes[0, 0].set_ylabel('Average Rating (out of 5)')
axes[0, 0].set_ylim(0, 5.5)
axes[0, 0].tick_params(axis='x', rotation=15)
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Chart 2 — Activity (avg movies rated)
avg_activity = df.groupby('segment')['total_ratings'].mean().sort_values(ascending=False)
bars2 = axes[0, 1].bar(avg_activity.index, avg_activity.values,
                       color=[colors.get(s, '#95A5A6') for s in avg_activity.index],
                       edgecolor='white', linewidth=1.5)
for bar, val in zip(bars2, avg_activity.values):
    axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.0f}', ha='center', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Avg Movies Rated per Segment (Activity)', fontsize=13, fontweight='bold')
axes[0, 1].set_xlabel('Segment')
axes[0, 1].set_ylabel('Avg Number of Movies Rated')
axes[0, 1].tick_params(axis='x', rotation=15)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Chart 3 — Age distribution
for seg in df['segment'].unique():
    seg_data = df[df['segment'] == seg]['estimated_age']
    axes[1, 0].hist(seg_data, alpha=0.6, label=seg,
                    color=colors.get(seg, '#95A5A6'), bins=10, edgecolor='white')
axes[1, 0].set_title('Age Distribution per Segment', fontsize=13, fontweight='bold')
axes[1, 0].set_xlabel('Age')
axes[1, 0].set_ylabel('Number of Customers')
axes[1, 0].legend(fontsize=9)
axes[1, 0].grid(True, alpha=0.3)

# Chart 4 — Segment share (pie)
seg_counts = df['segment'].value_counts()
pie_colors = [colors.get(s, '#95A5A6') for s in seg_counts.index]
axes[1, 1].pie(seg_counts.values, labels=seg_counts.index,
               autopct='%1.1f%%', colors=pie_colors,
               startangle=90, pctdistance=0.85,
               wedgeprops=dict(edgecolor='white', linewidth=2))
axes[1, 1].set_title('Customer Share per Segment', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(VISUALS / 'plot4_segment_insights.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved: plot4_segment_insights.png")
print()

# ── 8. Plot — Rating & Engagement Deep Dive ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Step 2 — Rating & Engagement Deep Dive', fontsize=14, fontweight='bold')

seg_order  = sorted(df['segment'].unique())
box_colors = [colors.get(s, '#95A5A6') for s in seg_order]

bp = axes[0].boxplot(
    [df[df['segment'] == s]['avg_rating'].values for s in seg_order],
    labels=seg_order, patch_artist=True, notch=False
)
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[0].set_title('Rating Spread per Segment', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Segment')
axes[0].set_ylabel('Average Rating')
axes[0].tick_params(axis='x', rotation=15)
axes[0].grid(True, alpha=0.3, axis='y')

top3  = df.groupby(['segment', 'top_genre']).size().reset_index(name='count')
top3  = top3.sort_values('count', ascending=False).groupby('segment').head(3)
pivot = top3.pivot(index='segment', columns='top_genre', values='count').fillna(0)
pivot.plot(kind='bar', ax=axes[1], colormap='tab10', edgecolor='white', linewidth=0.5)
axes[1].set_title('Top 3 Genres per Segment', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Segment')
axes[1].set_ylabel('Number of Customers')
axes[1].tick_params(axis='x', rotation=15)
axes[1].legend(title='Genre', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(VISUALS / 'plot5_rating_engagement.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved: plot5_rating_engagement.png")
print()

# ── 9. Save insights summary ───────────────────────────────────────────────────
summary = df.groupby('segment').agg(
    total_customers  = ('customer_name', 'count'),
    avg_age          = ('estimated_age', 'mean'),
    avg_rating       = ('avg_rating', 'mean'),
    avg_movies_rated = ('total_ratings', 'mean'),
    top_genre        = ('top_genre', lambda x: x.value_counts().index[0]),
    most_active_user = ('customer_name', 'first')
).round(2).reset_index()
summary.to_csv(DATA_OUT / 'segment_insights.csv', index=False)
print("=" * 60)
print("Step 2 Complete!")
print("   data/processed/segment_insights.csv")
print("   visuals/plot4_segment_insights.png")
print("   visuals/plot5_rating_engagement.png")
print("=" * 60)
