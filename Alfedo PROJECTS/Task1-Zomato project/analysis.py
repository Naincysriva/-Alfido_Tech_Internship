
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import matplotlib

print("=" * 60)
print("🍽️  ZOMATO DATA ANALYSIS REPORT".center(60))
print("=" * 60)

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")

csv_path = r'C:\Users\ACER\Desktop\zomato project\zomato.csv'
df = pd.read_csv(csv_path, encoding='utf-8')  # Fix for garbled characters
print("✅ Dataset loaded. Columns:", df.columns.tolist())

df.drop_duplicates(inplace=True)

if 'rate' in df.columns:
    df['rate'] = df['rate'].astype(str).str.extract(r'(\d+\.\d+)')
    df['rate'] = pd.to_numeric(df['rate'], errors='coerce')

cost_col = 'approx_cost(for two people)'
if cost_col in df.columns:
    df[cost_col] = df[cost_col].astype(str).str.replace(',', '')
    df[cost_col] = pd.to_numeric(df[cost_col], errors='coerce')

cuisine_col = next((c for c in df.columns if 'cuisines' in c.lower()), None)
if cuisine_col:
    df.dropna(subset=[cuisine_col], inplace=True)

def save_plot(fig, fname):
    fig.tight_layout()
    fig.savefig(fname)
    print(f"✅ Saved: {fname}")
    plt.close(fig)

# Top 10 Cuisines
if cuisine_col:
    top_cuisines = df[cuisine_col].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top_cuisines.values, y=top_cuisines.index, ax=ax, palette='Blues_d')
    ax.set(title="Top 10 Cuisines", xlabel="Number of Restaurants", ylabel="Cuisine")
    save_plot(fig, 'top_cuisines.png')

# Word Cloud of Cuisines
if cuisine_col:
    text = ' '.join(df[cuisine_col].dropna())
    wc = WordCloud(width=1000, height=500, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title("Word Cloud of Cuisines")
    save_plot(fig, 'cuisine_wordcloud.png')

#  Rating Distribution
if 'rate' in df.columns:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df['rate'].dropna(), bins=20, kde=True, ax=ax, color='skyblue')
    ax.set(title="Restaurant Ratings Distribution", xlabel="Rating", ylabel="Count")
    save_plot(fig, 'rating_distribution_cleaned.png')

#  Cost Distribution
if cost_col in df.columns:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df[cost_col].dropna(), bins=30, kde=True, ax=ax, color='teal')
    ax.set(title="Cost for Two People Distribution", xlabel="Approx. Cost", ylabel="Count")
    save_plot(fig, 'cost_distribution_cleaned.png')

#  Top Restaurant Types
if 'rest_type' in df.columns:
    top_types = df['rest_type'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top_types.values, y=top_types.index, ax=ax, palette='Set3')
    ax.set(title="Top 10 Restaurant Types", xlabel="Count", ylabel="Type")
    save_plot(fig, 'top_rest_types.png')

# Table Booking Availability (Cleaned)
if 'book_table' in df.columns:
    valid_booking = df[df['book_table'].isin(['Yes', 'No'])]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x='book_table', data=valid_booking, ax=ax, palette='Set2')
    ax.set(title="Table Booking Availability", xlabel="Book Table", ylabel="Count")
    save_plot(fig, 'table_booking.png')

# Votes Distribution
if 'votes' in df.columns:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['votes'], bins=30, kde=True, ax=ax, color='orange')
    ax.set(title="User Votes Distribution", xlabel="Votes", ylabel="Count")
    save_plot(fig, 'votes_distribution.png')

# High-Rated Locations (>4.0)
if 'rate' in df.columns and 'location' in df.columns:
    high_rated = df[df['rate'] > 4.0]
    top_loc = high_rated['location'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top_loc.values, y=top_loc.index, ax=ax, palette='coolwarm')
    ax.set(title="Top Locations with Highly Rated Restaurants (>4.0)", xlabel="Count", ylabel="Location")
    save_plot(fig, 'top_high_rated_locations.png')

# Online Order Pie Chart
if 'online_order' in df.columns:
    valid_orders = df[df['online_order'].isin(['Yes', 'No'])]
    counts = valid_orders['online_order'].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140, colors=['#3498db', '#e67e22'])
    ax.set_title("Online Order Availability")
    save_plot(fig, 'online_order_pie.png')


