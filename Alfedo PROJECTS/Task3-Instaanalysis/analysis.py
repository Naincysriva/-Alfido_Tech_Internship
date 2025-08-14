import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tkinter import Tk, filedialog

Tk().withdraw()  # hide tkinter root window

# Safe CSV loader
def load_csv_safe(filename):
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        print(f"⚠️ '{filename}' not found. Please select it manually...")
        path = filedialog.askopenfilename(
            title=f"Select {filename}",
            filetypes=[("CSV Files", "*.csv")]
        )
        if not path:
            print(f"❌ No file selected for {filename}. Exiting.")
            exit()
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()  # remove spaces from column names
    return df

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load datasets
likes_df = load_csv_safe("likes.csv")
photos_df = load_csv_safe("photos.csv")
photo_tags_df = load_csv_safe("photo_tags.csv")
tags_df = load_csv_safe("tags.csv")
users_df = load_csv_safe("users.csv")

print("✅ CSV files loaded successfully!\n")
print(f"📸 Total Photos: {len(photos_df)}")
print(f"❤️ Total Likes: {len(likes_df)}")
print(f"🏷️ Total Tags: {len(tags_df)}")
print(f"👤 Total Users: {len(users_df)}")

# ---- Top Liked Photos ----
like_counts = likes_df['photo'].value_counts().reset_index()
like_counts.columns = ['photo', 'like_count']
top_photos = pd.merge(photos_df, like_counts, left_on='id', right_on='photo')
top_10_liked = top_photos.sort_values(by='like_count', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='like_count', y='id', data=top_10_liked, hue='id', legend=False, palette='rocket')
plt.title("Top 10 Most Liked Instagram Photos")
plt.xlabel("Number of Likes")
plt.ylabel("Photo ID")
plt.tight_layout()
plt.show()

# ---- Most Used Hashtags ----
tagged_photos = photo_tags_df.merge(tags_df, left_on='tag ID', right_on='id')
top_tags = tagged_photos['tag text'].value_counts().head(10).reset_index()
top_tags.columns = ['Hashtag', 'Count']

plt.figure(figsize=(10, 6))
sns.barplot(x='Count', y='Hashtag', data=top_tags, hue='Hashtag', legend=False, palette='viridis')
plt.title("Top 10 Most Used Instagram Hashtags")
plt.tight_layout()
plt.show()

# ---- Most Active Users ----
user_post_counts = photos_df['user ID'].value_counts().head(10).reset_index()
user_post_counts.columns = ['user ID', 'post_count']
active_users = pd.merge(user_post_counts, users_df, left_on='user ID', right_on='id')

plt.figure(figsize=(10, 6))
sns.barplot(x='post_count', y='user ID', data=active_users, hue='user ID', legend=False, palette='magma')
plt.title("Top 10 Most Active Users (By Photo Uploads)")
plt.xlabel("Number of Photos Uploaded")
plt.ylabel("User ID")
plt.tight_layout()
plt.show()
