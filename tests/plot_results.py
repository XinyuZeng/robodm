import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the data files
anime_df = pd.read_csv('/home/xinyu/robodm/data/res1028-anime-av1.csv')
res_df = pd.read_csv('/home/xinyu/robodm/data/res1028.csv')

# Filter res1028.csv for only av1 codec
av1_df = res_df[res_df['Codec'] == 'av1'].copy()

# Create figure with 2 rows and 3 columns (2 datasets, 3 metrics each)
fig, axes = plt.subplots(2, 3, figsize=(9, 6))

# Dataset 1: res1028-anime-av1.csv
dataset_name_1 = "res1028-anime-av1"
# Sort by interval for better plotting
anime_df_sorted = anime_df.sort_values('Interval')

# Plot 1: Size vs Interval
axes[0, 0].plot(anime_df_sorted['Interval'], anime_df_sorted['Size_MB'], 
                marker='o', linewidth=2, markersize=6)
axes[0, 0].set_xlabel('Interval', fontsize=11)
axes[0, 0].set_ylabel('Size (MB)', fontsize=11)
axes[0, 0].set_xlim(left=0)
axes[0, 0].set_ylim(bottom=0)
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Access Time vs Interval
axes[0, 1].plot(anime_df_sorted['Interval'], anime_df_sorted['Access_Time_s'], 
                marker='o', linewidth=2, markersize=6, color='orange')
axes[0, 1].set_xlabel('Interval', fontsize=11)
axes[0, 1].set_ylabel('Random Access Time (s)', fontsize=11)
axes[0, 1].set_xlim(left=0)
axes[0, 1].set_ylim(bottom=0)
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Whole File Read Time vs Interval
axes[0, 2].plot(anime_df_sorted['Interval'], anime_df_sorted['Whole_File_Read_Time_s'], 
                marker='o', linewidth=2, markersize=6, color='green')
axes[0, 2].set_xlabel('Interval', fontsize=11)
axes[0, 2].set_ylabel('Whole File Read Time (s)', fontsize=11)
axes[0, 2].set_xlim(left=0)
axes[0, 2].set_ylim(bottom=0)
axes[0, 2].grid(True, alpha=0.3)

# Dataset 2: res1028.csv (av1 only)
dataset_name_2 = "res1028 (AV1)"
# Sort by interval for better plotting
av1_df_sorted = av1_df.sort_values('Interval')

# Plot 4: Size vs Interval
axes[1, 0].plot(av1_df_sorted['Interval'], av1_df_sorted['Size_MB'], 
                marker='s', linewidth=2, markersize=6)
axes[1, 0].set_xlabel('Interval', fontsize=11)
axes[1, 0].set_ylabel('Size (MB)', fontsize=11)
axes[1, 0].set_xlim(left=0)
axes[1, 0].set_ylim(bottom=0)
axes[1, 0].grid(True, alpha=0.3)

# Plot 5: Access Time vs Interval
axes[1, 1].plot(av1_df_sorted['Interval'], av1_df_sorted['Access_Time_s'], 
                marker='s', linewidth=2, markersize=6, color='orange')
axes[1, 1].set_xlabel('Interval', fontsize=11)
axes[1, 1].set_ylabel('Random Access Time (s)', fontsize=11)
axes[1, 1].set_xlim(left=0)
axes[1, 1].set_ylim(bottom=0)
axes[1, 1].grid(True, alpha=0.3)

# Plot 6: Whole File Read Time vs Interval
axes[1, 2].plot(av1_df_sorted['Interval'], av1_df_sorted['Whole_File_Read_Time_s'], 
                marker='s', linewidth=2, markersize=6, color='green')
axes[1, 2].set_xlabel('Interval', fontsize=11)
axes[1, 2].set_ylabel('Whole File Read Time (s)', fontsize=11)
axes[1, 2].set_xlim(left=0)
axes[1, 2].set_ylim(bottom=0)
axes[1, 2].grid(True, alpha=0.3)

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the figure
plt.savefig('/home/xinyu/robodm/tests/plot_results.png', dpi=300, bbox_inches='tight')
print("Plot saved to: /home/xinyu/robodm/tests/plot_results.png")

# Display the plot
plt.show()

