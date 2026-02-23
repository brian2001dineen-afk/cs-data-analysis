import pandas as pd

df_map = pd.read_csv('https://raw.githubusercontent.com/brian2001dineen-afk/cs-data-analysis/refs/heads/main/datasets/maps_statistics.csv')
df_top = pd.read_csv('https://raw.githubusercontent.com/brian2001dineen-afk/cs-data-analysis/refs/heads/main/datasets/top_100_players.csv')
df_weapon = pd.read_csv('https://raw.githubusercontent.com/brian2001dineen-afk/cs-data-analysis/refs/heads/main/datasets/weapons_statistics.csv')
print(df_map)
print(df_top)
print(df_weapon)

df_map.info()
df_top.describe()
df_top.info()
cols = ['CS Rating', 'Wins']
df = df_top[cols]
df.describe()

# Sort by wins in descending order
df['Wins'].sort_values(ascending=False)
