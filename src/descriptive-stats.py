import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go


# Payout Data
df = pd.read_csv('data/raw/payouts.csv')
geoplot = False
payout_timeplot = False
# Report initial row count
initial_rows = len(df)

# Drop rows with missing values in the Amount (USD) column
df = df.dropna(subset=['Amount (USD)'])

# Convert Amount (USD) from '$4,282,733' format to integer
df['Amount (USD)'] = df['Amount (USD)'].str.replace('$', '').str.replace(',', '').astype(float)

# Report dropped rows
dropped_rows = initial_rows - len(df)
print(f"Number of rows dropped: {dropped_rows}")

# Report statistics
print(f"Number of rows (country-year observations): {len(df)}")
print(f"Number of unique Countries: {df['Country'].nunique()}")
print(f"Number of unique Years: {df['Year'].nunique()}")
print(f"Number of unique Country-Years: {df.groupby(['Country', 'Year']).ngroups}")
print(f"Maximum payout amount: {df['Amount (USD)'].max()}")
print(f"Minimum payout amount: {df['Amount (USD)'].min()}")
print(f"Average payout amount: {df['Amount (USD)'].mean()}")
print(f"Median payout amount: {df['Amount (USD)'].median()}")
print(f"Standard deviation of payout amount: {df['Amount (USD)'].std()}")
top5_table = (
    df.groupby('Country', as_index=False)
      .size()
      .rename(columns={'size': 'payout_events'})
      .sort_values('payout_events', ascending=False)
      .head(5)
)

print(top5_table)

# Basic Plots

if payout_timeplot:
    # Filter rows that have both month and year values
    df_with_date = df[df['Month'].notna() & df['Year'].notna()].copy()
    df_with_date['Month-Year'] = pd.to_datetime(
        df_with_date['Month'].astype(int).astype(str) + '-' + df_with_date['Year'].astype(int).astype(str),
        format='%m-%Y')
    df_with_date = df_with_date.sort_values('Month-Year')

    # Plot
    plt.figure(figsize=(12, 6))
    plt.scatter(df_with_date['Month-Year'], df_with_date['Amount (USD)'])
    plt.xlabel('Month-Year')
    plt.ylabel('Amount (USD)')
    plt.title('Payouts Over Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Plot
    plt.figure(figsize=(12, 6))
    for pool in df_with_date['Pool'].unique():
        mask = df_with_date['Pool'] == pool
        plt.scatter(df_with_date[mask]['Month-Year'], df_with_date[mask]['Amount (USD)'], label=pool, alpha=0.6)
    plt.xlabel('Month-Year')
    plt.ylabel('Amount (USD)')
    plt.title('Payouts Over Time')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if geoplot:
    # Geo plot without geopandas or plotly.express:
    country_pool = (
        df.groupby(['Country', 'Pool'], as_index=False)['Amount (USD)']
        .sum()
    )
    country_pool['Country'] = country_pool['Country'].str.strip()

    # Color map for pools
    pools = sorted(country_pool['Pool'].dropna().unique())
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    color_map = {p: palette[i % len(palette)] for i, p in enumerate(pools)}

    # Bubble size scaling (area-like)
    max_amt = country_pool['Amount (USD)'].max()
    country_pool['bubble_size'] = (country_pool['Amount (USD)'] / max_amt).pow(0.5) * 40

    fig = go.Figure()

    for p in pools:
        d = country_pool[country_pool['Pool'] == p]
        fig.add_trace(go.Scattergeo(
            locationmode='country names',
            locations=d['Country'],
            text=d['Country'],
            customdata=d['Amount (USD)'],
            name=str(p),
            mode='markers',
            marker=dict(
                size=d['bubble_size'],
                color=color_map[p],
                opacity=0.75,
                line=dict(width=0.5, color='white')
            ),
            hovertemplate="<b>%{text}</b><br>Pool: " + str(p) +
                        "<br>Total payout: $%{customdata:,.0f}<extra></extra>"
        ))

    fig.update_layout(
        title='Total Payouts by Country (size) and Pool (color)',
        geo=dict(projection_type='natural earth', showland=True, landcolor='rgb(243,243,243)'),
        legend_title='Pool'
    )

    fig.show()

# World Bank Data: 1
# worldbank_goods_services_expense_lcu_usd

goods_services = pd.read_csv('data/processed/worldbank_goods_services_expense_lcu_usd.csv')
print(goods_services.head())

# Report statistics
print(f"Number of rows (country-year observations): {len(goods_services)}")
print(f"Number of unique Countries: {goods_services['country'].nunique()}")
print(f"Number of unique Years: {goods_services['year'].nunique()}")
print(f"Number of unique Country-Years: {goods_services.groupby(['country', 'year']).ngroups}")
print(f"Maximum goods and services expense (USD): {goods_services['goods_services_expense_usd'].max()}")
print(f"Minimum goods and services expense (USD): {goods_services['goods_services_expense_usd'].min()}")
print(f"Average goods and services expense (USD): {goods_services['goods_services_expense_usd'].mean()}")
print(f"Median goods and services expense (USD): {goods_services['goods_services_expense_usd'].median()}")
print(f"Standard deviation of goods and services expense (USD): {goods_services['goods_services_expense_usd'].std()}")  

# World Bank Data: 2
# worldbank_gov_current_consumption_usd
gov_consumption = pd.read_csv('data/processed/worldbank_gov_current_consumption_usd.csv')
print(gov_consumption.head())

# Report statistics
print(f"Number of rows (country-year observations): {len(gov_consumption)}")
print(f"Number of unique Countries: {gov_consumption['country'].nunique()}")
print(f"Number of unique Years: {gov_consumption['year'].nunique()}")
print(f"Number of unique Country-Years: {gov_consumption.groupby(['country', 'year']).ngroups}")
print(f"Maximum government current consumption (USD): {gov_consumption['gov_consumption_usd'].max()}")
print(f"Minimum government current consumption (USD): {gov_consumption['gov_consumption_usd'].min()}")
print(f"Average government current consumption (USD): {gov_consumption['gov_consumption_usd'].mean()}")
print(f"Median government current consumption (USD): {gov_consumption['gov_consumption_usd'].median()}")
print(f"Standard deviation of government current consumption (USD): {gov_consumption['gov_consumption_usd'].std()}")  

