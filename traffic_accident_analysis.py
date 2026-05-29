import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap

plt.style.use('seaborn-v0_8-whitegrid') 


def load_data(path=None):
    # Try common filenames or use path provided
    candidates = []
    if path:
        candidates.append(path)
    candidates += [
        'accidents.csv',
        'traffic_accidents.csv',
        os.path.join(os.path.dirname(__file__), 'accidents.csv')
    ]

    for p in candidates:
        if p and os.path.exists(p):
            df = pd.read_csv(p)
            print(f'Loaded: {p} -> {df.shape[0]} rows')
            return df

    # If no file found, create a synthetic example dataset
    print('No dataset found; creating synthetic sample data for demonstration.')
    rng = np.random.default_rng(0)
    n = 2000
    df = pd.DataFrame({
        'datetime': pd.to_datetime('2021-01-01') + pd.to_timedelta(rng.integers(0, 365*24, n), unit='h'),
        'latitude': 37.6 + rng.normal(0, 0.05, n),
        'longitude': -122.4 + rng.normal(0, 0.05, n),
        'road_surface': rng.choice(['Dry', 'Wet', 'Snow', 'Ice', 'Unknown'], n, p=[0.7,0.2,0.05,0.03,0.02]),
        'weather': rng.choice(['Clear','Rain','Fog','Snow','Overcast'], n, p=[0.6,0.2,0.05,0.05,0.1]),
        'severity': rng.choice([1,2,3,4], n, p=[0.5,0.3,0.15,0.05])
    })
    return df


def preprocess(df):
    df = df.copy()
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df = df.dropna(subset=['datetime'])
        df['hour'] = df['datetime'].dt.hour
        df['date'] = df['datetime'].dt.date
        df['dow'] = df['datetime'].dt.day_name()
    if 'road_surface' in df.columns:
        df['road_surface'] = df['road_surface'].fillna('Unknown')
    if 'weather' in df.columns:
        df['weather'] = df['weather'].fillna('Unknown')
    return df


def time_of_day_analysis(df, save_fig=True):
    ct = df.groupby(['hour']).size()
    plt.figure(figsize=(10,4))
    sns.barplot(x=ct.index, y=ct.values, color='C0')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Accidents')
    plt.title('Accidents by Hour of Day')
    plt.tight_layout()
    if save_fig: plt.savefig('accidents_by_hour.png') 

    # stacked by severity
    if 'severity' in df.columns:
        pivot = df.groupby(['hour','severity']).size().unstack(fill_value=0)
        pivot.plot(kind='area', stacked=True, figsize=(10,4), colormap='tab20')
        plt.xlabel('Hour')
        plt.ylabel('Accidents')
        plt.title('Accidents by Hour and Severity')
        plt.tight_layout()
        if save_fig: plt.savefig('accidents_by_hour_severity.png')


def weather_road_analysis(df, save_fig=True):
    plt.figure(figsize=(8,5))
    order = df['road_surface'].value_counts().index
    sns.countplot(data=df, x='road_surface', order=order)
    plt.title('Accidents by Road Surface')
    plt.tight_layout()
    if save_fig: plt.savefig('accidents_by_road_surface.png')

    plt.figure(figsize=(8,5))
    order = df['weather'].value_counts().index
    sns.countplot(data=df, x='weather', order=order)
    plt.title('Accidents by Weather')
    plt.tight_layout()
    if save_fig: plt.savefig('accidents_by_weather.png')

    # Cross analysis
    ct = pd.crosstab(df['weather'], df['road_surface'])
    plt.figure(figsize=(8,6))
    sns.heatmap(ct, annot=True, fmt='d', cmap='YlGnBu')
    plt.title('Weather vs Road Surface (counts)')
    if save_fig: plt.savefig('weather_vs_road_surface.png')


def plot_hotspots_as_image(df, output_image='accident_hotspots.png'):
    plt.figure(figsize=(10, 8))
    
    # 1. Create a density/heatmap layer using Seaborn
    sns.kdeplot(
        data=df, x='longitude', y='latitude', 
        fill=True, thresh=0.05, cmap='Reds', alpha=0.6
    )

    # 2. Overlay the severe accident points
    severe = df[df.get('severity', 0) >= 3]
    plt.scatter(
        severe['longitude'], severe['latitude'], 
        color='darkred', s=15, label='Severe Accidents', alpha=0.8
    )
    
    plt.title('Accident Hotspots Density Map')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    # Save directly as a picture
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Static map image saved to {output_image}")
    
def contributing_factors(df):
    factors = []
    if 'weather' in df.columns:
        factors.append(('weather', df['weather'].value_counts(normalize=True)))
    if 'road_surface' in df.columns:
        factors.append(('road_surface', df['road_surface'].value_counts(normalize=True)))
    if 'hour' in df.columns:
        factors.append(('hour', df.groupby('hour').size().sort_values(ascending=False)))
    return factors


def main(data_path=None):
    df = load_data(data_path)
    df = preprocess(df)
    time_of_day_analysis(df)
    weather_road_analysis(df)
    plot_hotspots_as_image(df)
    facts = contributing_factors(df)
    print('\nTop contributing factors (by relative frequency):')
    for name, series in facts:
        print(f'-- {name} --')
        print(series.head(5).to_string())


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    main(path)
