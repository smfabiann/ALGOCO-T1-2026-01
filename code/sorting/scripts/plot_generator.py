#!/usr/bin/env python3
"""
Script para generar gráficas comparativas de los algoritmos de sorting.
Compara: tiempo de ejecución, uso de memoria, y comportamiento según el tipo de datos.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Configuración de estilos
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# Rutas
DATA_DIR = Path(__file__).parent.parent / "data" / "measurements"
PLOTS_DIR = Path(__file__).parent.parent / "data" / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Colores para los algoritmos
COLORS = {
    'mergesort': '#2E86AB',
    'quicksort': '#A23B72',
    'sort': '#F18F01'
}

MARKERS = {
    'mergesort': 'o',
    'quicksort': 's',
    'sort': '^'
}


def load_data():
    """Carga los datos de los tres algoritmos."""
    data = {}
    
    for algo in ['mergesort', 'quicksort', 'sort']:
        csv_file = DATA_DIR / f'{algo}.csv'
        if csv_file.exists():
            data[algo] = pd.read_csv(csv_file)
        else:
            print(f"⚠️  Archivo no encontrado: {csv_file}")
    
    return data


def combine_data(data):
    """Combina los datos de todos los algoritmos en un DataFrame único."""
    combined = []
    
    for algo, df in data.items():
        df_copy = df.copy()
        df_copy['algoritmo'] = algo
        combined.append(df_copy)
    
    return pd.concat(combined, ignore_index=True)


def plot_time_vs_size(data_dict, combined_df):
    """Gráfica 1: Tiempo de ejecución vs Tamaño de entrada (escala logarítmica)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for algo in ['mergesort', 'quicksort', 'sort']:
        if algo in data_dict:
            df = data_dict[algo]
            size_mean = df.groupby('size')['time_ms'].mean()
            size_std = df.groupby('size')['time_ms'].std()
            
            ax.errorbar(
                size_mean.index, size_mean.values,
                yerr=size_std.values,
                label=algo.capitalize(),
                marker=MARKERS[algo],
                color=COLORS[algo],
                linewidth=2,
                markersize=8,
                capsize=5,
                alpha=0.8
            )
    
    ax.set_xlabel('Tamaño de Entrada (elementos)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Tiempo de Ejecución (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Tiempo vs Tamaño - Escala Logarítmica', fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / '01_tiempo_vs_tamaño.png', dpi=300, bbox_inches='tight')
    print("✓ Guardada: 01_tiempo_vs_tamaño.png")
    plt.close()


def plot_time_by_data_type(combined_df):
    """Gráfica 2: Tiempo por tipo de dato (ascendente, descendente, aleatorio)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    tipos = ['ascendente', 'descendente', 'aleatorio']
    
    for idx, tipo in enumerate(tipos):
        ax = axes[idx]
        df_filtered = combined_df[combined_df['tipo'] == tipo]
        
        for algo in ['mergesort', 'quicksort', 'sort']:
            df_algo = df_filtered[df_filtered['algoritmo'] == algo]
            if len(df_algo) > 0:
                size_mean = df_algo.groupby('size')['time_ms'].mean()
                
                ax.plot(
                    size_mean.index, size_mean.values,
                    label=algo.capitalize(),
                    marker=MARKERS[algo],
                    color=COLORS[algo],
                    linewidth=2.5,
                    markersize=8,
                    alpha=0.8
                )
        
        ax.set_xlabel('Tamaño de Entrada', fontsize=11, fontweight='bold')
        ax.set_ylabel('Tiempo (ms)', fontsize=11, fontweight='bold')
        ax.set_title(f'Datos {tipo.capitalize()}', fontsize=12, fontweight='bold')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which='both')
    
    plt.suptitle('Comparación de Tiempo por Tipo de Dato', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / '02_tiempo_por_tipo_dato.png', dpi=300, bbox_inches='tight')
    print("✓ Guardada: 02_tiempo_por_tipo_dato.png")
    plt.close()


def plot_memory_usage(combined_df):
    """Gráfica 3: Uso de memoria vs Tamaño (mediana + IQR, escala symlog)."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    memory_metrics = [
        ('delta_rss_kb', 'Delta RSS (KB)', axes[0, 0]),
        ('peak_kb', 'Peak Memory (KB)', axes[0, 1]),
        ('delta_peak_from_start_kb', 'Delta Peak desde Inicio (KB)', axes[1, 0]),
        ('rss_after_kb', 'RSS Después (KB)', axes[1, 1])
    ]
    
    for metric, title, ax in memory_metrics:
        metric_positive = combined_df[combined_df[metric] > 0][metric]
        if len(metric_positive) > 0:
            linthresh = max(1.0, float(np.nanpercentile(metric_positive, 20)))
        else:
            linthresh = 1.0

        for algo in ['mergesort', 'quicksort', 'sort']:
            df_algo = combined_df[combined_df['algoritmo'] == algo]
            if len(df_algo) > 0:
                grouped = df_algo.groupby('size')[metric]
                size_median = grouped.median()
                size_q1 = grouped.quantile(0.25)
                size_q3 = grouped.quantile(0.75)

                x_vals = size_median.index.values
                y_median = size_median.values
                y_q1 = size_q1.values
                y_q3 = size_q3.values

                ax.fill_between(
                    x_vals,
                    y_q1,
                    y_q3,
                    color=COLORS[algo],
                    alpha=0.15
                )
                
                ax.plot(
                    x_vals, y_median,
                    label=algo.capitalize(),
                    marker=MARKERS[algo],
                    color=COLORS[algo],
                    linewidth=2.5,
                    markersize=8,
                    alpha=0.8
                )
        
        ax.set_xlabel('Tamaño de Entrada', fontsize=11, fontweight='bold')
        ax.set_ylabel(title, fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xscale('log')
        ax.set_yscale('symlog', linthresh=linthresh)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which='both')
        ax.text(
            0.02,
            0.95,
            'Linea: mediana | Banda: IQR',
            transform=ax.transAxes,
            fontsize=8,
            va='top',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )
    
    plt.suptitle('Comparación de Uso de Memoria (Escala symlog)', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / '03_memoria_vs_tamaño.png', dpi=300, bbox_inches='tight')
    print("✓ Guardada: 03_memoria_vs_tamaño.png")
    plt.close()


def plot_efficiency_ratio(combined_df):
    """Gráfica 4: Relación tiempo/memoria (eficiencia)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Eficiencia: tiempo / delta_rss
    ax = axes[0]
    for algo in ['mergesort', 'quicksort', 'sort']:
        df_algo = combined_df[combined_df['algoritmo'] == algo]
        if len(df_algo) > 0:
            df_algo = df_algo[df_algo['delta_rss_kb'] > 0]  # Evitar división por cero
            df_algo['efficiency'] = df_algo['time_ms'] / df_algo['delta_rss_kb']
            size_mean = df_algo.groupby('size')['efficiency'].mean()
            
            ax.plot(
                size_mean.index, size_mean.values,
                label=algo.capitalize(),
                marker=MARKERS[algo],
                color=COLORS[algo],
                linewidth=2.5,
                markersize=8,
                alpha=0.8
            )
    
    ax.set_xlabel('Tamaño de Entrada', fontsize=11, fontweight='bold')
    ax.set_ylabel('Tiempo/Delta RSS (ms/KB)', fontsize=11, fontweight='bold')
    ax.set_title('Eficiencia Tiempo/Memoria', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Relación para tamaño más grande
    ax = axes[1]
    large_size = combined_df[combined_df['size'] == combined_df['size'].max()]
    
    algos = ['mergesort', 'quicksort', 'sort']
    times = [large_size[large_size['algoritmo'] == algo]['time_ms'].mean() for algo in algos]
    mems = [large_size[large_size['algoritmo'] == algo]['delta_rss_kb'].mean() for algo in algos]
    
    x_pos = np.arange(len(algos))
    width = 0.35
    
    bars1 = ax.bar(x_pos - width/2, times, width, label='Tiempo (ms)', 
                   color=[COLORS[algo] for algo in algos], alpha=0.7)
    ax2 = ax.twinx()
    bars2 = ax2.bar(x_pos + width/2, mems, width, label='Memoria (KB)', 
                    color=[COLORS[algo] for algo in algos], alpha=0.4)
    
    ax.set_xlabel('Algoritmo', fontsize=11, fontweight='bold')
    ax.set_ylabel('Tiempo (ms)', fontsize=11, fontweight='bold', color='black')
    ax2.set_ylabel('Memoria (KB)', fontsize=11, fontweight='bold', color='gray')
    ax.set_title(f'Comparativa para Tamaño = {combined_df["size"].max()}', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([algo.capitalize() for algo in algos])
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / '04_eficiencia.png', dpi=300, bbox_inches='tight')
    print("✓ Guardada: 04_eficiencia.png")
    plt.close()


def plot_domain_comparison(combined_df):
    """Gráfica 5: Comparación por dominio (D1 vs D7)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    dominios = ['D1', 'D7']
    
    for idx, dominio in enumerate(dominios):
        ax = axes[idx]
        df_filtered = combined_df[combined_df['dominio'] == dominio]
        
        for algo in ['mergesort', 'quicksort', 'sort']:
            df_algo = df_filtered[df_filtered['algoritmo'] == algo]
            if len(df_algo) > 0:
                size_mean = df_algo.groupby('size')['time_ms'].mean()
                
                ax.plot(
                    size_mean.index, size_mean.values,
                    label=algo.capitalize(),
                    marker=MARKERS[algo],
                    color=COLORS[algo],
                    linewidth=2.5,
                    markersize=8,
                    alpha=0.8
                )
        
        ax.set_xlabel('Tamaño de Entrada', fontsize=11, fontweight='bold')
        ax.set_ylabel('Tiempo (ms)', fontsize=11, fontweight='bold')
        ax.set_title(f'Dominio {dominio}', fontsize=12, fontweight='bold')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, which='both')
    
    plt.suptitle('Comparación por Dominio de Datos', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / '05_comparar_dominios.png', dpi=300, bbox_inches='tight')
    print("✓ Guardada: 05_comparar_dominios.png")
    plt.close()



def plot_box_plot_comparison(combined_df):
    """Gráfica 7: Box plots comparativos."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Por tipo de dato
    ax = axes[0, 0]
    sns.boxplot(data=combined_df, x='tipo', y='time_ms', hue='algoritmo', ax=ax, palette=COLORS)
    ax.set_xlabel('Tipo de Dato', fontsize=11, fontweight='bold')
    ax.set_ylabel('Tiempo (ms)', fontsize=11, fontweight='bold')
    ax.set_title('Distribución de Tiempos por Tipo de Dato', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(title='Algoritmo')
    
    # Por dominio
    ax = axes[0, 1]
    sns.boxplot(data=combined_df, x='dominio', y='time_ms', hue='algoritmo', ax=ax, palette=COLORS)
    ax.set_xlabel('Dominio', fontsize=11, fontweight='bold')
    ax.set_ylabel('Tiempo (ms)', fontsize=11, fontweight='bold')
    ax.set_title('Distribución de Tiempos por Dominio', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(title='Algoritmo')
    
    # Memoria por algoritmo
    ax = axes[1, 0]
    sns.boxplot(data=combined_df, x='algoritmo', y='delta_rss_kb', hue='algoritmo', ax=ax, palette=COLORS)
    ax.set_xlabel('Algoritmo', fontsize=11, fontweight='bold')
    ax.set_ylabel('Delta RSS (KB)', fontsize=11, fontweight='bold')
    ax.set_title('Distribución de Delta RSS por Algoritmo', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=0)
    
    # Peak memory por algoritmo
    ax = axes[1, 1]
    sns.boxplot(data=combined_df, x='algoritmo', y='peak_kb', hue='algoritmo', ax=ax, palette=COLORS)
    ax.set_xlabel('Algoritmo', fontsize=11, fontweight='bold')
    ax.set_ylabel('Peak Memory (KB)', fontsize=11, fontweight='bold')
    ax.set_title('Distribución de Peak Memory por Algoritmo', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=0)
    
    plt.suptitle('Análisis de Distribuciones', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / '07_box_plots.png', dpi=300, bbox_inches='tight')
    print("✓ Guardada: 07_box_plots.png")
    plt.close()





def main():
    """Función principal."""
    print("\n" + "="*60)
    print("  Generador de Gráficas - Algoritmos de Sorting")
    print("="*60 + "\n")
    
    # Cargar datos
    print("📊 Cargando datos...")
    data_dict = load_data()
    
    if not data_dict:
        print("❌ No se encontraron archivos CSV. Verifica la ruta.")
        return
    
    print(f"✓ Se cargaron {len(data_dict)} algoritmos\n")
    
    # Combinar datos
    combined_df = combine_data(data_dict)
    
    # Generar gráficas
    print("📈 Generando gráficas...\n")
    
    plot_time_vs_size(data_dict, combined_df)
    plot_time_by_data_type(combined_df)
    plot_memory_usage(combined_df)
    plot_efficiency_ratio(combined_df)
    plot_domain_comparison(combined_df)
    plot_box_plot_comparison(combined_df)
    
    print("\n" + "="*60)
    print(f"✅ ¡Gráficas generadas exitosamente!")
    print(f"📁 Ubicación: {PLOTS_DIR}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
