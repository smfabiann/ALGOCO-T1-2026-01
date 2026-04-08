#!/usr/bin/env python3
"""
Script para generar graficas comparativas de los algoritmos de
multiplicacion de matrices.
"""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# Estilo general
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["font.size"] = 10

# Rutas
DATA_DIR = Path(__file__).parent.parent / "data" / "measurements"
PLOTS_DIR = Path(__file__).parent.parent / "data" / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

ALGOS = ["naive", "strassen"]

COLORS = {
	"naive": "#2E86AB",
	"strassen": "#A23B72",
}

MARKERS = {
	"naive": "o",
	"strassen": "s",
}


def load_data():
	"""Carga los datos de los algoritmos disponibles."""
	data = {}

	for algo in ALGOS:
		csv_file = DATA_DIR / f"{algo}.csv"
		if csv_file.exists():
			data[algo] = pd.read_csv(csv_file)
		else:
			print(f"[WARN] Archivo no encontrado: {csv_file}")

	return data


def combine_data(data_dict):
	"""Combina todos los CSV en un solo DataFrame."""
	combined = []

	for algo, df in data_dict.items():
		df_copy = df.copy()
		df_copy["algoritmo"] = algo
		combined.append(df_copy)

	return pd.concat(combined, ignore_index=True)


def plot_time_vs_size(data_dict):
	"""Grafica 1: Tiempo vs tamano para todos los algoritmos."""
	fig, ax = plt.subplots(figsize=(10, 6))

	for algo in ALGOS:
		if algo in data_dict:
			df = data_dict[algo]
			grouped = df.groupby("size")["time_ms"]
			size_mean = grouped.mean()
			size_std = grouped.std().fillna(0.0)

			ax.errorbar(
				size_mean.index,
				size_mean.values,
				yerr=size_std.values,
				label=algo.capitalize(),
				marker=MARKERS[algo],
				color=COLORS[algo],
				linewidth=2,
				markersize=8,
				capsize=5,
				alpha=0.85,
			)

	ax.set_xlabel("Tamano de Entrada (N)", fontsize=12, fontweight="bold")
	ax.set_ylabel("Tiempo de Ejecucion (ms)", fontsize=12, fontweight="bold")
	ax.set_title("Tiempo vs Tamano - Todos los algoritmos", fontsize=13, fontweight="bold")
	ax.set_xscale("log")
	ax.set_yscale("log")
	ax.legend(fontsize=10)
	ax.grid(True, alpha=0.3, which="both")

	plt.tight_layout()
	plt.savefig(PLOTS_DIR / "01_tiempo_vs_tamano.png", dpi=300, bbox_inches="tight")
	print("[OK] Guardada: 01_tiempo_vs_tamano.png")
	plt.close()


def plot_time_without_largest(data_dict):
	"""Grafica 2: Tiempo vs tamano excluyendo el mayor N."""
	all_sizes = sorted({int(size) for df in data_dict.values() for size in df["size"].unique()})
	if len(all_sizes) <= 1:
		print("[WARN] No hay suficientes tamanos para excluir el mayor.")
		return

	max_size = all_sizes[-1]
	fig, ax = plt.subplots(figsize=(10, 6))

	for algo in ALGOS:
		if algo in data_dict:
			df = data_dict[algo]
			df_small = df[df["size"] != max_size]
			grouped = df_small.groupby("size")["time_ms"]
			size_mean = grouped.mean()
			size_std = grouped.std().fillna(0.0)

			ax.errorbar(
				size_mean.index,
				size_mean.values,
				yerr=size_std.values,
				label=algo.capitalize(),
				marker=MARKERS[algo],
				color=COLORS[algo],
				linewidth=2,
				markersize=8,
				capsize=5,
				alpha=0.85,
			)

	ax.set_xlabel("Tamano de Entrada (N)", fontsize=12, fontweight="bold")
	ax.set_ylabel("Tiempo de Ejecucion (ms)", fontsize=12, fontweight="bold")
	ax.set_title(f"Tiempo vs Tamano - Sin N = {max_size}", fontsize=13, fontweight="bold")
	ax.set_xscale("log")
	ax.set_yscale("log")
	ax.legend(fontsize=10)
	ax.grid(True, alpha=0.3, which="both")

	plt.tight_layout()
	plt.savefig(PLOTS_DIR / "02_tiempo_sin_mayor.png", dpi=300, bbox_inches="tight")
	print("[OK] Guardada: 02_tiempo_sin_mayor.png")
	plt.close()


def plot_time_by_type(combined_df):
	"""Grafica 3: Tiempo por tipo de matriz (densa/diagonal/dispersa)."""
	fig, axes = plt.subplots(1, 3, figsize=(18, 5))
	tipos = ["densa", "diagonal", "dispersa"]

	for idx, tipo in enumerate(tipos):
		ax = axes[idx]
		df_tipo = combined_df[combined_df["tipo"] == tipo]

		for algo in ALGOS:
			df_algo = df_tipo[df_tipo["algoritmo"] == algo]
			if len(df_algo) > 0:
				size_mean = df_algo.groupby("size")["time_ms"].mean()
				ax.plot(
					size_mean.index,
					size_mean.values,
					label=algo.capitalize(),
					marker=MARKERS[algo],
					color=COLORS[algo],
					linewidth=2.5,
					markersize=8,
					alpha=0.85,
				)

		ax.set_xlabel("Tamano de Entrada (N)", fontsize=11, fontweight="bold")
		ax.set_ylabel("Tiempo (ms)", fontsize=11, fontweight="bold")
		ax.set_title(f"Matrices {tipo.capitalize()}", fontsize=12, fontweight="bold")
		ax.set_xscale("log")
		ax.set_yscale("log")
		ax.legend(fontsize=9)
		ax.grid(True, alpha=0.3, which="both")

	plt.suptitle("Comparacion de Tiempo por Tipo de Matriz", fontsize=14, fontweight="bold", y=1.02)
	plt.tight_layout()
	plt.savefig(PLOTS_DIR / "03_tiempo_por_tipo.png", dpi=300, bbox_inches="tight")
	print("[OK] Guardada: 03_tiempo_por_tipo.png")
	plt.close()


def plot_time_by_domain(combined_df):
	"""Grafica 4: Tiempo por dominio (D0 vs D10)."""
	fig, axes = plt.subplots(1, 2, figsize=(16, 6))
	dominios = ["D0", "D10"]

	for idx, dominio in enumerate(dominios):
		ax = axes[idx]
		df_dom = combined_df[combined_df["dominio"] == dominio]

		for algo in ALGOS:
			df_algo = df_dom[df_dom["algoritmo"] == algo]
			if len(df_algo) > 0:
				size_mean = df_algo.groupby("size")["time_ms"].mean()
				ax.plot(
					size_mean.index,
					size_mean.values,
					label=algo.capitalize(),
					marker=MARKERS[algo],
					color=COLORS[algo],
					linewidth=2.5,
					markersize=8,
					alpha=0.85,
				)

		ax.set_xlabel("Tamano de Entrada (N)", fontsize=11, fontweight="bold")
		ax.set_ylabel("Tiempo (ms)", fontsize=11, fontweight="bold")
		ax.set_title(f"Dominio {dominio}", fontsize=12, fontweight="bold")
		ax.set_xscale("log")
		ax.set_yscale("log")
		ax.legend(fontsize=10)
		ax.grid(True, alpha=0.3, which="both")

	plt.suptitle("Comparacion de Tiempo por Dominio", fontsize=14, fontweight="bold", y=1.00)
	plt.tight_layout()
	plt.savefig(PLOTS_DIR / "04_tiempo_por_dominio.png", dpi=300, bbox_inches="tight")
	print("[OK] Guardada: 04_tiempo_por_dominio.png")
	plt.close()


def plot_memory_usage(combined_df):
	"""Grafica 5: Memoria vs tamano (mediana + IQR, escala symlog)."""
	fig, axes = plt.subplots(2, 2, figsize=(16, 12))

	memory_metrics = [
		("delta_rss_kb", "Delta RSS (KB)", axes[0, 0]),
		("peak_kb", "Peak Memory (KB)", axes[0, 1]),
		("delta_peak_from_start_kb", "Delta Peak desde Inicio (KB)", axes[1, 0]),
		("rss_after_kb", "RSS Despues (KB)", axes[1, 1]),
	]

	for metric, title, ax in memory_metrics:
		metric_positive = combined_df[combined_df[metric] > 0][metric]
		linthresh = max(1.0, float(np.nanpercentile(metric_positive, 20))) if len(metric_positive) > 0 else 1.0

		for algo in ALGOS:
			df_algo = combined_df[combined_df["algoritmo"] == algo]
			if len(df_algo) > 0:
				grouped = df_algo.groupby("size")[metric]
				size_median = grouped.median()
				size_q1 = grouped.quantile(0.25)
				size_q3 = grouped.quantile(0.75)

				x_vals = size_median.index.values
				y_median = size_median.values
				y_q1 = size_q1.values
				y_q3 = size_q3.values

				ax.fill_between(x_vals, y_q1, y_q3, color=COLORS[algo], alpha=0.18)
				ax.plot(
					x_vals,
					y_median,
					label=algo.capitalize(),
					marker=MARKERS[algo],
					color=COLORS[algo],
					linewidth=2.5,
					markersize=8,
					alpha=0.85,
				)

		ax.set_xlabel("Tamano de Entrada (N)", fontsize=11, fontweight="bold")
		ax.set_ylabel(title, fontsize=11, fontweight="bold")
		ax.set_title(title, fontsize=12, fontweight="bold")
		ax.set_xscale("log")
		ax.set_yscale("symlog", linthresh=linthresh)
		ax.legend(fontsize=9)
		ax.grid(True, alpha=0.3, which="both")
		ax.text(
			0.02,
			0.95,
			"Linea: mediana | Banda: IQR",
			transform=ax.transAxes,
			fontsize=8,
			va="top",
			bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
		)

	plt.suptitle("Comparacion de Uso de Memoria (Escala symlog)", fontsize=14, fontweight="bold", y=1.00)
	plt.tight_layout()
	plt.savefig(PLOTS_DIR / "05_memoria_vs_tamano.png", dpi=300, bbox_inches="tight")
	print("[OK] Guardada: 05_memoria_vs_tamano.png")
	plt.close()


def main():
	"""Funcion principal."""
	print("\n" + "=" * 60)
	print("  Generador de Graficas - Multiplicacion de Matrices")
	print("=" * 60 + "\n")

	print("[INFO] Cargando datos...")
	data_dict = load_data()
	if not data_dict:
		print("[ERROR] No se encontraron archivos CSV. Verifica la ruta.")
		return

	print(f"[OK] Se cargaron {len(data_dict)} algoritmos\n")
	combined_df = combine_data(data_dict)

	print("[INFO] Generando graficas...\n")
	plot_time_vs_size(data_dict)
	plot_time_without_largest(data_dict)
	plot_time_by_type(combined_df)
	plot_time_by_domain(combined_df)
	plot_memory_usage(combined_df)

	print("\n" + "=" * 60)
	print("[OK] Graficas generadas exitosamente")
	print(f"[INFO] Ubicacion: {PLOTS_DIR}")
	print("=" * 60 + "\n")


if __name__ == "__main__":
	main()
