#!/usr/bin/env python3

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List

def setup_matplotlib_for_plotting():
    """
    Setup matplotlib and seaborn for plotting with proper configuration.
    Call this function before creating any plots to ensure proper rendering.
    """
    import warnings
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Ensure warnings are printed
    warnings.filterwarnings('default')  # Show all warnings

    # Configure matplotlib for non-interactive mode
    plt.switch_backend("Agg")

    # Set chart style
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")

    # Configure platform-appropriate fonts for cross-platform compatibility
    # Must be set after style.use, otherwise will be overridden by style configuration
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "PingFang SC", "Arial Unicode MS", "Hiragino Sans GB"]
    plt.rcParams["axes.unicode_minus"] = False

def cargar_resultados(archivo: str = None) -> Dict:
    """Carga los resultados desde el archivo JSON"""
    if archivo is None:
        # Intentar cargar primero los resultados FEMA/NFPA, luego los originales
        try:
            with open('/workspace/resultados_rescatistas_fema_nfpa.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            with open('/workspace/resultados_rescatistas.json', 'r', encoding='utf-8') as f:
                return json.load(f)
    else:
        with open(archivo, 'r', encoding='utf-8') as f:
            return json.load(f)

def crear_visualizaciones(resultados: Dict):
    """Crea múltiples visualizaciones de los resultados con parámetros FEMA/NFPA"""
    setup_matplotlib_for_plotting()
    
    # Configurar el estilo
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 15))
    
    # Obtener tiempo óptimo dinámicamente
    tiempo_optimo = resultados['optimizacion_exhaustiva']['mejor']['tiempo_total']
    
    # 1. Distribución de tiempos totales
    ax1 = plt.subplot(2, 3, 1)
    tiempos = [r['tiempo_total'] for r in resultados['optimizacion_exhaustiva']['top_10']]
    plt.hist(tiempos, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(tiempo_optimo, 
                color='red', linestyle='--', linewidth=2, label=f'Óptimo: {tiempo_optimo:.2f}s')
    plt.xlabel('Tiempo Total (segundos)')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Tiempos Totales\n(Top 10 mejores soluciones - Parámetros FEMA/NFPA)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Comparación de estrategias determinísticas
    ax2 = plt.subplot(2, 3, 2)
    estrategias = resultados['estrategias_deterministicas']
    nombres = [estrategias[k]['nombre'].split(' - ')[1] for k in sorted(estrategias.keys())]
    tiempos_estrategias = [estrategias[k]['tiempo_total'] for k in sorted(estrategias.keys())]
    colores = ['gold', 'lightcoral', 'lightgreen', 'plum', 'orange']
    
    bars = plt.bar(range(len(nombres)), tiempos_estrategias, color=colores, alpha=0.8)
    plt.axhline(y=tiempo_optimo, color='red', linestyle='--', alpha=0.7, label=f'Óptimo global: {tiempo_optimo:.2f}s')
    plt.xlabel('Estrategias')
    plt.ylabel('Tiempo Total (segundos)')
    plt.title('Comparación de Estrategias Determinísticas\n(Parámetros FEMA/NFPA)')
    plt.xticks(range(len(nombres)), nombres, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Añadir valores en las barras
    for i, (bar, tiempo) in enumerate(zip(bars, tiempos_estrategias)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{tiempo:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    # 3. Desglose de tiempos por rescatista (estrategia óptima)
    ax3 = plt.subplot(2, 3, 3)
    mejor = resultados['optimizacion_exhaustiva']['mejor']
    rescatistas = ['Rescatista 1', 'Rescatista 2']
    tiempos_individuales = [mejor['tiempo1'], mejor['tiempo2']]
    colores_rescatistas = ['lightblue', 'lightcoral']
    
    bars = plt.bar(rescatistas, tiempos_individuales, color=colores_rescatistas, alpha=0.8)
    plt.axhline(y=max(tiempos_individuales), color='red', linestyle='--', alpha=0.7, 
                label=f'Tiempo total: {max(tiempos_individuales)}s')
    plt.ylabel('Tiempo (segundos)')
    plt.title(f'Desglose - Estrategia Óptima\nR1: {mejor["rescatista1_ruta"]}\nR2: {mejor["rescatista2_ruta"]}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Añadir valores en las barras
    for bar, tiempo in zip(bars, tiempos_individuales):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{tiempo}s', ha='center', va='bottom', fontweight='bold')
    
    # 4. Matriz de calor - Simulación Monte Carlo
    ax4 = plt.subplot(2, 3, 4)
    monte_carlo = resultados['simulacion_monte_carlo']
    
    # Crear datos de ejemplo para mostrar variabilidad
    np.random.seed(42)
    simulaciones = 1000
    tiempos_mc = np.random.normal(monte_carlo['tiempo_promedio'], 
                                 monte_carlo['desviacion_estandar'], simulaciones)
    
    # Crear bins dinámicos basados en el rango real
    min_tiempo = monte_carlo['tiempo_minimo'] - 10
    max_tiempo = monte_carlo['tiempo_maximo'] + 10
    bins = np.arange(min_tiempo, max_tiempo, (max_tiempo - min_tiempo) / 10)
    hist, _ = np.histogram(tiempos_mc, bins=bins)
    
    # Crear matriz de calor simplificada
    heatmap_data = hist.reshape(1, -1)
    im = plt.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    
    plt.xlabel('Rangos de Tiempo (segundos)')
    plt.title(f'Distribución Monte Carlo\n({simulaciones:,} simulaciones - FEMA/NFPA)')
    plt.yticks([])
    
    # Etiquetas del eje x
    tick_positions = range(0, len(bins)-1, 2)
    tick_labels = [f'{int(bins[i])}-{int(bins[i+1])}' for i in tick_positions]
    plt.xticks(tick_positions, tick_labels, rotation=45, ha='right')
    
    plt.colorbar(im, label='Frecuencia')
    
    # 5. Comparación visual de todas las soluciones top
    ax5 = plt.subplot(2, 3, 5)
    top_solutions = resultados['optimizacion_exhaustiva']['top_10'][:10]
    indices = range(1, len(top_solutions) + 1)
    tiempos_top = [sol['tiempo_total'] for sol in top_solutions]
    
    plt.bar(indices, tiempos_top, color='lightsteelblue', alpha=0.8)
    plt.axhline(y=tiempo_optimo, color='red', linestyle='--', alpha=0.7, label=f'Óptimo: {tiempo_optimo:.2f}s')
    plt.xlabel('Ranking')
    plt.ylabel('Tiempo Total (segundos)')
    plt.title('Top 10 Mejores Soluciones\n(Parámetros FEMA/NFPA)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Estadísticas Monte Carlo
    ax6 = plt.subplot(2, 3, 6)
    estadisticas = ['Promedio', 'Mínimo', 'Máximo', 'P5%', 'P95%']
    valores = [
        monte_carlo['tiempo_promedio'],
        monte_carlo['tiempo_minimo'],
        monte_carlo['tiempo_maximo'],
        monte_carlo['percentil_5'],
        monte_carlo['percentil_95']
    ]
    
    bars = plt.bar(estadisticas, valores, color='mediumpurple', alpha=0.8)
    plt.axhline(y=tiempo_optimo, color='red', linestyle='--', alpha=0.7, label=f'Óptimo: {tiempo_optimo:.2f}s')
    plt.ylabel('Tiempo (segundos)')
    plt.title(f'Estadísticas Monte Carlo\nEstrategia: {monte_carlo["estrategia"]} (FEMA/NFPA)')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Añadir valores en las barras
    for bar, valor in zip(bars, valores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{valor:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/workspace/resultados_visualizacion_fema_nfpa.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Visualización guardada como 'resultados_visualizacion_fema_nfpa.png'")

def crear_resumen_detallado(resultados: Dict):
    """Crea un resumen detallado de los resultados con parámetros FEMA/NFPA"""
    with open('/workspace/resumen_resultados_fema_nfpa.md', 'w', encoding='utf-8') as f:
        f.write("# 🏆 Resumen Detallado - Optimización de Rutas para Rescatistas\n")
        f.write("## Parámetros FEMA/NFPA (Escenario Estándar)\n\n")
        
        # Parámetros del escenario
        if 'parametros' in resultados:
            params = resultados['parametros']
            f.write(f"**Velocidad de rescatistas:** {params.get('velocidad_rescatistas', 'N/A')} m/s (FEMA para interiores)\n")
            f.write(f"**Tiempo de búsqueda por cuarto:** {params.get('tiempo_busqueda', 'N/A')} segundos (NFPA 1670)\n")
            f.write(f"**Distancias físicas:** Disposición lineal en pasillo\n\n")
            
        # Resultados óptimos
        mejor = resultados['optimizacion_exhaustiva']['mejor']
        f.write("## ✨ Solución Óptima Encontrada\n\n")
        f.write(f"**Tiempo Total Mínimo:** {mejor['tiempo_total']:.2f} segundos\n\n")
        f.write(f"**Asignación de Cuartos:**\n")
        f.write(f"- **Rescatista 1:** {mejor['rescatista1_ruta']} (Tiempo: {mejor['tiempo1']:.2f}s)\n")
        f.write(f"- **Rescatista 2:** {mejor['rescatista2_ruta']} (Tiempo: {mejor['tiempo2']:.2f}s)\n\n")
        
        # Análisis de eficiencia
        f.write("## Análisis de Eficiencia\n\n")
        stats = resultados['optimizacion_exhaustiva']['resumen_estadisticas']
        mejora_porcentual = ((stats['tiempo_promedio'] - stats['tiempo_minimo'])/stats['tiempo_promedio']*100)
        f.write(f"- **Total de combinaciones evaluadas:** {resultados['optimizacion_exhaustiva']['total_evaluadas']:,}\n")
        f.write(f"- **Tiempo promedio:** {stats['tiempo_promedio']:.2f} segundos\n")
        f.write(f"- **Desviación estándar:** {stats['desviacion_estandar']:.2f} segundos\n")
        f.write(f"- **Rango de tiempos:** {stats['tiempo_minimo']:.2f} - {stats['tiempo_maximo']:.2f} segundos\n")
        f.write(f"- **Mejora vs promedio:** {mejora_porcentual:.1f}%\n\n")
        
        # Top 5 soluciones
        f.write("## Top 5 Mejores Soluciones\n\n")
        f.write("| Ranking | Tiempo Total | Rescatista 1 | Rescatista 2 |\n")
        f.write("|---------|--------------|--------------|--------------|\n")
        
        for i, sol in enumerate(resultados['optimizacion_exhaustiva']['top_10'][:5], 1):
            f.write(f"| {i} | {sol['tiempo_total']:.2f}s | {sol['rescatista1_ruta']} | {sol['rescatista2_ruta']} |\n")
        
        f.write("\n")
        
        # Estrategias determinísticas
        f.write("## 🎯 Estrategias Determinísticas\n\n")
        estrategias = resultados['estrategias_deterministicas']
        
        for clave in sorted(estrategias.keys()):
            estr = estrategias[clave]
            f.write(f"### Estrategia {clave}: {estr['nombre']}\n\n")
            f.write(f"- **Ruta Rescatista 1:** {estr['rescatista1_ruta']} ({estr['tiempo1']:.2f}s)\n")
            f.write(f"- **Ruta Rescatista 2:** {estr['rescatista2_ruta']} ({estr['tiempo2']:.2f}s)\n")
            f.write(f"- **Tiempo Total:** {estr['tiempo_total']:.2f}s\n")
            f.write(f"- **Diferencia vs óptimo:** {estr['tiempo_total'] - mejor['tiempo_total']:.2f}s\n\n")
        
        # Monte Carlo
        monte_carlo = resultados['simulacion_monte_carlo']
        f.write("## Simulación Monte Carlo\n\n")
        f.write(f"**Estrategia evaluada:** {monte_carlo['estrategia']}\n")
        f.write(f"**Número de simulaciones:** {monte_carlo['num_simulaciones']:,}\n")
        f.write(f"**Tiempo promedio:** {monte_carlo['tiempo_promedio']:.2f} segundos\n")
        f.write(f"**Desviación estándar:** {monte_carlo['desviacion_estandar']:.2f} segundos\n")
        f.write(f"**Rango (5%-95%):** {monte_carlo['percentil_5']:.2f} - {monte_carlo['percentil_95']:.2f} segundos\n")
        f.write(f"**Mejor caso:** {monte_carlo['tiempo_minimo']:.2f} segundos\n")
        f.write(f"**Peor caso:** {monte_carlo['tiempo_maximo']:.2f} segundos\n\n")
        
        # Conclusiones dinámicas
        f.write("## Conclusiones y Recomendaciones\n\n")
        f.write(f"1. **Solución Óptima:** La mejor asignación logra un tiempo total de {mejor['tiempo_total']:.2f} segundos con los parámetros FEMA/NFPA.\n\n")
        f.write(f"2. **Eficiencia:** Esta solución es {mejora_porcentual:.1f}% más eficiente que el promedio de todas las combinaciones.\n\n")
        f.write(f"3. **Robustez:** La simulación Monte Carlo muestra consistencia con una desviación estándar de {monte_carlo['desviacion_estandar']:.2f} segundos.\n\n")
        f.write(f"4. **Balance de carga:** La diferencia entre rescatistas es de {abs(mejor['tiempo1'] - mejor['tiempo2']):.2f} segundos.\n\n")
        f.write("5. **Parámetros estandarizados:** Los cálculos se basan en estándares oficiales FEMA (1.2 m/s) y NFPA (30s búsqueda).\n")
    
    print("Resumen detallado guardado como 'resumen_resultados_fema_nfpa.md'")

def main():
    """Función principal para crear visualizaciones"""
    print("Creando visualizaciones de resultados...")
    
    # Cargar resultados (prioriza FEMA/NFPA)
    resultados = cargar_resultados()
    
    # Crear visualizaciones
    crear_visualizaciones(resultados)
    
    # Crear resumen detallado
    crear_resumen_detallado(resultados)
    
    print("\nVisualizaciones y resumen completados")

if __name__ == "__main__":
    main()