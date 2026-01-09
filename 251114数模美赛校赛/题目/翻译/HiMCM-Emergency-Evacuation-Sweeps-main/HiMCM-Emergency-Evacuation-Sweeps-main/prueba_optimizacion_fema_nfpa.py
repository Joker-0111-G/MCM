#!/usr/bin/env python3
"""
Prueba rápida del optimizador con los nuevos parámetros FEMA/NFPA
"""

import sys
sys.path.append('/workspace/user_input_files')
from rescatistas_optimizacion import OptimizadorRescatistas
import json

def prueba_optimizacion():
    """Ejecuta una optimización rápida para verificar que todo funciona"""
    print("="*60)
    print(" PRUEBA RÁPIDA - OPTIMIZACIÓN CON PARÁMETROS FEMA/NFPA")
    print("="*60)
    
    # Crear optimizador
    optimizador = OptimizadorRescatistas(tiempo_busqueda=30, velocidad_rescatistas=1.2)
    
    print(f"✅ Optimizador creado exitosamente")
    print(f"   Velocidad: {optimizador.velocidad_rescatistas} m/s")
    print(f"   Tiempo búsqueda: {optimizador.tiempo_busqueda} segundos")
    print(f"   Cuartos: {optimizador.cuartos}")
    print()
    
    # Ejecutar optimización (solo la primera parte)
    print("🔍 Ejecutando optimización exhaustiva...")
    resultados = optimizador.optimizar_rutas()
    
    mejor = resultados['mejor']
    print(f"\n✨ RESULTADO ÓPTIMO:")
    print(f"   Tiempo total: {mejor['tiempo_total']:.2f} segundos")
    print(f"   Rescatista 1: {mejor['rescatista1_ruta']} ({mejor['tiempo1']:.2f}s)")
    print(f"   Rescatista 2: {mejor['rescatista2_ruta']} ({mejor['tiempo2']:.2f}s)")
    print(f"   Combinaciones evaluadas: {resultados['total_evaluadas']:,}")
    
    # Estadísticas
    stats = resultados['resumen_estadisticas']
    print(f"\n📊 ESTADÍSTICAS:")
    print(f"   Tiempo promedio: {stats['tiempo_promedio']:.2f} segundos")
    print(f"   Desviación estándar: {stats['desviacion_estandar']:.2f} segundos")
    print(f"   Rango: {stats['tiempo_minimo']:.2f} - {stats['tiempo_maximo']:.2f} segundos")
    
    # Probar estrategias determinísticas
    print(f"\n🎯 Evaluando estrategias determinísticas...")
    estrategias = optimizador.estrategias_deterministicas()
    
    mejor_estrategia = min(estrategias.values(), key=lambda x: x['tiempo_total'])
    print(f"   Mejor estrategia determinística:")
    print(f"   {mejor_estrategia['nombre']}: {mejor_estrategia['tiempo_total']:.2f}s")
    print(f"   R1: {mejor_estrategia['rescatista1_ruta']} ({mejor_estrategia['tiempo1']:.2f}s)")
    print(f"   R2: {mejor_estrategia['rescatista2_ruta']} ({mejor_estrategia['tiempo2']:.2f}s)")
    
    return optimizador, resultados, estrategias

def guardar_prueba():
    """Guarda los resultados de la prueba"""
    optimizador, resultados, estrategias = prueba_optimizacion()
    
    # Guardar solo el resultado principal
    resultado_resumen = {
        'mejor_solucion': resultados['mejor'],
        'estadisticas': resultados['resumen_estadisticas'],
        'mejor_estrategia': min(estrategias.values(), key=lambda x: x['tiempo_total']),
        'parametros_usados': {
            'velocidad_rescatistas': optimizador.velocidad_rescatistas,
            'tiempo_busqueda': optimizador.tiempo_busqueda,
            'distancias_fisicas': optimizador.distancias_fisicas
        }
    }
    
    with open('/workspace/prueba_parametros_fema_nfpa.json', 'w', encoding='utf-8') as f:
        json.dump(resultado_resumen, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Resultados de la prueba guardados en 'prueba_parametros_fema_nfpa.json'")
    print(f"\n✅ PRUEBA COMPLETADA - Los parámetros FEMA/NFPA funcionan correctamente!")

if __name__ == "__main__":
    guardar_prueba()