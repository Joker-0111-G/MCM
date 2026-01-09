#!/usr/bin/env python3
"""
Ejemplo de uso del Optimizador de Rutas para Rescatistas
Demuestra cómo usar el programa de forma modular
"""

from rescatistas_optimizacion import OptimizadorRescatistas
import json

def ejemplo_uso_basico():
    """Ejemplo de uso básico del optimizador"""
    print("=" * 60)
    print("📋 EJEMPLO DE USO BÁSICO")
    print("=" * 60)
    
    # Crear optimizador con parámetros estándar FEMA/NFPA
    optimizador = OptimizadorRescatistas(tiempo_busqueda=30, velocidad_rescatistas=1.2)
    
    # Ejecutar solo la optimización exhaustiva
    print("🔍 Ejecutando optimización exhaustiva...")
    resultados = optimizador.optimizar_rutas()
    
    # Mostrar solo el mejor resultado
    mejor = resultados['mejor']
    print(f"\n✨ Mejor solución encontrada:")
    print(f"   Tiempo total: {mejor['tiempo_total']} segundos")
    print(f"   Rescatista 1: {mejor['rescatista1_ruta']} ({mejor['tiempo1']}s)")
    print(f"   Rescatista 2: {mejor['rescatista2_ruta']} ({mejor['tiempo2']}s)")
    
    return optimizador, resultados

def ejemplo_uso_completo():
    """Ejemplo de uso completo con todas las funcionalidades"""
    print("\n" + "=" * 60)
    print("🎯 EJEMPLO DE USO COMPLETO")
    print("=" * 60)
    
    # Crear optimizador con parámetros estándar FEMA/NFPA
    optimizador = OptimizadorRescatistas(tiempo_busqueda=30, velocidad_rescatistas=1.2)
    
    # 1. Optimización exhaustiva
    print("1️⃣ Optimización exhaustiva...")
    optimizacion = optimizador.optimizar_rutas()
    
    # 2. Estrategias determinísticas
    print("2️⃣ Evaluando estrategias determinísticas...")
    estrategias = optimizador.estrategias_deterministicas()
    
    # Encontrar la mejor estrategia determinística
    mejor_estrategia = min(estrategias.values(), key=lambda x: x['tiempo_total'])
    print(f"   Mejor estrategia determinística: {mejor_estrategia['nombre']} ({mejor_estrategia['tiempo_total']}s)")
    
    # 3. Monte Carlo (solo para la mejor estrategia)
    clave_mejor = [k for k, v in estrategias.items() if v == mejor_estrategia][0]
    print(f"3️⃣ Simulando Monte Carlo para estrategia {clave_mejor}...")
    monte_carlo = optimizador.simulacion_monte_carlo(clave_mejor, 500)  # Menos simulaciones para el ejemplo
    
    print(f"   Resultado Monte Carlo: {monte_carlo['tiempo_promedio']:.1f} ± {monte_carlo['desviacion_estandar']:.1f}s")
    
    return {
        'optimizacion': optimizacion,
        'estrategias': estrategias,
        'monte_carlo': monte_carlo
    }

def ejemplo_analisis_comparativo():
    """Ejemplo de análisis comparativo de diferentes configuraciones"""
    print("\n" + "=" * 60)
    print("🔬 ANÁLISIS COMPARATIVO")
    print("=" * 60)
    
    # Probar diferentes tiempos de búsqueda
    tiempos_busqueda = [20, 25, 30, 35, 40]
    resultados_comparativos = []
    
    for tiempo in tiempos_busqueda:
        print(f"   Evaluando con tiempo de búsqueda: {tiempo}s")
        optimizador = OptimizadorRescatistas(tiempo_busqueda=tiempo, velocidad_rescatistas=1.2)
        resultado = optimizador.optimizar_rutas()
        
        resultados_comparativos.append({
            'tiempo_busqueda': tiempo,
            'tiempo_optimo': resultado['mejor']['tiempo_total'],
            'estrategia_optima': {
                'r1': resultado['mejor']['rescatista1_ruta'],
                'r2': resultado['mejor']['rescatista2_ruta']
            }
        })
    
    # Mostrar resultados comparativos
    print(f"\n📊 Resultados comparativos:")
    print(f"{'Tiempo Búsqueda':<15} {'Tiempo Óptimo':<15} {'Estrategia Óptima'}")
    print("-" * 60)
    
    for resultado in resultados_comparativos:
        estrategia = f"R1:{resultado['estrategia_optima']['r1']} | R2:{resultado['estrategia_optima']['r2']}"
        print(f"{resultado['tiempo_busqueda']:<15} {resultado['tiempo_optimo']:<15} {estrategia}")
    
    return resultados_comparativos

def guardar_ejemplo_resultados(resultados):
    """Guarda los resultados del ejemplo en un archivo específico"""
    with open('/workspace/ejemplo_resultados.json', 'w', encoding='utf-8') as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n💾 Resultados del ejemplo guardados en 'ejemplo_resultados.json'")

def main():
    """Función principal que ejecuta todos los ejemplos"""
    print("🚀 DEMO COMPLETA - OPTIMIZADOR DE RUTAS PARA RESCATISTAS")
    print("=" * 60)
    
    # Ejecutar ejemplos
    optimizador, resultados_basicos = ejemplo_uso_basico()
    resultados_completos = ejemplo_uso_completo()
    resultados_comparativos = ejemplo_analisis_comparativo()
    
    # Compilar todos los resultados
    ejemplo_completo = {
        'uso_basico': resultados_basicos,
        'uso_completo': resultados_completos,
        'analisis_comparativo': resultados_comparativos
    }
    
    # Guardar resultados
    guardar_ejemplo_resultados(ejemplo_completo)
    
    print(f"\n✅ Demostración completa finalizada")
    print(f"📁 Archivos generados:")
    print(f"   - rescatistas_optimizacion.py (programa principal)")
    print(f"   - visualizar_resultados.py (visualizaciones)")
    print(f"   - resultados_rescatistas.json (resultados completos)")
    print(f"   - resumen_resultados.md (resumen detallado)")
    print(f"   - resultados_visualizacion.png (gráficos)")
    print(f"   - ejemplo_resultados.json (demostración de uso)")

if __name__ == "__main__":
    main()