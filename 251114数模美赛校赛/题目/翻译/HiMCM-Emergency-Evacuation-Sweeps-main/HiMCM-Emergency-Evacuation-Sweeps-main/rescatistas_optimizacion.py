#!/usr/bin/env python3

import itertools
import numpy as np
from typing import List, Tuple, Dict, Any
import json

class OptimizadorRescatistas:
    def __init__(self, tiempo_busqueda: int = 30, velocidad_rescatistas: float = 1.2):
        """
        Inicializa el optimizador con parámetros basados en estándares oficiales
        
        Args:
            tiempo_busqueda: Tiempo fijo en segundos para revisar cada cuarto (NFPA 1670: 30s)
            velocidad_rescatistas: Velocidad en m/s según protocolos FEMA (1.2 m/s estándar)
        """
        self.tiempo_busqueda = tiempo_busqueda
        self.velocidad_rescatistas = velocidad_rescatistas
        self.cuartos = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6']
        self.num_rescatistas = 2
        # Distancias físicas en metros (disposición lineal en pasillo)
        self.distancias_fisicas = {
            'Start': 0,
            'R1': 10,
            'R2': 15,
            'R3': 20,
            'R4': 25,
            'R5': 30,
            'R6': 35
        }
        self.matriz_distancias = self._crear_matriz_distancias()
        self.resultados = []
        
    def _crear_matriz_distancias(self) -> Dict[str, Dict[str, float]]:
        """
        Crea una matriz de distancias basada en distancias físicas reales
        y velocidad estándar FEMA (1.2 m/s para interiores)
        
        Distancias físicas del pasillo lineal:
        - Start: 0m, R1: 10m, R2: 15m, R3: 20m, R4: 25m, R5: 30m, R6: 35m
        
        Returns:
            Matriz de tiempos de recorrido en segundos (distancia/velocidad)
        """
        matriz_tiempos = {}
        
        # Calcular tiempo desde cada punto a todos los demás
        ubicaciones = ['Start', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6']
        
        for origen in ubicaciones:
            matriz_tiempos[origen] = {}
            for destino in ubicaciones:
                if origen == destino:
                    matriz_tiempos[origen][destino] = 0
                else:
                    # Tiempo = distancia física / velocidad
                    distancia = abs(self.distancias_fisicas[origen] - self.distancias_fisicas[destino])
                    tiempo = distancia / self.velocidad_rescatistas
                    # Redondear a 2 decimales para precisión
                    matriz_tiempos[origen][destino] = round(tiempo, 2)
        
        return matriz_tiempos
    
    def _calcular_tiempo_rescatista(self, ruta: List[str]) -> float:
        """
        Calcula el tiempo total para un rescatista siguiendo una ruta
        
        Args:
            ruta: Lista de cuartos a visitar en orden
            
        Returns:
            Tiempo total en segundos (float para mayor precisión)
        """
        if not ruta:
            return 0.0
            
        tiempo_total = 0.0
        
        # Tiempo de caminar desde Start al primer cuarto
        tiempo_total += self.matriz_distancias['Start'][ruta[0]]
        
        # Tiempo de caminar entre cuartos consecutivos
        for i in range(len(ruta) - 1):
            tiempo_total += self.matriz_distancias[ruta[i]][ruta[i + 1]]
        
        # Tiempo de búsqueda en cada cuarto
        tiempo_total += self.tiempo_busqueda * len(ruta)
        
        return tiempo_total
    
    def _generar_particiones(self) -> List[Tuple[List[str], List[str]]]:
        """
        Genera todas las posibles particiones de 6 cuartos entre 2 rescatistas
        
        Returns:
            Lista de tuplas, cada una contiene las asignaciones para los 2 rescatistas
        """
        particiones = []
        cuartos = self.cuartos
        
        # Generar todas las combinaciones posibles para el rescatista 1
        for i in range(len(cuartos) + 1):
            for comb_rescatista1 in itertools.combinations(cuartos, i):
                # Los cuartos restantes van al rescatista 2
                rescatista1 = list(comb_rescatista1)
                rescatista2 = [q for q in cuartos if q not in rescatista1]
                particiones.append((rescatista1, rescatista2))
        
        return particiones
    
    def _generar_permutaciones_ruta(self, cuartos_asignados: List[str]) -> List[List[str]]:
        """
        Genera todas las permutaciones posibles para una lista de cuartos
        
        Args:
            cuartos_asignados: Lista de cuartos asignados a un rescatista
            
        Returns:
            Lista de permutaciones (rutas) posibles
        """
        if len(cuartos_asignados) <= 1:
            return [cuartos_asignados]
        
        permutaciones = []
        for perm in itertools.permutations(cuartos_asignados):
            permutaciones.append(list(perm))
        
        return permutaciones
    
    def optimizar_rutas(self) -> Dict[str, Any]:
        """
        Realiza la optimización exhaustiva de todas las combinaciones posibles
        
        Returns:
            Diccionario con los resultados de la optimización
        """
        print("Iniciando optimización")
        print(f"Cuartos: {self.cuartos}")
        print(f"Tiempo de búsqueda por cuarto: {self.tiempo_busqueda} segundos")
        print(f"Matriz de distancias creada con {len(self.matriz_distancias)} nodos")
        print()
        
        particiones = self._generar_particiones()
        print(f"Total de particiones a evaluar: {len(particiones)}")
        
        mejores_resultados = []
        contador = 0
        
        for particion in particiones:
            rescatista1_cuartos, rescatista2_cuartos = particion
            
            # Si uno de los rescatistas no tiene cuartos, saltamos
            if not rescatista1_cuartos or not rescatista2_cuartos:
                continue
            
            # Generar permutaciones para ambos rescatistas
            permutaciones1 = self._generar_permutaciones_ruta(rescatista1_cuartos)
            permutaciones2 = self._generar_permutaciones_ruta(rescatista2_cuartos)
            
            # Evaluar todas las combinaciones de permutaciones
            for ruta1 in permutaciones1:
                for ruta2 in permutaciones2:
                    contador += 1
                    
                    # Calcular tiempos individuales
                    tiempo1 = self._calcular_tiempo_rescatista(ruta1)
                    tiempo2 = self._calcular_tiempo_rescatista(ruta2)
                    
                    # El tiempo total es el máximo de ambos (el más lento)
                    tiempo_total = max(tiempo1, tiempo2)
                    
                    resultado = {
                        'rescatista1_ruta': ruta1,
                        'rescatista2_ruta': ruta2,
                        'rescatista1_cuartos': len(ruta1),
                        'rescatista2_cuartos': len(ruta2),
                        'tiempo1': tiempo1,
                        'tiempo2': tiempo2,
                        'tiempo_total': tiempo_total
                    }
                    
                    mejores_resultados.append(resultado)
                    
                    # Mostrar progreso cada 1000 combinaciones
                    if contador % 1000 == 0:
                        print(f"   Evaluadas {contador:,} combinaciones...")
        
        print(f"Evaluación completada: {contador:,} combinaciones evaluadas")
        
        # Encontrar el mejor resultado
        mejor_resultado = min(mejores_resultados, key=lambda x: x['tiempo_total'])
        
        # Ordenar resultados por tiempo total
        mejores_resultados.sort(key=lambda x: x['tiempo_total'])
        
        return {
            'mejor': mejor_resultado,
            'top_10': mejores_resultados[:10],
            'total_evaluadas': contador,
            'resumen_estadisticas': self._calcular_estadisticas(mejores_resultados)
        }
    
    def _calcular_estadisticas(self, resultados: List[Dict]) -> Dict:
        """Calcula estadísticas de los resultados"""
        tiempos = [r['tiempo_total'] for r in resultados]
        
        return {
            'tiempo_minimo': round(min(tiempos), 2),
            'tiempo_maximo': round(max(tiempos), 2),
            'tiempo_promedio': round(np.mean(tiempos), 2),
            'desviacion_estandar': round(np.std(tiempos), 2),
            'total_resultados': len(resultados)
        }
    
    def estrategias_deterministicas(self) -> Dict[str, Dict]:
        """
        Implementa y evalúa estrategias determinísticas predefinidas
        
        Returns:
            Diccionario con resultados de cada estrategia
        """
        print("\nEvaluando estrategias determinísticas...")
        
        estrategias = {}
        
        # Estrategia A: Dividir en "izquierda" (R1-R3) y "derecha" (R4-R6)
        estrategia_a = {
            'nombre': 'A - Izq/Der',
            'rescatista1_ruta': ['R1', 'R2', 'R3'],  # Orden natural
            'rescatista2_ruta': ['R4', 'R5', 'R6']   # Orden natural
        }
        estrategias['A'] = self._evaluar_estrategia(estrategia_a)
        
        # Estrategia B: Ordenar por distancia desde Start (más cercano primero)
        distancias_start = [(cuarto, self.matriz_distancias['Start'][cuarto]) 
                          for cuarto in self.cuartos]
        distancias_start.sort(key=lambda x: x[1])
        
        estrategia_b = {
            'nombre': 'B - Más cercano primero',
            'rescatista1_ruta': [cuarto for cuarto, _ in distancias_start[:3]],
            'rescatista2_ruta': [cuarto for cuarto, _ in distancias_start[3:]]
        }
        estrategias['B'] = self._evaluar_estrategia(estrategia_b)
        
        # Estrategia C: Ordenar por distancia desde Start (más lejano primero)
        estrategia_c = {
            'nombre': 'C - Más lejano primero',
            'rescatista1_ruta': [cuarto for cuarto, _ in distancias_start[::-1][:3]],
            'rescatista2_ruta': [cuarto for cuarto, _ in distancias_start[::-1][3:]]
        }
        estrategias['C'] = self._evaluar_estrategia(estrategia_c)
        
        # Estrategia D: Equilibrar por distancia total
        estrategia_d = {
            'nombre': 'D - Equilibrado por distancia',
            'rescatista1_ruta': ['R1', 'R4', 'R5'],  # Distancias variadas
            'rescatista2_ruta': ['R2', 'R3', 'R6']   # Distancias variadas
        }
        estrategias['D'] = self._evaluar_estrategia(estrategia_d)
        
        # Estrategia E: Minimizar tiempo individual
        estrategia_e = {
            'nombre': 'E - Optimizado manualmente',
            'rescatista1_ruta': ['R1', 'R6', 'R2'],  # Ruta optimizada
            'rescatista2_ruta': ['R3', 'R4', 'R5']   # Ruta optimizada
        }
        estrategias['E'] = self._evaluar_estrategia(estrategia_e)
        
        return estrategias
    
    def _evaluar_estrategia(self, estrategia: Dict) -> Dict:
        """Evalúa una estrategia específica"""
        ruta1 = estrategia['rescatista1_ruta']
        ruta2 = estrategia['rescatista2_ruta']
        
        tiempo1 = self._calcular_tiempo_rescatista(ruta1)
        tiempo2 = self._calcular_tiempo_rescatista(ruta2)
        tiempo_total = max(tiempo1, tiempo2)
        
        return {
            'nombre': estrategia['nombre'],
            'rescatista1_ruta': ruta1,
            'rescatista2_ruta': ruta2,
            'tiempo1': tiempo1,
            'tiempo2': tiempo2,
            'tiempo_total': tiempo_total
        }
    
    def simulacion_monte_carlo(self, estrategia: str, num_simulaciones: int = 1000) -> Dict:
        """
        Realiza simulación Monte Carlo para una estrategia específica
        
        Args:
            estrategia: Clave de la estrategia ('A', 'B', 'C', 'D', 'E')
            num_simulaciones: Número de simulaciones a realizar
            
        Returns:
            Estadísticas de la simulación
        """
        print(f"\nIniciando simulación Monte Carlo para estrategia {estrategia}...")
        print(f"   Simulaciones: {num_simulaciones:,}")
        
        # Obtener la estrategia base
        estrategias = self.estrategias_deterministicas()
        estrategia_base = estrategias[estrategia]
        
        tiempos_totales = []
        
        for i in range(num_simulaciones):
            # Variar tiempos de búsqueda (±20%)
            tiempo_busqueda_var = self.tiempo_busqueda * np.random.uniform(0.8, 1.2)
            
            # Variar distancias de caminar (±10%)
            matriz_var = self._crear_matriz_distancias_variada()
            
            # Evaluar con valores variados
            tiempo1 = self._calcular_tiempo_rescatista_con_matriz(
                estrategia_base['rescatista1_ruta'], matriz_var, tiempo_busqueda_var
            )
            tiempo2 = self._calcular_tiempo_rescatista_con_matriz(
                estrategia_base['rescatista2_ruta'], matriz_var, tiempo_busqueda_var
            )
            
            tiempo_total = max(tiempo1, tiempo2)
            tiempos_totales.append(tiempo_total)
            
            if (i + 1) % 200 == 0:
                print(f"   Completadas {i + 1:,} simulaciones...")
        
        # Calcular estadísticas
        tiempos_totales = np.array(tiempos_totales)
        
        estadisticas = {
            'estrategia': estrategia_base['nombre'],
            'num_simulaciones': num_simulaciones,
            'tiempo_promedio': np.mean(tiempos_totales),
            'desviacion_estandar': np.std(tiempos_totales),
            'tiempo_minimo': np.min(tiempos_totales),
            'tiempo_maximo': np.max(tiempos_totales),
            'percentil_5': np.percentile(tiempos_totales, 5),
            'percentil_95': np.percentile(tiempos_totales, 95)
        }
        
        print(f"\nSimulación completada")
        return estadisticas
    
    def _crear_matriz_distancias_variada(self) -> Dict[str, Dict[str, int]]:
        """Crea una matriz de distancias con variación aleatoria"""
        matriz_base = self._crear_matriz_distancias()
        matriz_variada = {}
        
        for origen in matriz_base:
            matriz_variada[origen] = {}
            for destino in matriz_base[origen]:
                # Variar ±10%
                valor_base = matriz_base[origen][destino]
                valor_variado = int(valor_base * np.random.uniform(0.9, 1.1))
                matriz_variada[origen][destino] = max(1, valor_variado)  # Mínimo 1 segundo
        
        return matriz_variada
    
    def _calcular_tiempo_rescatista_con_matriz(self, ruta: List[str], matriz: Dict, tiempo_busqueda: float) -> float:
        """Calcula tiempo usando una matriz de distancias específica"""
        if not ruta:
            return 0
        
        tiempo_total = 0
        tiempo_total += matriz['Start'][ruta[0]]
        
        for i in range(len(ruta) - 1):
            tiempo_total += matriz[ruta[i]][ruta[i + 1]]
        
        tiempo_total += tiempo_busqueda * len(ruta)
        return tiempo_total
    
    def imprimir_resultados(self, optimizacion: Dict, estrategias: Dict):
        """Imprime todos los resultados de forma organizada"""
        print("\n" + "="*80)
        print(" RESULTADOS DE OPTIMIZACIÓN DE RUTAS PARA RESCATISTAS")
        print("="*80)
        
        # Parámetros del escenario
        print(f"\n PARÁMETROS DEL ESCENARIO (Escenario Estándar):")
        print(f"   Velocidad de rescatistas: {self.velocidad_rescatistas} m/s (FEMA para interiores)")
        print(f"   Tiempo de revisión por cuarto: {self.tiempo_busqueda} segundos (NFPA 1670)")
        print(f"   Disposición física: Pasillo lineal")
        print(f"   Distancias físicas:")
        for ubicacion, distancia in self.distancias_fisicas.items():
            print(f"      {ubicacion}: {distancia}m")
        
        # Resultados de optimización
        mejor = optimizacion['mejor']
        print(f"\n SOLUCIÓN ÓPTIMA ENCONTRADA:")
        print(f"   Tiempo total mínimo: {mejor['tiempo_total']:.2f} segundos")
        print(f"   Rescatista 1: {mejor['rescatista1_ruta']} ({mejor['tiempo1']:.2f}s)")
        print(f"   Rescatista 2: {mejor['rescatista2_ruta']} ({mejor['tiempo2']:.2f}s)")
        print(f"   Combinaciones evaluadas: {optimizacion['total_evaluadas']:,}")
        
        # Estadísticas generales
        stats = optimizacion['resumen_estadisticas']
        print(f"\n ESTADÍSTICAS GENERALES:")
        print(f"   Tiempo promedio: {stats['tiempo_promedio']:.1f} segundos")
        print(f"   Desviación estándar: {stats['desviacion_estandar']:.1f} segundos")
        print(f"   Rango: {stats['tiempo_minimo']:.1f} - {stats['tiempo_maximo']:.1f} segundos")
        
        # Top 5 resultados
        print(f"\nTOP 5 MEJORES SOLUCIONES:")
        for i, resultado in enumerate(optimizacion['top_10'][:5], 1):
            print(f"   {i}. {resultado['tiempo_total']:.2f}s - R1:{resultado['rescatista1_ruta']} | R2:{resultado['rescatista2_ruta']}")
        
        # Estrategias determinísticas
        print(f"\n ESTRATEGIAS DETERMINÍSTICAS:")
        for clave, estrategia in estrategias.items():
            print(f"   {clave}: {estrategia['nombre']}")
            print(f"      R1: {estrategia['rescatista1_ruta']} ({estrategia['tiempo1']:.2f}s)")
            print(f"      R2: {estrategia['rescatista2_ruta']} ({estrategia['tiempo2']:.2f}s)")
            print(f"      Total: {estrategia['tiempo_total']:.2f}s")
            print()
    
    def simulacion_monte_carlo_todas_estrategias(self, num_simulaciones: int = 1000) -> Dict:
        """
        Realiza simulación Monte Carlo para TODAS las estrategias determinísticas
        
        Args:
            num_simulaciones: Número de simulaciones por estrategia
            
        Returns:
            Diccionario con estadísticas de Monte Carlo para cada estrategia
        """
        print(f"\n Iniciando simulación Monte Carlo para TODAS las estrategias...")
        print(f"   Simulaciones por estrategia: {num_simulaciones:,}")
        
        estrategias = self.estrategias_deterministicas()
        resultados_monte_carlo = {}
        
        # Ordenar estrategias por tiempo (mejor primero)
        estrategias_ordenadas = sorted(estrategias.items(), key=lambda x: x[1]['tiempo_total'])
        
        for clave_estrategia, estrategia_data in estrategias_ordenadas:
            print(f"\n--- Analizando {clave_estrategia}: {estrategia_data['nombre']} ---")
            
            # Ejecutar simulación para esta estrategia
            resultado = self.simulacion_monte_carlo(clave_estrategia, num_simulaciones)
            resultados_monte_carlo[clave_estrategia] = resultado
            
            print(f"Completada: {resultado['tiempo_promedio']:.1f}±{resultado['desviacion_estandar']:.1f}s")
        
        return resultados_monte_carlo
    
    def guardar_resultados(self, optimizacion: Dict, estrategias: Dict, monte_carlo: Dict):
        """Guarda los resultados en archivos JSON"""
        resultados_completos = {
            'optimizacion_exhaustiva': optimizacion,
            'estrategias_deterministicas': estrategias,
            'simulacion_monte_carlo': monte_carlo,
            'parametros': {
                'tiempo_busqueda': self.tiempo_busqueda,
                'velocidad_rescatistas': self.velocidad_rescatistas,
                'distancias_fisicas': self.distancias_fisicas,
                'cuartos': self.cuartos,
                'num_rescatistas': self.num_rescatistas,
                'justificacion_velocidad': 'Velocidad promedio en interiores según protocolos FEMA para movimiento seguro en espacios con posibles obstáculos',
                'justificacion_tiempo': 'Tiempo estándar para búsqueda visual rápida según NFPA 1670'
            }
        }
        
        with open('/resultados_rescatistas.json', 'w', encoding='utf-8') as f:
            json.dump(resultados_completos, f, indent=2, ensure_ascii=False)
        with open('resultados_rescatistas.json', 'w', encoding='utf-8') as f:
            json.dump(resultados_completos, f, indent=2, ensure_ascii=False)
        
        print(f"Resultados guardados en 'resultados_rescatistas.json'")
def main():
    """Función principal del programa"""
    print("OPTIMIZADOR DE RUTAS PARA RESCATISTAS")
    print("=" * 50)
    
    # Crear el optimizador con parámetros estándar FEMA/NFPA
    optimizador = OptimizadorRescatistas(tiempo_busqueda=30, velocidad_rescatistas=1.2)
    
    # Ejecutar optimización exhaustiva
    optimizacion = optimizador.optimizar_rutas()
    
    # Evaluar estrategias determinísticas
    estrategias = optimizador.estrategias_deterministicas()
    
    # Ejecutar simulación Monte Carlo para TODAS las estrategias
    print(f"\n Ejecutando simulación Monte Carlo para TODAS las estrategias...")
    monte_carlo = optimizador.simulacion_monte_carlo_todas_estrategias(1000)
    
    # Mostrar resultados
    optimizador.imprimir_resultados(optimizacion, estrategias)
    
    # Mostrar resultados Monte Carlo
    print(f"\n" + "="*80)
    print(" SIMULACIÓN MONTE CARLO - ANÁLISIS DE ROBUSTEZ PARA TODAS LAS ESTRATEGIAS")
    print("="*80)
    
    for clave, resultado in monte_carlo.items():
        print(f"\n ESTRATEGIA {clave}: {resultado['estrategia']}")
        print(f"   Tiempo promedio: {resultado['tiempo_promedio']:.1f} segundos")
        print(f"   Desviación estándar: {resultado['desviacion_estandar']:.1f} segundos")
        print(f"   Coeficiente de variación: {(resultado['desviacion_estandar']/resultado['tiempo_promedio']*100):.1f}%")
        print(f"   Rango (5%-95%): {resultado['percentil_5']:.1f} - {resultado['percentil_95']:.1f} segundos")
        print(f"   Rango completo: {resultado['tiempo_minimo']:.1f} - {resultado['tiempo_maximo']:.1f} segundos")
    
    # Mostrar ranking de robustez (menor coeficiente de variación = más robusta)
    print(f"\n RANKING DE ROBUSTEZ (menor variabilidad = más robusta):")
    robustez_ranking = sorted(monte_carlo.items(), 
                             key=lambda x: x[1]['desviacion_estandar']/x[1]['tiempo_promedio'])
    
    for i, (clave, resultado) in enumerate(robustez_ranking, 1):
        cv = (resultado['desviacion_estandar']/resultado['tiempo_promedio']*100)
        print(f"   {i}. {clave} ({resultado['estrategia']}): CV = {cv:.1f}%")
    
    # Guardar resultados
    optimizador.guardar_resultados(optimizacion, estrategias, monte_carlo)


if __name__ == "__main__":
    main()