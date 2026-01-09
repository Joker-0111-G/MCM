# 🏆 Resumen Detallado - Optimización de Rutas para Rescatistas

## ✨ Solución Óptima Encontrada

**Tiempo Total Mínimo:** 113 segundos

**Asignación de Cuartos:**
- **Rescatista 1:** ['R1', 'R2', 'R3'] (Tiempo: 110s)
- **Rescatista 2:** ['R6', 'R5', 'R4'] (Tiempo: 113s)

## 📊 Análisis de Eficiencia

- **Total de combinaciones evaluadas:** 3,600
- **Tiempo promedio:** 166.0 segundos
- **Desviación estándar:** 28.9 segundos
- **Rango de tiempos:** 113 - 214 segundos
- **Mejora vs promedio:** 31.9%

## 🥇 Top 5 Mejores Soluciones

| Ranking | Tiempo Total | Rescatista 1 | Rescatista 2 |
|---------|--------------|--------------|--------------|
| 1 | 113s | ['R1', 'R2', 'R3'] | ['R6', 'R5', 'R4'] |
| 2 | 113s | ['R1', 'R3', 'R2'] | ['R6', 'R5', 'R4'] |
| 3 | 113s | ['R3', 'R2', 'R1'] | ['R6', 'R5', 'R4'] |
| 4 | 113s | ['R6', 'R1', 'R2'] | ['R3', 'R4', 'R5'] |
| 5 | 113s | ['R1', 'R6', 'R5'] | ['R3', 'R2', 'R4'] |

## 🎯 Estrategias Determinísticas

### Estrategia A: A - Izq/Der

- **Ruta Rescatista 1:** ['R1', 'R2', 'R3'] (110s)
- **Ruta Rescatista 2:** ['R4', 'R5', 'R6'] (119s)
- **Tiempo Total:** 119s
- **Diferencia vs óptimo:** 6s

### Estrategia B: B - Más cercano primero

- **Ruta Rescatista 1:** ['R1', 'R6', 'R3'] (118s)
- **Ruta Rescatista 2:** ['R5', 'R2', 'R4'] (118s)
- **Tiempo Total:** 118s
- **Diferencia vs óptimo:** 5s

### Estrategia C: C - Más lejano primero

- **Ruta Rescatista 1:** ['R4', 'R2', 'R5'] (122s)
- **Ruta Rescatista 2:** ['R3', 'R6', 'R1'] (120s)
- **Tiempo Total:** 122s
- **Diferencia vs óptimo:** 9s

### Estrategia D: D - Equilibrado por distancia

- **Ruta Rescatista 1:** ['R1', 'R4', 'R5'] (117s)
- **Ruta Rescatista 2:** ['R2', 'R3', 'R6'] (121s)
- **Tiempo Total:** 121s
- **Diferencia vs óptimo:** 8s

### Estrategia E: E - Optimizado manualmente

- **Ruta Rescatista 1:** ['R1', 'R6', 'R2'] (115s)
- **Ruta Rescatista 2:** ['R3', 'R4', 'R5'] (113s)
- **Tiempo Total:** 115s
- **Diferencia vs óptimo:** 2s

## 🎲 Simulación Monte Carlo

**Estrategia evaluada:** E - Optimizado manualmente
**Número de simulaciones:** 1,000
**Tiempo promedio:** 113.8 segundos
**Desviación estándar:** 10.7 segundos
**Rango (5%-95%):** 97.2 - 129.6 segundos
**Mejor caso:** 94.3 segundos
**Peor caso:** 132.7 segundos

## 🎯 Conclusiones y Recomendaciones

1. **Solución Óptima:** La mejor asignación es dividir los cuartos R1-R3 para el Rescatista 1 y R6-R5-R4 para el Rescatista 2, logrando un tiempo total de 113 segundos.

2. **Eficiencia:** Esta solución es 31.9% más eficiente que el promedio de todas las combinaciones.

3. **Robustez:** La simulación Monte Carlo muestra que la estrategia E es consistente con una desviación estándar de solo 10.7 segundos.

4. **Distribución equilibrada:** El balance de carga entre rescatistas es óptimo (110s vs 113s).

5. **Flexibilidad:** Existen múltiples soluciones con el mismo tiempo óptimo, proporcionando alternativas viables.
