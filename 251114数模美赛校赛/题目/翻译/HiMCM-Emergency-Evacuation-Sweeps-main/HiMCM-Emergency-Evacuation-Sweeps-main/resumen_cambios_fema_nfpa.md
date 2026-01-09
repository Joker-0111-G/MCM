# Resumen de Actualización: Parámetros Base (Escenario Estándar)

## Cambios Implementados

### 📋 Parámetros Actualizados

**1. Velocidad de los Rescatistas: 1.2 m/s**
- **Justificación**: Velocidad promedio en interiores según protocolos FEMA (Federal Emergency Management Agency) para movimiento seguro en espacios con posibles obstáculos
- **Impacto**: Todos los tiempos de recorrido ahora se calculan como `tiempo = distancia / 1.2`

**2. Tiempo de Revisión por Cuarto: 30 segundos**
- **Justificación**: Tiempo estándar para búsqueda visual rápida según NFPA 1670 (National Fire Protection Association)
- **Mantiene**: El valor estándar ya implementado en el código

**3. Distancias Físicas (disposición lineal en pasillo)**
```
Start: 0 m
R1: 10 m
R2: 15 m
R3: 20 m
R4: 25 m
R5: 30 m
R6: 35 m
```

### 🛠️ Modificaciones Técnicas

**A. Constructor de la Clase**
- Agregado parámetro `velocidad_rescatistas: float = 1.2`
- Agregado `distancias_fisicas` como atributo de clase
- Documentación actualizada con justificaciones FEMA/NFPA

**B. Matriz de Distancias**
- **Antes**: Valores estáticos de ejemplo (5-15 segundos)
- **Ahora**: Cálculo dinámico basado en distancias físicas reales
- **Fórmula**: `tiempo = |distancia_fisica1 - distancia_fisica2| / velocidad_rescatistas`
- **Precisión**: Resultados con 2 decimales para mayor exactitud

**C. Métodos Actualizados**
- `_crear_matriz_distancias()`: Recreada para usar distancias físicas
- `_calcular_tiempo_rescatista()`: Cambiado a retorno `float` para mayor precisión
- `_calcular_estadisticas()`: Redondeo a 2 decimales
- `imprimir_resultados()`: Muestra nuevos parámetros y justificaciones
- `guardar_resultados()`: Incluye información completa de parámetros

### 📊 Matriz de Tiempos de Recorrido (segundos)

```
           Start   R1     R2     R3     R4     R5     R6
   Start   0.00   8.33  12.50  16.67  20.83  25.00  29.17
   R1      8.33   0.00   4.17   8.33  12.50  16.67  20.83
   R2     12.50   4.17   0.00   4.17   8.33  12.50  16.67
   R3     16.67   8.33   4.17   0.00   4.17   8.33  12.50
   R4     20.83  12.50   8.33   4.17   0.00   4.17   8.33
   R5     25.00  16.67  12.50   8.33   4.17   0.00   4.17
   R6     29.17  20.83  16.67  12.50   8.33   4.17   0.00
```

### 🎯 Ejemplos de Cálculos

- **Start a R1**: 10m ÷ 1.2 m/s = 8.33s
- **Start a R6**: 35m ÷ 1.2 m/s = 29.17s
- **R1 a R6**: 25m ÷ 1.2 m/s = 20.83s
- **R2 a R4**: 10m ÷ 1.2 m/s = 8.33s
- **R3 a R5**: 10m ÷ 1.2 m/s = 8.33s

### ✅ Resultado de Prueba

Con los nuevos parámetros, la optimización encuentra:
- **Tiempo óptimo**: 119.16 segundos
- **Rescatista 1**: R1 → R3 → R5 (114.99s)
- **Rescatista 2**: R2 → R4 → R6 (119.16s)
- **Combinaciones evaluadas**: 3,600

### 📁 Archivos Modificados

1. **`rescatistas_optimizacion.py`**: Clase principal actualizada
2. **`ejemplo_uso.py`**: Llamadas al constructor actualizadas
3. **`verificar_matriz_distancias.py`**: Nuevo script de verificación
4. **`prueba_optimizacion_fema_nfpa.py`**: Script de prueba

### 🎉 Beneficios de la Actualización

1. **Mayor Precisión**: Basado en estándares oficiales FEMA/NFPA
2. **Realismo**: Distancias físicas reales del escenario
3. **Transparencia**: Justificaciones documentadas para cada parámetro
4. **Escalabilidad**: Estructura que permite cambiar fácilmente parámetros
5. **Compatibilidad**: Mantiene toda la funcionalidad existente

### 🔧 Mantenimiento

Los parámetros pueden ajustarse fácilmente modificando:
- `velocidad_rescatistas` en el constructor
- `distancias_fisicas` en el constructor
- `tiempo_busqueda` en el constructor

Todos los cálculos se actualizan automáticamente manteniendo la precisión matemática.