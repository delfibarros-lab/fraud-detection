# Detección de fraude con optimización de costo de negocio

Modelo de clasificación para detectar transacciones fraudulentas en un escenario con **fuerte desbalance de clases (~1.4% de fraude)**. En lugar de optimizar solo una métrica estadística, el umbral de decisión se elige minimizando el **costo económico**: lo que se pierde por fraudes no detectados más el costo de revisar manualmente cada alerta.

> **English summary:** Fraud detection on a highly imbalanced synthetic dataset. Compares logistic regression and gradient boosting using a time-based split, and picks the decision threshold on a separate validation window by minimizing business cost (missed fraud amount + manual review cost). Cuts total cost by about 33% on the held-out test period.

## Resultados (período de prueba, split temporal)

| Modelo | PR-AUC | ROC-AUC |
|--------|-------:|--------:|
| Regresión logística | 0.155 | 0.821 |
| **Gradient Boosting** | **0.162** | 0.821 |

Con el umbral elegido en validación (0.04), el modelo detecta **44% de los fraudes** con 10% de precisión y **reduce el costo total en 33%** frente a no tener ningún modelo (USD 65.3k → 43.6k sobre el período de prueba).

El PR-AUC de referencia al azar es 0.014, así que el modelo rinde unas 11 veces mejor. Los datos son sintéticos con ruido deliberado, por lo que los números absolutos no son comparables con un caso real.

| Curva Precision-Recall | Costo según umbral |
|:--:|:--:|
| ![pr](reports/pr_curve.png) | ![cost](reports/cost_curve.png) |

## Decisiones de diseño

- **Split temporal** en lugar de aleatorio: se entrena con el pasado y se evalúa en el futuro, como en producción.
- **Tres ventanas:** entrenamiento, validación (elige el modelo y el umbral) y prueba (solo se mira al final), para no sobreestimar el resultado.
- **Métrica correcta para desbalance:** PR-AUC en lugar de exactitud.
- **Umbral por costo:** el umbral de 0.5 por defecto ignora que perder un fraude de USD 500 cuesta mucho más que revisar una alerta.
- **Variables de comportamiento:** relación del monto con el promedio histórico del cliente, horario nocturno, dispositivo nuevo, frecuencia en 24 horas.

## Cómo correrlo

```bash
pip install -r requirements.txt
cd src
python run.py
```

Genera los datos, entrena, evalúa y guarda métricas y gráficos en `reports/`. Tarda unos 10 segundos.

## Estructura

```
src/data.py       generador de transacciones sintéticas
src/features.py   ingeniería de variables y preprocesamiento
src/run.py        entrenamiento, selección de umbral y reportes
reports/          métricas y gráficos
```
