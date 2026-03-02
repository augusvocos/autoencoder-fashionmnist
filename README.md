# Autoencoder Convolucional para Clasificación en FashionMNIST

Este repositorio documenta el desarrollo, entrenamiento y evaluación de un modelo de autoencoder convolucional aplicado al conjunto de datos FashionMNIST. El objetivo general del trabajo fue explorar la utilidad de representaciones latentes generadas por modelos no supervisados para tareas de clasificación supervisada, analizando tanto el comportamiento del modelo como las implicancias metodológicas detrás de cada decisión de diseño.

## Motivación

El uso de autoencoders como herramienta de reducción de dimensionalidad y extracción de características es útil en contextos donde se busca aprender representaciones de los datos sin requerir etiquetas. En este proyecto, el foco estuvo puesto en evaluar si esas representaciones pueden ser aprovechadas en un segundo paso por modelos de clasificación, y bajo qué condiciones su uso es efectivo. 

El trabajo se dividió en dos etapas principales:

### 1. Reconstrucción

Se diseñó un autoencoder convolucional con PyTorch, compuesto por:

- Un **encoder** con dos capas convolucionales, max pooling y una capa lineal que proyecta a un espacio latente de dimensión ajustable (`n`).
- Un **decoder** simétrico, que reconstruye las imágenes a partir de la representación latente mediante capas lineales y convolucionales transpuestas.

Se implementó una búsqueda sistemática de hiperparámetros basada en un árbol de decisiones secuencial. Se evaluaron distintos valores para:

- Dimensión del espacio latente (`n`)
- Probabilidad de dropout
- Tasa de aprendizaje
- Tamaño del lote
- Optimizador (SGD vs. Adam)
- Arquitectura (dos variantes distintas)

El criterio de selección fue el error cuadrático medio (MSE) sobre el conjunto de validación, procurando evitar el sobreajuste.

### 2. Clasificación

Una vez entrenado el autoencoder, se diseñó un clasificador simple que toma como entrada el vector latente (`n = 256`) y produce una salida de 10 clases. Se evaluaron tres configuraciones:

- Entrenamiento conjunto del encoder y el clasificador
- Clasificador entrenado por separado con encoder preentrenado
- Clasificador entrenado por separado con encoder no entrenado

Los modelos fueron comparados en términos de error, precisión y estabilidad de la validación cruzada, complementando los resultados con gráficas y una matriz de confusión.

## Resultados principales

- La **arquitectura alternativa** del autoencoder, junto con la configuración `n = 256`, `dropout = 0.1`, `learning rate = 0.001`, `batch size = 50` y optimizador **Adam**, fue la más efectiva en reconstrucción.
- Entrenar el encoder y el clasificador **simultáneamente** mejora la precisión predictiva, aunque genera mayor sobreajuste.
- El clasificador **entrenado por separado** (con o sin encoder preentrenado) presenta menor riesgo de sobreajuste, pero una capacidad predictiva más limitada.
- Sorprendentemente, la representación latente generada por el encoder no mejora significativamente la clasificación si no se entrena en conjunto con el clasificador, lo que invita a reflexionar sobre la relación entre objetivos de reconstrucción y discriminación.
