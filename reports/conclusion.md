# Conclusión · Ejercicio 04

**INF-8239 Ciencia de Datos II · Unidad 02** · Alexis De La Cruz · 100805370

<!-- inicio -->
El ejercicio pedía comparar una línea base con una CNN sobre Fashion-MNIST y justificar la elección
con su costo. El resultado más útil fue el que no esperaba: **la CNN del proyecto base perdió contra
la línea base densa**, 0.765 de F1 macro frente a 0.855, y además tardó diez veces más en entrenar,
43 segundos contra 4. Una red convolucional más lenta y peor que aplanar la imagen y poner una capa
densa no es un resultado cómodo, pero es el que salió.

En lugar de aceptarlo o de cambiar de modelo, busqué la causa. Las dos redes comparten las mismas
capas convolucionales; la diferencia está en cómo resumen el mapa de características antes de
decidir. La del proyecto base usa un promedio global: convierte cada uno de los 64 mapas de 7×7 en
un solo número, de modo que conserva **qué** rasgos aparecieron y descarta **dónde**. En
Fashion-MNIST la posición es información: una camisa y un bolso pueden compartir texturas, pero no
la distribución de bordes dentro del marco de 28×28.

Para comprobarlo entrené una tercera configuración que cambia una sola cosa: sustituir ese promedio
por un aplanado que conserva la rejilla completa. Subió a **0.901 de F1 macro**, 13.6 puntos por
encima de la CNN original y 4.6 por encima de la línea base. El costo de esa decisión está casi
todo en la última capa: pasa de 19 466 a 50 186 parámetros, prácticamente los mismos 50 890 de la
línea base densa, y el archivo crece de 79 a 199 KiB. El extractor de rasgos no cambió.

El análisis por clase explica dónde está el techo. Las prendas de torso concentran los errores:
camisa falla en el 27 % de los casos y se confunde sobre todo con camiseta; camiseta falla en el
21 % y se confunde con camisa; abrigo y pulóver se confunden entre sí y con camisa. En cambio, el
calzado y los bolsos rondan el 2 o 3 % de error. Tiene sentido: a 28×28 píxeles y en escala de
grises, lo que distingue una camisa de una camiseta son los botones o el cuello, detalles que a esa
resolución casi desaparecen. No es un problema del modelo, es un límite de los datos.

Dos limitaciones que conviene declarar. La primera: Fashion-MNIST son imágenes centradas, del mismo
tamaño y sin fondo, así que nada de esto anticipa el rendimiento con fotografías reales. La segunda:
el entrenamiento se detuvo a las seis épocas y la variante aún mejoraba, de modo que 0.901 es un
piso y no un techo.

Antes de usar algo así en producción entrenaría con más épocas y con aumento de datos, mediría
contra fotografías reales en lugar de recortes limpios, y revisaría el par camisa-camiseta con un
umbral o una clase combinada, porque dos de cada diez camisas mal clasificadas es demasiado para
cualquier decisión automática.
<!-- fin -->
