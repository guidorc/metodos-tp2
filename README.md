# Métodos Numéricos - TP Nº 2

Trabajo Practico Nro 2 de Métodos Numéricos 1c 2023

Usamos la biblioteca de algebra lineal para C++, Eigen 3.4.0.

### Metodo de la potencia con deflación:
Parámetros:

- Nombre del archivo de texto a ejecutar desde el directorio `/matrices`
- Nro de iteraciones para método de la potencia
- Tolerancia para convergencia del método de la potencia
- Cantidad de Componentes *k* (opcional): Determina la cantidad de pares 
autovalor/autovector que se calculan con deflación

Ejecucion:

  - Buildear: `cmake CMakeLists.txt`
  - Compilar: `make`
  - Ejecutar: `./tp2 <archivo> <#iteraciones> <tolerancia> <k>`

### Experimentación en Python:
El archivo desde donde se ejecutan los experimentos es `main.py`, 
para ejecutar un experimento en particular, 
comentar el resto de items ubicados en el método main