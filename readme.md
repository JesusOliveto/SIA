# TP Final — Sistemas Inteligentes Artificiales (SIA)
**Profesor:** Esp. Ing. Agustín Fernandez  
**Tema:** Implementación de **K-Medias (K-Means)**

## Índice
- [Introducción](#introducción)
- [Requerimientos de implementación](#requerimientos-de-implementación)
- [Objetivos del trabajo](#objetivos-del-trabajo)
- [Aclaraciones importantes](#aclaraciones-importantes)
- [Criterios esperados en la mesa de examen](#criterios-esperados-en-la-mesa-de-examen)

---

## Introducción
El laboratorio de Inteligencia Artificial de la empresa **Ultralistic** solicita realizar una implementación propia del algoritmo **K-Medias**, dado que cuenta con un dataset sobre **calidad vitivinícola** y desea analizar cuáles son las agrupaciones relevantes y “naturales” del mismo, más allá de las clases presentes en el dataset.

---

## Requerimientos de implementación
Para la implementación se establecen los siguientes requerimientos:

a) Se puede escribir en el lenguaje de programación preferido por el grupo.  
b) No se admiten implementaciones que utilicen librerías o frameworks que ya contengan el algoritmo solicitado.  
c) Debe realizarse una implementación del algoritmo **sin vectorizar** y otra **vectorizada**.  
d) Está permitido utilizar librerías/frameworks para soporte de cálculos matemáticos (operaciones matriciales, cálculos estadísticos, etc.).  
e) Las implementaciones deben ser **flexibles**:
   - Permitir setear la cantidad de clusters (**k**).
   - Manejar cualquier número de ejemplos y atributos del dataset.
f) Los tipos de datos soportados deben ser **normalizados** de alguna forma para evitar inconvenientes.  
g) Se debe realizar una **comparativa** entre las implementaciones propias y la implementación de alguna librería/framework disponible en el mercado (puede estar en otro lenguaje).

---

## Objetivos del trabajo
Teniendo en cuenta los requerimientos, se debe lograr:

1) Construir las implementaciones de K-Medias solicitadas cumpliendo con los requerimientos.  
2) Realizar una comparativa (puede ser gráfica) con **múltiples ejecuciones** del dataset para diferentes valores de **k**, utilizando:
   - Implementación propia no vectorizada
   - Implementación propia vectorizada
   - Implementación de terceros
3) Las implementaciones propias deben:
   - Tener una **interfaz amigable**
   - Permitir “predecir” a qué cluster pertenecería un elemento (registro) **fuera del dataset**

---

## Aclaraciones importantes
- El TP puede realizarse en **grupos de hasta 2 integrantes** y deben estar **todos** los integrantes en la mesa examinadora.  
- Se pueden solicitar al profesor (por **TEAMS**) clases de consulta referidas al TP final luego de que finalice el cursado.

---

## Criterios esperados en la mesa de examen
En la mesa de examen final se espera:

| Ítem | Descripción | Peso |
|------|-------------|------|
| 1 | Presentación de un informe en formato **PDF**, con prolijidad acorde | 10% |
| 2 | Explicación teórico/práctica del TP por parte de **todos** los integrantes | 40% |
| 3 | Construir una aplicación que **corra sin errores** | 20% |
| 4 | Interfaz gráfica funcional para interactuar con las implementaciones propias + alguna funcionalidad extra | 30% |
