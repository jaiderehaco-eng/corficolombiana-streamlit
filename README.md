# 📊 Análisis Bursátil — Corficolombiana (CORFICOLCF.CL)

> ¿Qué explica el comportamiento bursátil de Corficolombiana en la BVC?  
> Variables regresoras: **USD/COP** y **COLCAP**  
> Fuente de datos: Yahoo Finance (tiempo real)

---

## Estructura del proyecto

```
corficolombiana/
├── app.py                          # Dashboard Streamlit
├── corficolombiana_analisis.ipynb  # Google Colab — análisis y conceptos
├── requirements.txt                # Dependencias Python
└── README.md                       # Este archivo
```

---

## Entregable 1 — Google Colab

**Archivo:** `corficolombiana_analisis.ipynb`

### Cómo usarlo

1. Abrir [colab.research.google.com](https://colab.research.google.com)
2. `Archivo → Subir notebook` → seleccionar el `.ipynb`
3. `Entorno de ejecución → Ejecutar todo` (`Ctrl+F9`)
4. Los datos se descargan automáticamente desde Yahoo Finance.

### Contenido del notebook

| Sección | Descripción |
|---|---|
| 1. Introducción | Contexto financiero de Corficolombiana, COLCAP y USD/COP |
| 2. Exploración de variables | Por qué estas variables y no otras (Open, High, etc.) |
| 3. Descarga de datos | yfinance: CORFICOLCF.CL, COP=X, ^COLCAP |
| 4. Estadísticas descriptivas | CV, rango, media, dispersión |
| 5. Exploración visual | Series temporales, doble eje, base 100, scatter, heatmap |
| 6. Regresión lineal | Modelo, coeficientes, R², RMSE, MAE |
| 7. Interpretación financiera | Análisis dinámico adaptado a resultados reales |
| 8. Limitaciones | Causalidad, no-linealidad, factores no observados |
| 9. Defensa conceptual | Preguntas del profesor + respuestas claras |
| 10. Resumen ejecutivo | Síntesis de hallazgos |

---

## Entregable 2 — Dashboard Streamlit

**Archivo:** `app.py`

### Opción A — Ejecución local

```bash
# 1. Clonar o copiar los archivos del proyecto
# 2. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar
streamlit run app.py
```

El dashboard abre en `http://localhost:8501`

### Opción B — Streamlit Community Cloud (producción)

1. Crear repositorio en GitHub con los tres archivos:
   - `app.py`
   - `requirements.txt`
   - `README.md` (opcional)

2. Ir a [share.streamlit.io](https://share.streamlit.io)

3. Conectar cuenta de GitHub → seleccionar el repositorio

4. Configurar:
   - **Main file path:** `app.py`
   - **Python version:** 3.11

5. Click en **Deploy** → URL pública generada en ~2 minutos

> **Importante:** No se requieren archivos CSV ni credenciales.  
> Los datos se descargan automáticamente desde Yahoo Finance en cada sesión.

---

## Variables del modelo

| Rol | Variable | Ticker Yahoo Finance | Justificación |
|---|---|---|---|
| **Dependiente (Y)** | Precio cierre Corficolombiana | `CORFICOLCF.CL` | Objeto de estudio |
| **Regresora X₁** | Tipo de cambio USD/COP | `COP=X` | Riesgo-país, flujo de capitales |
| **Regresora X₂** | Índice COLCAP | `^COLCAP` | Sentimiento bursátil colombiano |

---

## Notas académicas

- El modelo es **explicativo**, no predictivo.
- Se usa regresión lineal múltiple con split 80/20 respetando el orden cronológico.
- Las correlaciones se interpretan como **asociaciones estadísticas**, no causalidades.
- El R² se presenta como medida de explicación parcial.
- El dashboard actualiza **todos** los cálculos cuando cambia el filtro de período.
