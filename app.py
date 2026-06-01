"""
================================================================================
  DASHBOARD BURSÁTIL — Corficolombiana (CORFICOLCF.CL)
  ¿Qué explica el comportamiento bursátil de Corficolombiana?
  Variables: USD/COP · COLCAP
  Fuente: Yahoo Finance (datos en tiempo real)
================================================================================
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Corficolombiana — Análisis Bursátil",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# PALETA Y TOKENS DE DISEÑO
# Inspiración: Bloomberg / Financial Times — oscuro, limpio, refinado
# ─────────────────────────────────────────────────────────────────────────────
C = {
    "bg":        "#09090B",   # fondo principal
    "bg2":       "#111115",   # fondo secundario
    "surface":   "#18181B",   # tarjetas
    "border":    "#27272A",   # bordes
    "text":      "#FAFAFA",   # texto principal
    "muted":     "#71717A",   # texto secundario
    "blue":      "#3B82F6",   # acento primario
    "blue_dim":  "#1D4ED8",
    "green":     "#22C55E",   # positivo
    "red":       "#EF4444",   # negativo
    "amber":     "#F59E0B",   # señal
    "purple":    "#8B5CF6",   # terciario
}

# ─────────────────────────────────────────────────────────────────────────────
# CSS — TEMA EJECUTIVO OSCURO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=Inter:wght@300;400;500;600&display=swap');

/* Reset global */
html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background: {C['bg']} !important;
    color: {C['text']};
}}

/* Fondo de la app */
.stApp {{ background: {C['bg']} !important; }}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background: {C['bg2']} !important;
    border-right: 1px solid {C['border']};
}}
section[data-testid="stSidebar"] * {{ color: {C['text']} !important; }}

/* Títulos */
h1, h2, h3 {{ color: {C['text']} !important; font-family: 'Libre Baskerville', serif !important; }}

/* Inputs */
.stSelectbox > div > div, .stRadio > div {{
    background: {C['surface']} !important;
    border-color: {C['border']} !important;
    color: {C['text']} !important;
}}

/* DataFrames */
.stDataFrame {{ border-radius: 8px; overflow: hidden; }}

/* Scrollbars */
::-webkit-scrollbar {{ width: 5px; background: {C['bg']}; }}
::-webkit-scrollbar-thumb {{ background: {C['border']}; border-radius: 10px; }}

/* Expander */
details summary {{
    background: {C['surface']} !important;
    border: 1px solid {C['border']} !important;
    border-radius: 8px !important;
    color: {C['text']} !important;
    padding: 12px 16px !important;
}}
details[open] summary {{ border-bottom-left-radius: 0 !important; border-bottom-right-radius: 0 !important; }}

/* Eliminar padding superior */
.block-container {{ padding-top: 1.5rem !important; }}

/* Quitar footer */
footer {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# COMPONENTES DE UI REUTILIZABLES
# ─────────────────────────────────────────────────────────────────────────────
def card(content: str, padding: str = "20px 24px", border_left: str = None):
    border_style = f"border-left: 3px solid {border_left};" if border_left else ""
    st.markdown(f"""
    <div style='background:{C["surface"]};border:1px solid {C["border"]};
        border-radius:10px;padding:{padding};{border_style}'>
        {content}
    </div>""", unsafe_allow_html=True)

def kpi(label: str, value: str, delta: str = None, color: str = None):
    clr = color or C["blue"]
    delta_html = ""
    if delta is not None:
        dc = C["green"] if not delta.startswith("-") else C["red"]
        delta_html = f"<div style='font-size:12px;color:{dc};margin-top:4px;font-weight:500;'>{delta}</div>"
    st.markdown(f"""
    <div style='background:{C["surface"]};border:1px solid {C["border"]};
        border-radius:10px;padding:18px 20px;text-align:center;'>
        <div style='font-size:11px;color:{C["muted"]};text-transform:uppercase;
            letter-spacing:1.5px;font-weight:500;margin-bottom:8px;'>{label}</div>
        <div style='font-family:"Libre Baskerville",serif;font-size:26px;
            color:{clr};line-height:1;'>{value}</div>
        {delta_html}
    </div>""", unsafe_allow_html=True)

def section_title(title: str, subtitle: str = ""):
    sub_html = f"<div style='font-size:13px;color:{C['muted']};margin-top:4px;font-weight:400;'>{subtitle}</div>" if subtitle else ""
    st.markdown(f"""
    <div style='margin:40px 0 20px;'>
        <div style='font-family:"Libre Baskerville",serif;font-size:20px;
            color:{C["text"]};font-weight:700;'>{title}</div>
        {sub_html}
    </div>""", unsafe_allow_html=True)

def insight_text(text: str, icon: str = "💡"):
    st.markdown(f"""
    <div style='background:{C["bg2"]};border:1px solid {C["border"]};
        border-radius:8px;padding:14px 18px;margin-top:12px;
        font-size:13.5px;color:#A1A1AA;line-height:1.75;'>
        <span style='color:{C["amber"]};'>{icon}</span>&nbsp; {text}
    </div>""", unsafe_allow_html=True)

def r2_label(value: float) -> str:
    return f"{max(value, 0) * 100:.1f}% varianza explicada" if value >= 0 else "Sin poder explicativo fuera de muestra"

def r2_explained_pct(value: float) -> float:
    return max(value, 0) * 100

def r2_unexplained_pct(value: float) -> float:
    return min(max(1 - max(value, 0), 0), 1) * 100

# Layout de Plotly base
def plotly_base(height: int = 360) -> dict:
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=C["muted"], size=11),
        xaxis=dict(gridcolor=C["border"], showline=False, zeroline=False,
                   tickfont=dict(size=10)),
        yaxis=dict(gridcolor=C["border"], showline=False, zeroline=False,
                   tickfont=dict(size=10)),
        margin=dict(l=10, r=10, t=40, b=30),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)",
                    font=dict(size=11, color=C["muted"])),
        height=height,
    )

# ─────────────────────────────────────────────────────────────────────────────
# DESCARGA DE DATOS (cacheada 1 hora)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def cargar_datos(period: str) -> pd.DataFrame:
    def _get(ticker: str) -> pd.Series:
        raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if raw.empty:
            raise ValueError(f"Yahoo Finance no devolvio datos para {ticker}.")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        if "Close" not in raw.columns:
            raise ValueError(f"Yahoo Finance no devolvio columna Close para {ticker}.")
        return raw["Close"].copy()

    def _get_first(tickers: list[str]) -> pd.Series:
        errores = []
        for ticker in tickers:
            try:
                return _get(ticker)
            except Exception as exc:
                errores.append(f"{ticker}: {exc}")
        raise ValueError("No se pudo descargar ninguna fuente. " + " | ".join(errores))

    s_corf = _get_first(["CORFICOLCF.CL"])
    s_usdcop = _get_first(["COP=X"])
    s_colcap = _get_first(["^COLCAP", "ICOLCAP.CL"])

    df = pd.concat([s_corf, s_usdcop, s_colcap], axis=1, join="inner")
    df.columns = ["Corficolombiana", "USDCOP", "COLCAP"]
    df.index = pd.to_datetime(df.index)
    df.index.name = "Fecha"
    return df.dropna()

# ─────────────────────────────────────────────────────────────────────────────
# MODELO DE REGRESIÓN
# ─────────────────────────────────────────────────────────────────────────────
def ajustar_modelo(df: pd.DataFrame):
    X = df[["USDCOP", "COLCAP"]]
    y = df["Corficolombiana"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, shuffle=False)
    m = LinearRegression().fit(X_tr, y_tr)
    y_pred = m.predict(X_te)
    y_pred_tr = m.predict(X_tr)
    return dict(
        modelo=m,
        X_tr=X_tr, X_te=X_te,
        y_tr=y_tr, y_te=y_te,
        y_pred=y_pred, y_pred_tr=y_pred_tr,
        r2_tr=r2_score(y_tr, y_pred_tr),
        r2_te=r2_score(y_te, y_pred),
        rmse=np.sqrt(mean_squared_error(y_te, y_pred)),
        mae=mean_absolute_error(y_te, y_pred),
        b0=m.intercept_,
        b1=m.coef_[0],
        b2=m.coef_[1],
    )

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='padding:8px 0 20px;'>
        <div style='font-family:"Libre Baskerville",serif;font-size:17px;
            color:{C["text"]};font-weight:700;'>Corficolombiana</div>
        <div style='font-size:12px;color:{C["muted"]};margin-top:2px;'>
            CORFICOLCF.CL · BVC · COP
        </div>
    </div>
    <hr style='border:none;border-top:1px solid {C["border"]};margin:0 0 20px;'>
    """, unsafe_allow_html=True)

    periodo_map = {
        "3 meses": "3mo",
        "6 meses": "6mo",
        "1 año":   "1y",
        "2 años":  "2y",
    }
    periodo_label = st.selectbox(
        "Período de análisis",
        list(periodo_map.keys()),
        index=2,
    )
    period = periodo_map[periodo_label]

    st.markdown(f"""
    <hr style='border:none;border-top:1px solid {C["border"]};margin:20px 0;'>
    <div style='font-size:12px;color:{C["muted"]};line-height:1.7;'>
        <b style='color:{C["text"]};'>Variable dependiente</b><br>
        Close — Precio de cierre (COP)<br><br>
        <b style='color:{C["text"]};'>Variables regresoras</b><br>
        X₁ — USD/COP (riesgo macro)<br>
        X₂ — COLCAP (sentimiento bursátil)<br><br>
        <b style='color:{C["text"]};'>Fuente</b><br>
        Yahoo Finance (tiempo real)<br><br>
        <b style='color:{C["text"]};'>Modelo</b><br>
        Regresión lineal múltiple
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("↺ Actualizar datos", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# CARGA Y CÁLCULOS
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Descargando datos desde Yahoo Finance…"):
    try:
        df = cargar_datos(period)
    except Exception as exc:
        st.error(f"No se pudieron descargar los datos desde Yahoo Finance: {exc}")
        st.stop()

if df.empty or len(df) < 20:
    st.error("No se pudieron obtener datos suficientes. Intenta con otro período o actualiza.")
    st.stop()

mod = ajustar_modelo(df)
r_usdcop = df["Corficolombiana"].corr(df["USDCOP"])
r_colcap = df["Corficolombiana"].corr(df["COLCAP"])
precio_actual = df["Corficolombiana"].iloc[-1]
precio_inicio = df["Corficolombiana"].iloc[0]
variacion_pct = ((precio_actual / precio_inicio) - 1) * 100
variacion_abs = precio_actual - precio_inicio

# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 1 — HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='padding:8px 0 32px;border-bottom:1px solid {C["border"]};margin-bottom:36px;'>
    <div style='font-size:12px;color:{C["muted"]};letter-spacing:2px;
        text-transform:uppercase;font-weight:500;margin-bottom:10px;'>
        Análisis Bursátil · Bolsa de Valores de Colombia · {periodo_label}
    </div>
    <h1 style='font-family:"Libre Baskerville",serif;font-size:36px;
        font-weight:700;color:{C["text"]};margin:0 0 8px;line-height:1.15;'>
        ¿Qué explica el comportamiento<br>bursátil de Corficolombiana?
    </h1>
    <p style='font-size:15px;color:{C["muted"]};margin:0;max-width:720px;line-height:1.6;'>
        Relación entre el mercado colombiano, el riesgo macroeconómico y el precio de la acción.
        Corficolombiana es el <em>holding</em> de inversiones del Grupo Aval, con participaciones
        en infraestructura, energía y el sector financiero colombiano.
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 2 — KPIs
# ─────────────────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

with k1:
    kpi("Precio actual",
        f"${precio_actual:,.0f}",
        f"{variacion_abs:+,.0f} COP en el período",
        C["blue"])
with k2:
    delta_str = f"{variacion_pct:+.1f}% vs inicio del período"
    kpi("Variación del período",
        f"{variacion_pct:+.1f}%",
        delta_str,
        C["green"] if variacion_pct >= 0 else C["red"])
with k3:
    kpi("Correlación USD/COP",
        f"{r_usdcop:+.3f}",
        "Asociación con el tipo de cambio",
        C["green"] if r_usdcop > 0 else C["red"])
with k4:
    kpi("Correlación COLCAP",
        f"{r_colcap:+.3f}",
        "Asociación con el mercado colombiano",
        C["green"] if r_colcap > 0 else C["red"])

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 3 — EVOLUCIÓN BURSÁTIL
# ─────────────────────────────────────────────────────────────────────────────
section_title(
    "Evolución del precio de Corficolombiana",
    f"{df.index.min().strftime('%d %b %Y')} — {df.index.max().strftime('%d %b %Y')} · {len(df)} días hábiles"
)

ma30 = df["Corficolombiana"].rolling(30).mean()
idx_max = df["Corficolombiana"].idxmax()
idx_min = df["Corficolombiana"].idxmin()

fig_ts = go.Figure()

# Área bajo la línea
fig_ts.add_trace(go.Scatter(
    x=df.index, y=df["Corficolombiana"],
    fill="tozeroy",
    fillcolor=f"rgba(59,130,246,0.05)",
    line=dict(color=C["blue"], width=1.8),
    name="Precio cierre",
    hovertemplate="<b>%{x|%d %b %Y}</b><br>Precio: $%{y:,.0f} COP<extra></extra>",
))

# MA30
fig_ts.add_trace(go.Scatter(
    x=df.index, y=ma30,
    line=dict(color=C["amber"], width=1.3, dash="dot"),
    name="Media móvil 30d",
    hovertemplate="MA30: $%{y:,.0f}<extra></extra>",
))

# Máximo
fig_ts.add_trace(go.Scatter(
    x=[idx_max], y=[df["Corficolombiana"].max()],
    mode="markers+text",
    marker=dict(color=C["green"], size=8, symbol="triangle-up"),
    text=[f"  Máx ${df['Corficolombiana'].max():,.0f}"],
    textposition="top right",
    textfont=dict(color=C["green"], size=10),
    name="Máximo",
    showlegend=False,
    hovertemplate="Máximo: $%{y:,.0f}<extra></extra>",
))

# Mínimo
fig_ts.add_trace(go.Scatter(
    x=[idx_min], y=[df["Corficolombiana"].min()],
    mode="markers+text",
    marker=dict(color=C["red"], size=8, symbol="triangle-down"),
    text=[f"  Mín ${df['Corficolombiana'].min():,.0f}"],
    textposition="bottom right",
    textfont=dict(color=C["red"], size=10),
    name="Mínimo",
    showlegend=False,
    hovertemplate="Mínimo: $%{y:,.0f}<extra></extra>",
))

base = plotly_base(400)
base["yaxis"]["tickprefix"] = "$"
base["yaxis"]["tickformat"] = ",.0f"
fig_ts.update_layout(**base,
    title=dict(text="Precio de cierre (COP)", font=dict(size=13, color=C["muted"])),
    hovermode="x unified",
)
st.plotly_chart(fig_ts, use_container_width=True)

# Interpretación dinámica
rango = df["Corficolombiana"].max() - df["Corficolombiana"].min()
cv = df["Corficolombiana"].std() / df["Corficolombiana"].mean() * 100

insight_text(
    f"Durante los últimos {periodo_label}, Corficolombiana registró un precio promedio de "
    f"<b>${df['Corficolombiana'].mean():,.0f} COP</b> con una variación acumulada de "
    f"<b>{variacion_pct:+.1f}%</b>. El rango de precios fue de <b>${rango:,.0f} COP</b> "
    f"(CV: {cv:.1f}%), lo que refleja la volatilidad propia de un activo de renta variable "
    f"en un mercado emergente como la BVC. La media móvil de 30 días suaviza el ruido "
    f"de corto plazo y permite identificar la tendencia estructural del período.",
    icon="📈"
)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 4 — CORFICOLOMBIANA vs USD/COP
# ─────────────────────────────────────────────────────────────────────────────
section_title(
    "Corficolombiana y el tipo de cambio USD/COP",
    "El dólar como termómetro del riesgo-país"
)

# Normalizar para comparación (base 100)
base_norm = df.iloc[0]
df_n = (df / base_norm * 100)

fig_usd = make_subplots(specs=[[{"secondary_y": True}]])

fig_usd.add_trace(go.Scatter(
    x=df.index, y=df["Corficolombiana"],
    name="Corficolombiana",
    line=dict(color=C["blue"], width=2),
    hovertemplate="Corficol.: $%{y:,.0f}<extra></extra>",
), secondary_y=False)

fig_usd.add_trace(go.Scatter(
    x=df.index, y=df["USDCOP"],
    name="USD/COP",
    line=dict(color=C["amber"], width=1.5, dash="dash"),
    opacity=0.85,
    hovertemplate="USD/COP: %{y:,.0f}<extra></extra>",
), secondary_y=True)

fig_usd.update_layout(
    **plotly_base(380),
    title=dict(text="Comparación temporal sincronizada (doble eje)", font=dict(size=13, color=C["muted"])),
    hovermode="x unified",
)
fig_usd.update_yaxes(
    tickprefix="$", tickformat=",.0f",
    gridcolor=C["border"], zeroline=False,
    title_text="Corficolombiana (COP)", title_font=dict(color=C["blue"]),
    tickfont=dict(color=C["blue"]),
    secondary_y=False
)
fig_usd.update_yaxes(
    tickformat=",.0f",
    gridcolor="rgba(0,0,0,0)", zeroline=False,
    title_text="USD/COP", title_font=dict(color=C["amber"]),
    tickfont=dict(color=C["amber"]),
    secondary_y=True
)
st.plotly_chart(fig_usd, use_container_width=True)

# Interpretación dinámica USD/COP
dir_usd = "negativa" if r_usdcop < 0 else "positiva"
if r_usdcop < -0.3:
    interp_usd = (
        f"La correlación <b>{dir_usd} de {r_usdcop:.3f}</b> sugiere que cuando el dólar sube "
        f"(depreciación del peso colombiano), el precio de Corficolombiana tiende a bajar. "
        f"Esto es coherente con la dinámica de mercados emergentes: un dólar caro refleja "
        f"salida de capitales extranjeros y mayor percepción de riesgo sobre Colombia. "
        f"En ese contexto, los inversionistas tienden a reducir su exposición a activos "
        f"de renta variable local como Corficolombiana."
    )
elif r_usdcop > 0.3:
    interp_usd = (
        f"La correlación <b>{dir_usd} de {r_usdcop:.3f}</b> indica que en este período ambas "
        f"variables se movieron en la misma dirección. Una posible explicación: en ciclos "
        f"expansivos, tanto el dólar como las acciones colombianas pueden valorizarse "
        f"simultáneamente, o ambas variables pueden estar respondiendo a un tercer factor "
        f"como mejoras en los términos de intercambio de Colombia."
    )
else:
    interp_usd = (
        f"La correlación <b>{dir_usd} moderada de {r_usdcop:.3f}</b> entre el USD/COP y Corficolombiana "
        f"sugiere que en este período el tipo de cambio no fue el factor dominante en el "
        f"comportamiento de la acción. Otros elementos como decisiones corporativas del "
        f"Grupo Aval o dinámicas sectoriales pueden haber tenido mayor peso."
    )

insight_text(interp_usd, icon="💱")

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 5 — CORFICOLOMBIANA vs COLCAP
# ─────────────────────────────────────────────────────────────────────────────
section_title(
    "Corficolombiana y el índice COLCAP",
    "La acción en el contexto del mercado accionario colombiano"
)

fig_col = go.Figure()

fig_col.add_trace(go.Scatter(
    x=df_n.index, y=df_n["Corficolombiana"],
    name="Corficolombiana",
    line=dict(color=C["blue"], width=2),
    fill="tozeroy", fillcolor=f"rgba(59,130,246,0.04)",
    hovertemplate="Corficol.: %{y:.1f}<extra></extra>",
))

fig_col.add_trace(go.Scatter(
    x=df_n.index, y=df_n["COLCAP"],
    name="COLCAP",
    line=dict(color=C["green"], width=1.6, dash="dash"),
    opacity=0.85,
    hovertemplate="COLCAP: %{y:.1f}<extra></extra>",
))

fig_col.add_hline(y=100, line_dash="dot", line_color=C["border"], line_width=1)

base2 = plotly_base(380)
base2["yaxis"]["title"] = "Índice base 100 (inicio del período)"
fig_col.update_layout(
    **base2,
    title=dict(text="Desempeño relativo normalizado (Base 100 = inicio del período)", font=dict(size=13, color=C["muted"])),
    hovermode="x unified",
)
st.plotly_chart(fig_col, use_container_width=True)

# Interpretación dinámica COLCAP
rf = df_n["Corficolombiana"].iloc[-1] - 100
rm = df_n["COLCAP"].iloc[-1] - 100
if rf > rm + 5:
    perf_label = "Corficolombiana <b>superó al COLCAP</b> en este período"
    perf_exp = "lo que puede reflejar factores corporativos positivos específicos de la compañía o del Grupo Aval"
elif rf < rm - 5:
    perf_label = "Corficolombiana <b>tuvo un desempeño inferior al COLCAP</b>"
    perf_exp = "lo que puede indicar presiones específicas sobre la empresa o el sector financiero"
else:
    perf_label = "Corficolombiana <b>se movió en línea con el COLCAP</b>"
    perf_exp = "confirmando que la acción sigue de cerca el sentimiento general del mercado colombiano"

insight_text(
    f"La correlación entre Corficolombiana y el COLCAP es <b>{r_colcap:+.3f}</b>. "
    f"En el período analizado, {perf_label} ({rf:+.1f}% vs {rm:+.1f}%), {perf_exp}. "
    f"La gráfica de base 100 permite comparar la dirección del movimiento independientemente "
    f"de la escala de precios. Cuando el COLCAP cae, arrastra a la mayoría de acciones "
    f"colombianas: es el riesgo sistémico del mercado local, que ningún inversor puede "
    f"eliminar permaneciendo invertido únicamente en la BVC.",
    icon="📊"
)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 6 — INTERPRETACIÓN FINANCIERA (sección principal)
# ─────────────────────────────────────────────────────────────────────────────
section_title(
    "Interpretación financiera",
    "¿Qué nos dicen los datos sobre el comportamiento de Corficolombiana?"
)

col_a, col_b = st.columns(2)

with col_a:
    card(f"""
    <div style='font-size:12px;color:{C["muted"]};text-transform:uppercase;
        letter-spacing:1.5px;font-weight:500;margin-bottom:10px;'>
        Riesgo macroeconómico
    </div>
    <div style='font-size:22px;font-family:"Libre Baskerville",serif;
        color:{C["blue"]};margin-bottom:10px;'>USD/COP · r = {r_usdcop:+.3f}</div>
    <div style='font-size:13.5px;color:#A1A1AA;line-height:1.7;'>
        El tipo de cambio USD/COP actúa como <strong style='color:{C["text"]};'>
        termómetro de riesgo-país</strong>. Cuando el peso colombiano se deprecia,
        los inversionistas internacionales perciben mayor riesgo en activos colombianos.
        Para un <em>holding</em> como Corficolombiana —cuyas operaciones son
        completamente locales— esta presión puede traducirse en menor demanda por
        sus acciones. <br><br>
        El signo de la correlación refleja si la acción tiende a moverse
        en <strong>sentido contrario</strong> (correlación negativa, patrón clásico
        de mercados emergentes) o en <strong>el mismo sentido</strong> que el dólar.
    </div>
    """, border_left=C["blue"])

with col_b:
    card(f"""
    <div style='font-size:12px;color:{C["muted"]};text-transform:uppercase;
        letter-spacing:1.5px;font-weight:500;margin-bottom:10px;'>
        Sentimiento bursátil
    </div>
    <div style='font-size:22px;font-family:"Libre Baskerville",serif;
        color:{C["green"]};margin-bottom:10px;'>COLCAP · r = {r_colcap:+.3f}</div>
    <div style='font-size:13.5px;color:#A1A1AA;line-height:1.7;'>
        El COLCAP concentra las acciones más líquidas de la BVC y refleja el
        <strong style='color:{C["text"]};'>apetito general por riesgo</strong>
        en el mercado colombiano. Cuando los inversionistas confían en Colombia
        (COLCAP sube), Corficolombiana suele beneficiarse; cuando hay incertidumbre,
        el capital migra hacia activos más seguros. <br><br>
        Una correlación alta con el COLCAP indica que el comportamiento de
        Corficolombiana está más determinado por el <strong>ciclo bursátil
        sistémico</strong> que por factores exclusivamente corporativos.
    </div>
    """, border_left=C["green"])

st.markdown("<br>", unsafe_allow_html=True)

# Tabla síntesis dinámica
variacion_usd_p = ((df["USDCOP"].iloc[-1] / df["USDCOP"].iloc[0]) - 1) * 100
variacion_col_p = ((df["COLCAP"].iloc[-1] / df["COLCAP"].iloc[0]) - 1) * 100

st.markdown(f"**Síntesis del período · {periodo_label}**")
sin1, sin2, sin3 = st.columns(3)
with sin1:
    kpi("Corficolombiana", f"${precio_actual:,.0f}", f"{variacion_pct:+.1f}%", C["blue"])
with sin2:
    kpi("USD/COP", f"${df['USDCOP'].iloc[-1]:,.0f}", f"{variacion_usd_p:+.1f}%", C["amber"])
with sin3:
    kpi("COLCAP", f"{df['COLCAP'].iloc[-1]:,.0f}", f"{variacion_col_p:+.1f}%", C["green"])

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 7 — VALIDACIÓN ESTADÍSTICA (expander secundario)
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("📐 Validación estadística del modelo"):
    st.markdown(f"""
    <div style='font-size:13px;color:{C["muted"]};padding:4px 0 20px;line-height:1.6;'>
        Esta sección complementa el análisis financiero con la validación estadística
        del modelo de regresión lineal múltiple.
        <br>El modelo busca estimar: <code>Close = β₀ + β₁·USD/COP + β₂·COLCAP + ε</code>
    </div>
    """, unsafe_allow_html=True)

    # Métricas del modelo en columnas
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        kpi("R² (Entrenamiento)", f"{mod['r2_tr']:.4f}",
            r2_label(mod["r2_tr"]), C["blue"])
    with m2:
        kpi("R² (Prueba)", f"{mod['r2_te']:.4f}",
            r2_label(mod["r2_te"]), C["green"])
    with m3:
        kpi("RMSE (Prueba)", f"${mod['rmse']:,.0f}",
            "Error típico en COP", C["amber"])
    with m4:
        kpi("MAE (Prueba)", f"${mod['mae']:,.0f}",
            "Error absoluto medio", C["purple"])

    st.markdown("<br>", unsafe_allow_html=True)

    # Fórmula
    st.markdown(f"""
    <div style='background:{C["bg"]};border:1px solid {C["border"]};
        border-radius:8px;padding:16px 20px;font-family:"Courier New",monospace;
        font-size:14px;color:{C["green"]};text-align:center;margin:8px 0 20px;'>
        Close = {mod['b0']:,.2f}
        &nbsp;+&nbsp; {mod['b1']:.6f} × USD/COP
        &nbsp;+&nbsp; {mod['b2']:.6f} × COLCAP
    </div>
    """, unsafe_allow_html=True)

    # Scatter + Heatmap en columnas
    col_s1, col_s2 = st.columns(2)

    with col_s1:
        # Scatter Open vs Close (USD/COP)
        fig_sc1 = go.Figure()
        fig_sc1.add_trace(go.Scatter(
            x=df["USDCOP"], y=df["Corficolombiana"],
            mode="markers",
            marker=dict(color=C["amber"], size=5, opacity=0.5, line=dict(width=0)),
            name="Observaciones",
            hovertemplate="USD/COP: %{x:,.0f}<br>Corficol.: $%{y:,.0f}<extra></extra>",
        ))
        # Línea de tendencia
        from scipy import stats as sp_stats
        m_s, b_s, _, _, _ = sp_stats.linregress(df["USDCOP"], df["Corficolombiana"])
        x_l = np.linspace(df["USDCOP"].min(), df["USDCOP"].max(), 100)
        fig_sc1.add_trace(go.Scatter(
            x=x_l, y=m_s * x_l + b_s,
            mode="lines", line=dict(color="white", width=1.5),
            name=f"Tendencia (r={r_usdcop:.3f})", showlegend=True,
        ))
        b_sc1 = plotly_base(300)
        b_sc1["xaxis"]["title"] = "USD/COP"
        b_sc1["yaxis"]["title"] = "Corficolombiana (COP)"
        b_sc1["yaxis"]["tickprefix"] = "$"
        b_sc1["yaxis"]["tickformat"] = ",.0f"
        fig_sc1.update_layout(**b_sc1,
            title=dict(text=f"Scatter: USD/COP vs Close | r = {r_usdcop:.3f}",
                font=dict(size=12, color=C["muted"])))
        st.plotly_chart(fig_sc1, use_container_width=True)

    with col_s2:
        # Scatter COLCAP
        fig_sc2 = go.Figure()
        fig_sc2.add_trace(go.Scatter(
            x=df["COLCAP"], y=df["Corficolombiana"],
            mode="markers",
            marker=dict(color=C["green"], size=5, opacity=0.5, line=dict(width=0)),
            name="Observaciones",
            hovertemplate="COLCAP: %{x:,.0f}<br>Corficol.: $%{y:,.0f}<extra></extra>",
        ))
        m_s2, b_s2, _, _, _ = sp_stats.linregress(df["COLCAP"], df["Corficolombiana"])
        x_l2 = np.linspace(df["COLCAP"].min(), df["COLCAP"].max(), 100)
        fig_sc2.add_trace(go.Scatter(
            x=x_l2, y=m_s2 * x_l2 + b_s2,
            mode="lines", line=dict(color="white", width=1.5),
            name=f"Tendencia (r={r_colcap:.3f})",
        ))
        b_sc2 = plotly_base(300)
        b_sc2["xaxis"]["title"] = "COLCAP (puntos)"
        b_sc2["yaxis"]["title"] = "Corficolombiana (COP)"
        b_sc2["yaxis"]["tickprefix"] = "$"
        b_sc2["yaxis"]["tickformat"] = ",.0f"
        fig_sc2.update_layout(**b_sc2,
            title=dict(text=f"Scatter: COLCAP vs Close | r = {r_colcap:.3f}",
                font=dict(size=12, color=C["muted"])))
        st.plotly_chart(fig_sc2, use_container_width=True)

    # Heatmap de correlaciones
    corr = df.corr().round(3)
    etiquetas = {"Corficolombiana": "Corficol.", "USDCOP": "USD/COP", "COLCAP": "COLCAP"}
    corr.rename(columns=etiquetas, index=etiquetas, inplace=True)

    fig_hm = go.Figure(go.Heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        colorscale="RdYlGn",
        zmin=-1, zmax=1,
        text=corr.values.round(3),
        texttemplate="%{text}",
        textfont=dict(size=14, color="#0f0f0f", family="Inter"),
        showscale=True,
        hoverongaps=False,
    ))
    b_hm = plotly_base(260)
    b_hm["xaxis"]["side"] = "bottom"
    b_hm["xaxis"]["gridcolor"] = "rgba(0,0,0,0)"
    b_hm["yaxis"]["gridcolor"] = "rgba(0,0,0,0)"
    b_hm["yaxis"]["autorange"] = "reversed"
    fig_hm.update_layout(**b_hm,
        title=dict(text="Matriz de correlaciones de Pearson",
            font=dict(size=12, color=C["muted"])))
    st.plotly_chart(fig_hm, use_container_width=True)

    # Real vs predicho
    fechas_test = df.index[-len(mod["y_te"]):]
    fig_rp = go.Figure()
    fig_rp.add_trace(go.Scatter(
        x=fechas_test, y=mod["y_te"].values,
        name="Close real", line=dict(color=C["blue"], width=2),
        hovertemplate="Real: $%{y:,.0f}<extra></extra>",
    ))
    fig_rp.add_trace(go.Scatter(
        x=fechas_test, y=mod["y_pred"],
        name="Estimado por el modelo", line=dict(color=C["amber"], width=1.5, dash="dot"),
        hovertemplate="Estimado: $%{y:,.0f}<extra></extra>",
    ))
    b_rp = plotly_base(300)
    b_rp["yaxis"]["tickprefix"] = "$"
    b_rp["yaxis"]["tickformat"] = ",.0f"
    fig_rp.update_layout(**b_rp,
        title=dict(text=f"Precio real vs estimado — datos de prueba | R² = {mod['r2_te']:.4f}",
            font=dict(size=12, color=C["muted"])),
        hovermode="x unified")
    st.plotly_chart(fig_rp, use_container_width=True)

    st.markdown(f"""
    <div style='font-size:13px;color:{C["muted"]};line-height:1.7;padding:8px 0;'>
        <b style='color:{C["text"]};'>Interpretación técnica:</b><br>
        Con un R² de <b style='color:{C["blue"]};'>{mod['r2_te']:.4f}</b> en datos de prueba,
        el modelo captura el <b>{r2_explained_pct(mod['r2_te']):.1f}%</b> de la variabilidad del precio.
        El error típico (RMSE) es de <b>${mod['rmse']:,.0f} COP</b>, lo que indica cuánto
        se equivoca el modelo en promedio. El {r2_unexplained_pct(mod['r2_te']):.1f}% no explicado
        corresponde a factores corporativos, sectoriales y de mercado no incluidos en este modelo.
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 8 — CONCLUSIÓN FINAL
# ─────────────────────────────────────────────────────────────────────────────
section_title("Conclusión del análisis")

con1, con2 = st.columns(2)
with con1:
    st.markdown("**Hallazgos principales**")
    st.markdown(
        f"""
        - Corficolombiana refleja dinámicas del mercado colombiano y reacciona al contexto macroeconómico del país.
        - El USD/COP captura la percepción de riesgo-país y puede señalar cambios en el apetito por activos colombianos.
        - El COLCAP captura el sentimiento bursátil colectivo y el riesgo sistémico local.
        - En prueba, el modelo explica {r2_explained_pct(mod['r2_te']):.1f}% de la variabilidad del precio en el período analizado.
        - El mercado es multifactorial: resultados corporativos, política monetaria, liquidez y psicología del mercado también importan.
        """
    )
with con2:
    st.markdown("**Perspectiva de inversión**")
    st.markdown(
        """
        Invertir en Corficolombiana implica asumir exposición al riesgo sistémico del mercado colombiano.
        El inversionista que compra esta acción está, en parte, apostando por la estabilidad macroeconómica
        de Colombia y por el dinamismo del mercado bursátil local.

        Este análisis no predice el precio futuro. Su valor está en mostrar qué factores han acompañado
        históricamente su comportamiento, como punto de partida para un análisis fundamental más profundo.
        """
    )

# Footer
st.markdown(f"""
<div style='text-align:center;color:{C["muted"]};font-size:11px;
    padding:32px 0 16px;border-top:1px solid {C["border"]};margin-top:40px;'>
    Corficolombiana (CORFICOLCF.CL) · Bolsa de Valores de Colombia ·
    Datos: Yahoo Finance · Modelo: Regresión Lineal Múltiple · Streamlit
</div>
""", unsafe_allow_html=True)
