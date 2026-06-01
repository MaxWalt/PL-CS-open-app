import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="PL & CS Model — Open Distance", layout="wide")
st_autorefresh(interval=600_000, key="keepalive")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_time(t):
    """Convert 'mm:ss', 'hh:mm:ss', or raw seconds (str/float) to seconds."""
    if t is None:
        return None
    t = str(t).strip()
    if not t:
        return None
    parts = t.split(":")
    try:
        if len(parts) == 1:
            return float(parts[0])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except ValueError:
        return None


def fmt_time(seconds):
    """Format seconds as h:mm:ss or m:ss."""
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def fmt_pace(seconds_per_meter):
    """Return pace as mm:ss per km."""
    spm = seconds_per_meter * 1000
    m = int(spm) // 60
    s = spm % 60
    return f"{m}:{s:05.2f} /km"


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def fit_pl(distances, times):
    """OLS log-log fit: log(speed) = log(S) - b*log(t)  =>  speed = S * t^(-b)."""
    speeds = distances / times
    log_t = np.log(times)
    log_s = np.log(speeds)
    b_coef = np.polyfit(log_t, log_s, 1)
    b = -b_coef[0]
    S = np.exp(b_coef[1])
    E = 1 - b
    return S, b, E


def fit_cs(distances, times):
    """OLS linear fit: distance = CS * time + D_prime."""
    A = np.column_stack([times, np.ones_like(times)])
    result = np.linalg.lstsq(A, distances, rcond=None)
    CS, D_prime = result[0]
    return CS, D_prime


# ---------------------------------------------------------------------------
# Training zones (Hunter et al., 2024)
# ---------------------------------------------------------------------------

ZONE_COLORS = {
    "Z1": "#2980B9",
    "Z2": "#27AE60",
    "Z3": "#D4AC0D",
    "Z4": "#E67E22",
    "Z5": "#C0392B",
}

def get_z2_upper(CS_ms):
    """Upper boundary of Z2 as fraction of CS (speed-dependent)."""
    if CS_ms < 3.5:
        return 0.806
    elif CS_ms < 4.5:
        return 0.832
    else:
        return 0.842


def zone_boundaries(CS_ms):
    z2u = get_z2_upper(CS_ms)
    return {
        "Z1": (0, 0.70),
        "Z2": (0.70, z2u),
        "Z3": (z2u, 0.92),
        "Z4": (0.92, 1.05),
        "Z5": (1.05, 1.40),
    }


def classify_pace(speed_ms, CS_ms):
    bounds = zone_boundaries(CS_ms)
    frac = speed_ms / CS_ms
    for zone, (lo, hi) in bounds.items():
        if lo <= frac < hi:
            return zone
    if frac >= 1.40:
        return "Z5"
    return "Z1"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("Power Law & Critical Speed Model — Open Distance")
st.markdown(
    "Enter any distance–time pairs to fit the PL and CS models. "
    "No restrictions on which events you use."
)

# ---------------------------------------------------------------------------
# Data input
# ---------------------------------------------------------------------------

st.subheader("Enter your performances")
st.markdown(
    "Add rows below: **Distance (m)** and **Time** (format: `m:ss`, `h:mm:ss`, or raw seconds). "
    "You need at least **2 data points** to fit the models."
)

default_data = pd.DataFrame(
    {"Distance (m)": [1500, 3000, 5000], "Time": ["3:55", "8:10", "14:30"]}
)

edited = st.data_editor(
    default_data,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Distance (m)": st.column_config.NumberColumn(
            "Distance (m)", min_value=1, step=1, format="%d m"
        ),
        "Time": st.column_config.TextColumn("Time (m:ss or h:mm:ss)"),
    },
    key="data_editor",
)

# Parse inputs
rows = []
for _, row in edited.iterrows():
    try:
        d = float(row["Distance (m)"])
    except (ValueError, TypeError):
        continue
    t = parse_time(row["Time"])
    if d and t and d > 0 and t > 0:
        rows.append((d, t))

if len(rows) < 2:
    st.warning("Please enter at least 2 valid distance–time pairs to fit the models.")
    st.stop()

distances = np.array([r[0] for r in rows])
times = np.array([r[1] for r in rows])
speeds = distances / times

# Fit models
S, b, E = fit_pl(distances, times)
CS, D_prime = fit_cs(distances, times)

# CS sanity: check if any input durations fall in 2–20 min
cs_range_mask = (times >= 120) & (times <= 1200)
cs_valid = cs_range_mask.sum() >= 2

# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------

if D_prime < 0:
    st.warning(
        f"⚠️ D′ is negative ({D_prime:.0f} m). The CS model may not be appropriate "
        "for these data (e.g., all efforts are very short or very long). "
        "Interpret CS parameters with caution."
    )
if not (0.02 <= b <= 0.35):
    st.warning(
        f"⚠️ PL exponent b = {b:.3f} is outside the typical range (0.02–0.35). "
        "Check your inputs for errors."
    )
if len(rows) == 2:
    st.info(
        "ℹ️ With exactly 2 data points both models are perfectly fitted (no residual degrees of freedom). "
        "Add more data points for a meaningful goodness-of-fit assessment."
    )
if not cs_valid:
    st.info(
        "ℹ️ CS model is most reliable when at least 2 efforts fall in the 2–20 min duration range. "
        "Training Zones require this condition."
    )

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Parameters",
    "⏱️ Predictions",
    "📈 Speed–Duration Profile",
    "🏃 Training Zones",
    "🔬 Want a Deeper Dive?",
])

# ── Tab 1: Parameters ────────────────────────────────────────────────────────

with tab1:
    st.header("Model Parameters")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Power Law (PL)")
        st.markdown("**Model:** speed = S · t⁻ᵇ")
        pl_df = pd.DataFrame({
            "Parameter": ["S (speed at t=1 s, m/s)", "b (fatigue exponent)", "E = 1 − b (endurance index)"],
            "Value": [f"{S:.4f}", f"{b:.4f}", f"{E:.4f}"],
        })
        st.dataframe(pl_df, hide_index=True, use_container_width=True)

        st.markdown(
            f"**Interpretation:** A higher **b** (closer to 0.35) indicates greater speed "
            f"loss over longer durations. Your E = **{E:.3f}** "
            + ("suggests a strong endurance profile." if E > 0.85 else
               "suggests a mixed endurance–speed profile." if E > 0.75 else
               "suggests a speed-dominant profile.")
        )

    with col2:
        st.subheader("Critical Speed (CS)")
        st.markdown("**Model:** distance = CS · time + D′")
        cs_df = pd.DataFrame({
            "Parameter": [
                "CS (m/s)",
                "CS (km/h)",
                "CS (min/km pace)",
                "D′ (m)",
            ],
            "Value": [
                f"{CS:.3f}",
                f"{CS * 3.6:.2f}",
                fmt_pace(1 / CS) if CS > 0 else "—",
                f"{D_prime:.1f}",
            ],
        })
        st.dataframe(cs_df, hide_index=True, use_container_width=True)

        st.markdown(
            f"**Interpretation:** You can theoretically sustain **{CS:.3f} m/s** "
            f"({CS * 3.6:.2f} km/h) indefinitely. D′ = **{D_prime:.0f} m** is your "
            "finite anaerobic work capacity above CS."
        )

    st.divider()
    st.subheader("Input Data Summary")
    summary_rows = []
    for d, t in zip(distances, times):
        spd = d / t
        summary_rows.append({
            "Distance (m)": int(d),
            "Time": fmt_time(t),
            "Speed (m/s)": f"{spd:.3f}",
            "Pace (min/km)": fmt_pace(1 / spd),
            "PL predicted time": fmt_time(d / (S * (d / spd / 1) ** (-b))) if False else fmt_time((d / S) ** (1 / (1 - b))),
            "CS predicted time": fmt_time((d - D_prime) / CS) if CS > 0 and (d - D_prime) > 0 else "—",
        })

    # Recompute PL predicted time correctly: speed = S*t^(-b), distance = speed*t = S*t^(1-b)
    # => t = (d/S)^(1/E)
    for i, row in enumerate(summary_rows):
        d = distances[i]
        try:
            t_pl = (d / S) ** (1 / E)
            row["PL predicted time"] = fmt_time(t_pl)
        except Exception:
            row["PL predicted time"] = "—"

    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

    st.caption(
        "Walt et al. (2025). Using Multilevel Models to Compare Performance Prediction and "
        "Characterization Abilities Between Power-Law and Critical-Speed Models in Middle- and "
        "Long-Distance Running. *International Journal of Sports Physiology and Performance.*"
    )

# ── Tab 2: Predictions ───────────────────────────────────────────────────────

with tab2:
    st.header("Predictions")

    # Predictions for entered distances
    st.subheader("Predictions for your entered distances")
    pred_rows = []
    for d, t_actual in zip(distances, times):
        t_pl = (d / S) ** (1 / E) if E > 0 else None
        t_cs = (d - D_prime) / CS if CS > 0 and (d - D_prime) > 0 else None
        pred_rows.append({
            "Distance (m)": int(d),
            "Actual time": fmt_time(t_actual),
            "PL prediction": fmt_time(t_pl) if t_pl else "—",
            "PL error": f"{(t_pl - t_actual) / t_actual * 100:+.2f}%" if t_pl else "—",
            "CS prediction": fmt_time(t_cs) if t_cs else "—",
            "CS error": f"{(t_cs - t_actual) / t_actual * 100:+.2f}%" if t_cs else "—",
        })
    st.dataframe(pd.DataFrame(pred_rows), hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Custom Distance Predictor")
    col_a, col_b = st.columns(2)
    with col_a:
        custom_dist = st.number_input("Distance to predict (m)", min_value=100, max_value=100000,
                                      value=10000, step=100)
    with col_b:
        st.markdown("&nbsp;", unsafe_allow_html=True)

    t_pl_custom = (custom_dist / S) ** (1 / E) if E > 0 else None
    t_cs_custom = (custom_dist - D_prime) / CS if CS > 0 and (custom_dist - D_prime) > 0 else None

    c1, c2 = st.columns(2)
    with c1:
        st.metric("PL predicted time", fmt_time(t_pl_custom) if t_pl_custom else "—")
        if t_pl_custom:
            spd = custom_dist / t_pl_custom
            st.caption(f"{spd:.3f} m/s · {spd * 3.6:.2f} km/h · {fmt_pace(1/spd)}")
    with c2:
        st.metric("CS predicted time", fmt_time(t_cs_custom) if t_cs_custom else "—")
        if t_cs_custom:
            spd = custom_dist / t_cs_custom
            st.caption(f"{spd:.3f} m/s · {spd * 3.6:.2f} km/h · {fmt_pace(1/spd)}")

    if t_cs_custom is None and CS > 0:
        st.warning("CS model cannot predict this distance (D′ would be exceeded — distance too short relative to D′).")

# ── Tab 3: Speed–Duration Profile ────────────────────────────────────────────

with tab3:
    st.header("Speed–Duration Profile")

    t_min = max(times.min() * 0.5, 10)
    t_max = times.max() * 2.5
    t_range = np.linspace(t_min, t_max, 500)

    pl_speeds = S * t_range ** (-b)
    cs_speeds = np.where(
        (CS * t_range + D_prime) > 0,
        (CS * t_range + D_prime) / t_range,
        np.nan,
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=t_range / 60, y=pl_speeds,
        mode="lines", name="PL model",
        line=dict(color="#E74C3C", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=t_range / 60, y=cs_speeds,
        mode="lines", name="CS model",
        line=dict(color="#3498DB", width=2, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=times / 60, y=speeds,
        mode="markers", name="Actual",
        marker=dict(color="#2ECC71", size=10, symbol="circle",
                    line=dict(color="white", width=1)),
        text=[f"{int(d)} m: {fmt_time(t)}" for d, t in zip(distances, times)],
        hoverinfo="text+y",
    ))

    fig.update_layout(
        xaxis_title="Duration (min)",
        yaxis_title="Speed (m/s)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=480,
        template="plotly_white",
    )

    if cs_valid:
        fig.add_vrect(x0=2, x1=20, fillcolor="lightblue", opacity=0.08,
                      annotation_text="CS validity window", annotation_position="top left")

    st.plotly_chart(fig, use_container_width=True)
    st.caption("Shaded region (2–20 min) is the recommended CS fitting window.")

    # Secondary: log–log plot for PL
    st.subheader("Log–Log View (Power Law)")
    log_t = np.log10(t_range)
    log_pl = np.log10(pl_speeds)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=np.log10(t_range), y=log_pl,
        mode="lines", name="PL model",
        line=dict(color="#E74C3C", width=2),
    ))
    fig2.add_trace(go.Scatter(
        x=np.log10(times), y=np.log10(speeds),
        mode="markers", name="Actual",
        marker=dict(color="#2ECC71", size=10),
        text=[f"{int(d)} m: {fmt_time(t)}" for d, t in zip(distances, times)],
        hoverinfo="text+y",
    ))
    fig2.update_layout(
        xaxis_title="log₁₀ Duration (s)",
        yaxis_title="log₁₀ Speed (m/s)",
        height=380,
        template="plotly_white",
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Tab 4: Training Zones ────────────────────────────────────────────────────

with tab4:
    st.header("Training Zones")

    if not cs_valid:
        st.warning(
            "Training Zones require at least 2 input efforts with durations between 2 and 20 minutes. "
            "Please add appropriate data points."
        )
    else:
        bounds = zone_boundaries(CS)
        z2u = get_z2_upper(CS)

        zone_data = []
        for zone, (lo, hi) in bounds.items():
            lo_ms = lo * CS
            hi_ms = hi * CS
            lo_kmh = lo_ms * 3.6
            hi_kmh = hi_ms * 3.6
            lo_pace = fmt_pace(1 / lo_ms) if lo_ms > 0 else "—"
            hi_pace = fmt_pace(1 / hi_ms) if hi_ms > 0 else "—"
            zone_data.append({
                "Zone": zone,
                "% CS": f"{lo*100:.1f}–{hi*100:.1f}%",
                "Speed (m/s)": f"{lo_ms:.2f}–{hi_ms:.2f}",
                "Speed (km/h)": f"{lo_kmh:.2f}–{hi_kmh:.2f}",
                "Pace": f"{hi_pace} – {lo_pace}",
                "Description": {
                    "Z1": "Easy / Recovery",
                    "Z2": "Aerobic / Tempo",
                    "Z3": "Threshold",
                    "Z4": "VO₂max / Race pace",
                    "Z5": "Anaerobic / Speed",
                }[zone],
            })

        zone_df = pd.DataFrame(zone_data)

        # Styled table using st.markdown
        header_cols = list(zone_df.columns)
        html = "<table style='width:100%;border-collapse:collapse;font-size:0.9rem'>"
        html += "<thead><tr>" + "".join(
            f"<th style='padding:6px 10px;text-align:left;border-bottom:2px solid #ddd'>{c}</th>"
            for c in header_cols
        ) + "</tr></thead><tbody>"
        for _, row in zone_df.iterrows():
            color = ZONE_COLORS[row["Zone"]]
            html += f"<tr style='background:{color};color:white'>"
            for c in header_cols:
                html += f"<td style='padding:6px 10px'>{row[c]}</td>"
            html += "</tr>"
        html += "</tbody></table>"
        st.markdown(html, unsafe_allow_html=True)

        st.markdown(
            f"\n**CS = {CS:.3f} m/s ({CS*3.6:.2f} km/h · {fmt_pace(1/CS)})**  |  "
            f"D′ = {D_prime:.0f} m"
        )

        st.divider()
        st.subheader("Zone Bar")

        fig_z = go.Figure()
        for zone, (lo, hi) in bounds.items():
            fig_z.add_trace(go.Bar(
                x=[hi * CS * 3.6 - lo * CS * 3.6],
                y=["Zones"],
                base=[lo * CS * 3.6],
                orientation="h",
                name=zone,
                marker_color=ZONE_COLORS[zone],
                text=zone,
                textposition="inside",
                insidetextanchor="middle",
            ))
        fig_z.update_layout(
            barmode="stack",
            xaxis_title="Speed (km/h)",
            showlegend=False,
            height=140,
            margin=dict(t=10, b=40),
            template="plotly_white",
        )
        st.plotly_chart(fig_z, use_container_width=True)

        st.divider()
        st.subheader("Classify a pace")
        col_x, col_y = st.columns(2)
        with col_x:
            input_speed_kmh = st.number_input("Speed (km/h) to classify", min_value=0.1,
                                               max_value=40.0, value=round(CS * 3.6, 1), step=0.1)
        speed_ms = input_speed_kmh / 3.6
        zone_result = classify_pace(speed_ms, CS)
        color_result = ZONE_COLORS[zone_result]
        st.markdown(
            f"<div style='background:{color_result};color:white;padding:12px 18px;"
            f"border-radius:8px;font-size:1.1rem;display:inline-block'>"
            f"<b>{input_speed_kmh:.1f} km/h → {zone_result}</b> "
            f"({speed_ms / CS * 100:.1f}% of CS)</div>",
            unsafe_allow_html=True,
        )

        st.caption("Zone thresholds: Hunter et al. (2024). Z2 upper boundary depends on CS level.")

# ── Tab 5: Deeper Dive ───────────────────────────────────────────────────────

with tab5:
    st.header("Want a Deeper Dive?")
    st.markdown(
        "Interested in a personalised analysis, coaching consultation, or research collaboration? "
        "Book a meeting directly below."
    )
    calendly_html = """
    <div class="calendly-inline-widget"
         data-url="https://calendly.com/maxime-walt/meeting-data"
         style="min-width:320px;height:700px;"></div>
    <script type="text/javascript" src="https://assets.calendly.com/assets/external/widget.js" async></script>
    """
    st.components.v1.html(calendly_html, height=720, scrolling=False)
