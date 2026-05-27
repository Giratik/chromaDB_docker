import streamlit as st
import requests
import json
import time
import threading
import queue
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ollama KV Cache Benchmark",
    page_icon="⚡",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
code, .stCode, pre { font-family: 'JetBrains Mono', monospace !important; }

[data-testid="stAppViewContainer"] { color: #e8e8f0; }
[data-testid="stSidebar"] { border-right: 1px solid #1e1e30; }
[data-testid="stSidebar"] * { color: #e8e8f0 !important; }

.metric-card {
    border: 1px solid #1e1e30; border-radius: 12px;
    padding: 20px 24px; margin: 8px 0; transition: border-color 0.2s;
}
.metric-card:hover { border-color: #7c3aed; }
.metric-label { font-size: 11px; letter-spacing: 2px; color: #666; text-transform: uppercase; margin-bottom: 6px; }
.metric-value { font-size: 28px; font-weight: 800; color: #a78bfa; font-family: 'JetBrains Mono', monospace; }
.metric-sub { font-size: 12px; color: #555; margin-top: 4px; }

.result-row {
    background: #0f0f1a; border: 1px solid #1e1e30; border-radius: 8px;
    padding: 12px 16px; margin: 6px 0;
    font-family: 'JetBrains Mono', monospace; font-size: 13px;
}
.ok  { color: #34d399; }
.warn { color: #fbbf24; }
.hi  { color: #a78bfa; font-weight: 700; }

h1 { font-family: 'Syne', sans-serif !important; font-weight: 800 !important; color: #e8e8f0 !important; }
h2, h3 { font-family: 'Syne', sans-serif !important; color: #a78bfa !important; }

.stButton > button {
    background: #7c3aed !important; color: white !important; border: none !important;
    border-radius: 8px !important; font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important; padding: 12px 28px !important; width: 100% !important;
}
.stButton > button:hover { background: #6d28d9 !important; }
.stop-btn > button { background: #dc2626 !important; }
.stop-btn > button:hover { background: #b91c1c !important; }
</style>
""", unsafe_allow_html=True)

OLLAMA_BASE = "http://localhost:11434"

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_local_models():
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []

def ollama_ps():
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/ps", timeout=5)
        return r.json().get("models", [])
    except Exception:
        return []

def bytes_to_gb(b):
    return round(b / 1024**3, 3)

def get_vram(model_name):
    """Return VRAM used by a model from ollama ps."""
    ps = ollama_ps()
    for m in ps:
        if m.get("name", "").startswith(model_name.split(":")[0]):
            return bytes_to_gb(m.get("size_vram", 0)), bytes_to_gb(m.get("size", 0))
    return 0.0, 0.0

def stream_worker(model, prompt, num_ctx, num_predict, stop_event, result_queue, idx):
    """
    Send a streaming request and consume tokens until stop_event is set.
    This keeps the KV cache alive for the duration of the benchmark.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "temperature": 1.0,
        },
    }
    tokens = 0
    error = None
    try:
        with requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json=payload,
            stream=True,
            timeout=300,
        ) as resp:
            for line in resp.iter_lines():
                if stop_event.is_set():
                    break
                if line:
                    try:
                        chunk = json.loads(line)
                        if chunk.get("response"):
                            tokens += 1
                        if chunk.get("done"):
                            break
                    except Exception:
                        pass
    except Exception as e:
        error = str(e)
    result_queue.put({"idx": idx, "tokens": tokens, "error": error})


# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("measurements", []),
    ("log", []),
    ("running", False),
    ("stop_event", None),
    ("snapshots", []),
    ("threads", []),
    ("result_queue", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Layout ────────────────────────────────────────────────────────────────────
st.markdown("# ⚡ Ollama KV Cache Benchmark")
st.markdown(
    '<p style="color:#555;margin-top:-12px;margin-bottom:24px;">'
    "Maintient N requêtes en streaming actives · mesure la VRAM pendant la génération</p>",
    unsafe_allow_html=True,
)

col_side, col_main = st.columns([1, 2.5])

with col_side:
    st.markdown("### ⚙️ Configuration")

    models = get_local_models()
    if not models:
        st.error("❌ Ollama non accessible sur localhost:11434")
        st.stop()

    model = st.selectbox("Modèle", models, disabled=st.session_state.running)

    num_ctx = st.select_slider(
        "num_ctx (tokens)",
        options=[2048, 4096, 8192, 12288, 16384, 24576, 32768, 40000, 65536, 131072],
        value=16384,
        disabled=st.session_state.running,
    )

    n_parallel = st.slider(
        "Instances parallèles",
        1, 100, 5,
        disabled=st.session_state.running,
        help="Doit correspondre à OLLAMA_NUM_PARALLEL côté serveur",
    )

    num_predict = st.slider(
        "Tokens à générer / instance",
        50, 2000, 500,
        disabled=st.session_state.running,
        help="Plus c'est long, plus la fenêtre de mesure est large",
    )

    sample_interval = st.slider(
        "Intervalle snapshot (s)",
        1, 10, 2,
        disabled=st.session_state.running,
    )

    prompt = st.text_area(
        "Prompt de test",
        value=(
            "Écris une très longue histoire détaillée sur l'architecture des transformers, "
            "le mécanisme d'attention, le KV cache, et leurs implications pour le déploiement "
            "de modèles de langage à grande échelle. Continue indéfiniment avec des détails techniques."
        ),
        height=120,
        disabled=st.session_state.running,
    )

    st.markdown("---")

    if not st.session_state.running:
        run_btn = st.button("▶ Lancer le benchmark", use_container_width=True)
    else:
        st.markdown('<div class="stop-btn">', unsafe_allow_html=True)
        stop_btn = st.button("⏹ Arrêter", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    clear_btn = st.button("🗑 Reset historique", use_container_width=True, disabled=st.session_state.running)
    if clear_btn:
        st.session_state.measurements = []
        st.session_state.log = []
        st.session_state.snapshots = []
        st.rerun()

    st.markdown("---")
    st.markdown("### 📡 ollama ps (live)")
    ps_live = ollama_ps()
    if ps_live:
        for m in ps_live:
            vram = bytes_to_gb(m.get("size_vram", 0))
            total = bytes_to_gb(m.get("size", 0))
            st.markdown(
                f'<div class="result-row"><span class="hi">{m["name"]}</span><br>'
                f'VRAM: <span class="ok">{vram} GB</span> &nbsp;|&nbsp; Total: {total} GB</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown('<p style="color:#555;font-size:13px;">Aucun modèle actif</p>', unsafe_allow_html=True)

# ── Main panel ────────────────────────────────────────────────────────────────
with col_main:

    # ── START ─────────────────────────────────────────────────────────────────
    if not st.session_state.running and "run_btn" in dir() and run_btn:
        stop_event = threading.Event()
        result_queue = queue.Queue()
        st.session_state.stop_event = stop_event
        st.session_state.result_queue = result_queue
        st.session_state.snapshots = []
        st.session_state.running = True

        # Snapshot de base avant lancement
        vram_before, total_before = get_vram(model)
        st.session_state.snapshots.append({
            "t": 0,
            "label": "avant",
            "vram_gb": vram_before,
            "total_gb": total_before,
            "active_instances": 0,
        })

        # Lancer les threads streaming
        threads = []
        for i in range(n_parallel):
            t = threading.Thread(
                target=stream_worker,
                args=(model, prompt, num_ctx, num_predict, stop_event, result_queue, i),
                daemon=True,
            )
            t.start()
            threads.append(t)
        st.session_state.threads = threads
        st.session_state._bench_start = time.time()
        st.session_state._bench_model = model
        st.session_state._bench_n = n_parallel
        st.session_state._bench_ctx = num_ctx
        st.session_state._bench_interval = sample_interval
        st.session_state._last_snap_t = time.time()
        st.rerun()

    # ── STOP ──────────────────────────────────────────────────────────────────
    if st.session_state.running and "stop_btn" in dir() and stop_btn:
        st.session_state.stop_event.set()
        st.session_state.running = False

        # Attendre fin des threads (max 5s)
        for t in st.session_state.threads:
            t.join(timeout=5)

        # Collecter résultats
        results = []
        while not st.session_state.result_queue.empty():
            results.append(st.session_state.result_queue.get_nowait())

        # Snapshot final
        vram_f, total_f = get_vram(st.session_state._bench_model)
        elapsed = round(time.time() - st.session_state._bench_start, 1)

        snaps = st.session_state.snapshots
        vram_peak = max((s["vram_gb"] for s in snaps), default=0)
        vram_base = snaps[0]["vram_gb"] if snaps else 0
        delta = round(vram_peak - vram_base, 3)
        per_inst = round(delta / st.session_state._bench_n, 3) if st.session_state._bench_n > 0 and delta > 0 else "n/a"

        measurement = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "model": st.session_state._bench_model,
            "num_ctx": st.session_state._bench_ctx,
            "n_parallel": st.session_state._bench_n,
            "vram_base_gb": vram_base,
            "vram_peak_gb": vram_peak,
            "delta_vram_gb": delta,
            "per_instance_gb": per_inst,
            "elapsed_s": elapsed,
            "errors": sum(1 for r in results if r.get("error")),
            "snapshots": snaps.copy(),
        }
        st.session_state.measurements.append(measurement)
        st.session_state.log.append(
            f"[{measurement['timestamp']}] {measurement['model']} | "
            f"ctx={measurement['num_ctx']} | x{measurement['n_parallel']} → "
            f"pic VRAM {vram_peak} GB (+{delta} GB · {per_inst} GB/inst)"
        )
        st.rerun()

    # ── POLLING pendant le run ─────────────────────────────────────────────────
    if st.session_state.running:
        now = time.time()
        elapsed_run = round(now - st.session_state._bench_start, 1)
        interval = st.session_state._bench_interval

        if now - st.session_state._last_snap_t >= interval:
            vram_now, total_now = get_vram(st.session_state._bench_model)
            alive = sum(1 for t in st.session_state.threads if t.is_alive())
            st.session_state.snapshots.append({
                "t": elapsed_run,
                "label": f"t={elapsed_run}s",
                "vram_gb": vram_now,
                "total_gb": total_now,
                "active_instances": alive,
            })
            st.session_state._last_snap_t = now

            # Arrêt automatique si tous les threads sont terminés
            if alive == 0:
                st.session_state.stop_event.set()
                st.session_state.running = False

        # Live display
        snaps = st.session_state.snapshots
        vram_now = snaps[-1]["vram_gb"] if snaps else 0
        vram_base = snaps[0]["vram_gb"] if snaps else 0
        delta_live = round(vram_now - vram_base, 3)
        alive_now = sum(1 for t in st.session_state.threads if t.is_alive())

        st.markdown("### 🔴 En cours…")
        lc1, lc2, lc3, lc4 = st.columns(4)
        for col, label, value, sub in [
            (lc1, "VRAM actuelle", f"{vram_now} GB", "ollama ps"),
            (lc2, "Delta KV cache", f"+{delta_live} GB", f"base: {vram_base} GB"),
            (lc3, "Instances actives", f"{alive_now}/{st.session_state._bench_n}", "threads vivants"),
            (lc4, "Durée", f"{elapsed_run}s", "depuis le lancement"),
        ]:
            with col:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">{label}</div>'
                    f'<div class="metric-value">{value}</div>'
                    f'<div class="metric-sub">{sub}</div></div>',
                    unsafe_allow_html=True,
                )

        # Live chart des snapshots
        if len(snaps) > 1:
            df_live = pd.DataFrame(snaps)
            fig_live = go.Figure()
            fig_live.add_trace(go.Scatter(
                x=df_live["t"], y=df_live["vram_gb"],
                mode="lines+markers", name="VRAM (GB)",
                line=dict(color="#7c3aed", width=2),
                marker=dict(size=6, color="#a78bfa"),
                fill="tozeroy", fillcolor="rgba(124,58,237,0.08)",
            ))
            fig_live.update_layout(
                paper_bgcolor="#0a0a0f", plot_bgcolor="#0a0a0f",
                font=dict(color="#888", family="JetBrains Mono"),
                xaxis=dict(showgrid=False, title="Temps (s)"),
                yaxis=dict(gridcolor="#1e1e30", title="VRAM (GB)"),
                margin=dict(l=40, r=20, t=10, b=40), height=220,
                showlegend=False,
            )
            st.plotly_chart(fig_live, use_container_width=True)

        time.sleep(interval)
        st.rerun()

    # ── Historique des mesures ────────────────────────────────────────────────
    if st.session_state.measurements:
        last = st.session_state.measurements[-1]

        st.markdown("### 📊 Dernière mesure")
        m1, m2, m3, m4 = st.columns(4)
        for col, label, value, sub in [
            (m1, "VRAM pic",       f"{last['vram_peak_gb']} GB",    "max pendant le run"),
            (m2, "Delta KV total", f"+{last['delta_vram_gb']} GB",  "pic − base"),
            (m3, "Par instance",   f"{last['per_instance_gb']} GB", "KV cache / instance"),
            (m4, "Durée",          f"{last['elapsed_s']}s",         f"x{last['n_parallel']} instances"),
        ]:
            with col:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">{label}</div>'
                    f'<div class="metric-value">{value}</div>'
                    f'<div class="metric-sub">{sub}</div></div>',
                    unsafe_allow_html=True,
                )

        # Courbe des snapshots de la dernière mesure
        if last["snapshots"]:
            st.markdown("### 📈 Profil VRAM — dernière mesure")
            df_snap = pd.DataFrame(last["snapshots"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_snap["t"], y=df_snap["vram_gb"],
                mode="lines+markers", name="VRAM (GB)",
                line=dict(color="#7c3aed", width=2),
                marker=dict(size=7, color="#a78bfa"),
                fill="tozeroy", fillcolor="rgba(124,58,237,0.08)",
            ))
            fig.update_layout(
                paper_bgcolor="#0a0a0f", plot_bgcolor="#0a0a0f",
                font=dict(color="#888", family="JetBrains Mono"),
                xaxis=dict(showgrid=False, title="Temps (s)"),
                yaxis=dict(gridcolor="#1e1e30", title="VRAM (GB)"),
                margin=dict(l=40, r=20, t=10, b=40), height=260,
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Tableau comparatif
        st.markdown("### 🗂 Historique comparatif")
        df_hist = pd.DataFrame([
            {
                "Heure":         m["timestamp"],
                "Modèle":        m["model"],
                "num_ctx":       m["num_ctx"],
                "Instances":     m["n_parallel"],
                "VRAM base (GB)":m["vram_base_gb"],
                "VRAM pic (GB)": m["vram_peak_gb"],
                "ΔKV (GB)":      m["delta_vram_gb"],
                "GB/instance":   m["per_instance_gb"],
                "Durée (s)":     m["elapsed_s"],
                "Erreurs":       m["errors"],
            }
            for m in st.session_state.measurements
        ])
        st.dataframe(df_hist, use_container_width=True, hide_index=True)

        with st.expander("📋 Log"):
            for line in reversed(st.session_state.log):
                st.markdown(f'<div class="result-row">{line}</div>', unsafe_allow_html=True)

    elif not st.session_state.running:
        st.markdown("""
        <div style="text-align:center;padding:80px 40px;">
            <div style="font-size:48px;margin-bottom:16px;">⚡</div>
            <div style="font-size:18px;font-family:'Syne',sans-serif;color:#444;">Configure et lance un benchmark</div>
            <div style="font-size:13px;color:#333;margin-top:8px;">
                Les requêtes streaming restent actives le temps de la mesure<br>
                pour forcer l'allocation réelle du KV cache par instance.
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<hr style="border-color:#1e1e30;margin-top:40px;">
<p style="text-align:center;color:#333;font-size:12px;font-family:'JetBrains Mono',monospace;">
    VRAM via ollama ps · streaming maintenu pendant la mesure · delta = KV cache réel alloué
</p>
""", unsafe_allow_html=True)