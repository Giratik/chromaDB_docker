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
st.set_page_config(page_title="Ollama KV Cache Benchmark", page_icon="⚡", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
code, pre { font-family: 'JetBrains Mono', monospace !important; }
[data-testid="stAppViewContainer"] { background: #0a0a0f; color: #e8e8f0; }
[data-testid="stSidebar"] { background: #0f0f1a; border-right: 1px solid #1e1e30; }
[data-testid="stSidebar"] * { color: #e8e8f0 !important; }
.mc { background:#12121f; border:1px solid #1e1e30; border-radius:12px; padding:18px 20px; margin:6px 0; }
.mc:hover { border-color:#7c3aed; }
.ml { font-size:10px; letter-spacing:2px; color:#555; text-transform:uppercase; margin-bottom:4px; }
.mv { font-size:24px; font-weight:800; color:#a78bfa; font-family:'JetBrains Mono',monospace; }
.ms { font-size:11px; color:#444; margin-top:2px; }
.rr { background:#0f0f1a; border:1px solid #1e1e30; border-radius:8px; padding:10px 14px; margin:5px 0; font-family:'JetBrains Mono',monospace; font-size:12px; }
.ok { color:#34d399; } .warn { color:#fbbf24; } .hi { color:#a78bfa; font-weight:700; }
.think-badge { display:inline-block; background:#1e1430; border:1px solid #7c3aed; border-radius:20px; padding:2px 12px; font-size:11px; color:#a78bfa; font-family:'JetBrains Mono',monospace; }
.nothink-badge { display:inline-block; background:#0f1a14; border:1px solid #34d399; border-radius:20px; padding:2px 12px; font-size:11px; color:#34d399; font-family:'JetBrains Mono',monospace; }
h1 { font-family:'Syne',sans-serif !important; font-weight:800 !important; color:#e8e8f0 !important; }
h2, h3 { font-family:'Syne',sans-serif !important; color:#a78bfa !important; }
.stButton > button { background:#7c3aed !important; color:white !important; border:none !important; border-radius:8px !important; font-family:'Syne',sans-serif !important; font-weight:700 !important; padding:10px 20px !important; width:100% !important; }
.stButton > button:hover { background:#6d28d9 !important; }
.stop-btn > button { background:#dc2626 !important; }
.stop-btn > button:hover { background:#b91c1c !important; }
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
    for m in ollama_ps():
        if m.get("name", "").startswith(model_name.split(":")[0]):
            return bytes_to_gb(m.get("size_vram", 0)), bytes_to_gb(m.get("size", 0))
    return 0.0, 0.0

def build_prompt(user_prompt, think_mode):
    """Prepend /think or /no_think directive."""
    prefix = "/think\n" if think_mode else "/no_think\n"
    return prefix + user_prompt

def stream_worker(model, prompt, num_ctx, num_predict, think_mode, stop_event, result_queue, idx):
    """
    Streaming request that stays alive until stop_event.
    Tracks: time-to-first-token, tokens/s, think block length.
    """
    full_prompt = build_prompt(prompt, think_mode)
    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": True,
        "options": {
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "temperature": 0.8,
        },
    }
    t_start = time.time()
    t_first = None
    tokens = 0
    think_tokens = 0
    response_tokens = 0
    full_text = ""
    in_think = False
    error = None

    try:
        with requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json=payload, stream=True, timeout=600,
        ) as resp:
            for line in resp.iter_lines():
                if stop_event.is_set():
                    break
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    token_text = chunk.get("response", "")
                    if token_text:
                        if t_first is None:
                            t_first = time.time() - t_start
                        tokens += 1
                        full_text += token_text
                        # Track think block
                        if "<think>" in full_text:
                            in_think = True
                        if "</think>" in full_text:
                            in_think = False
                        if in_think:
                            think_tokens += 1
                        else:
                            response_tokens += 1
                    if chunk.get("done"):
                        break
                except Exception:
                    pass
    except Exception as e:
        error = str(e)

    elapsed = time.time() - t_start
    tps = round(tokens / elapsed, 1) if elapsed > 0 else 0

    result_queue.put({
        "idx": idx,
        "tokens_total": tokens,
        "think_tokens": think_tokens,
        "response_tokens": response_tokens,
        "ttft_s": round(t_first, 3) if t_first else None,
        "tps": tps,
        "elapsed_s": round(elapsed, 1),
        "error": error,
    })


# ── Session state ─────────────────────────────────────────────────────────────
DEFAULTS = {
    "measurements": [], "log": [], "running": False,
    "stop_event": None, "snapshots": [], "threads": [],
    "result_queue": None, "worker_results": [],
    "_bench_start": 0, "_bench_model": "", "_bench_n": 1,
    "_bench_ctx": 4096, "_bench_interval": 2, "_bench_think": False,
    "_last_snap_t": 0,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Layout ────────────────────────────────────────────────────────────────────
st.markdown("# ⚡ Ollama KV Cache & Think Benchmark")
st.markdown(
    '<p style="color:#555;margin-top:-12px;margin-bottom:24px;">'
    "Streaming réel · VRAM live · comparaison think ON/OFF</p>",
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

    # ── Think mode toggle ──────────────────────────────────────────────────────
    think_mode = st.toggle(
        "🧠 Think activé  (`/think`)",
        value=False,
        disabled=st.session_state.running,
        help="Injecte /think ou /no_think en tête du prompt",
    )
    if think_mode:
        st.markdown('<span class="think-badge">🧠 /think — raisonnement activé</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="nothink-badge">⚡ /no_think — réponse directe</span>', unsafe_allow_html=True)

    st.markdown("")

    num_ctx = st.select_slider(
        "num_ctx (tokens)",
        options=[2048, 4096, 8192, 12288, 16384, 24576, 32768, 40000, 65536, 131072],
        value=12288,
        disabled=st.session_state.running,
    )
    n_parallel = st.slider("Instances parallèles", 1, 20, 1, disabled=st.session_state.running,
                           help="Doit correspondre à OLLAMA_NUM_PARALLEL côté serveur")
    num_predict = st.slider("Tokens à générer / instance", 50, 2000, 600, disabled=st.session_state.running)
    sample_interval = st.slider("Intervalle snapshot VRAM (s)", 1, 10, 2, disabled=st.session_state.running)

    prompt = st.text_area(
        "Prompt de test",
        value="Résous ce problème étape par étape : une salle contient 23 personnes. Quelle est la probabilité qu'au moins deux d'entre elles partagent le même anniversaire ? Explique chaque étape du calcul en détail.",
        height=130,
        disabled=st.session_state.running,
    )

    st.markdown("---")

    if not st.session_state.running:
        run_btn = st.button("▶ Lancer", use_container_width=True)
    else:
        st.markdown('<div class="stop-btn">', unsafe_allow_html=True)
        stop_btn = st.button("⏹ Arrêter", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    clear_btn = st.button("🗑 Reset historique", use_container_width=True, disabled=st.session_state.running)
    if clear_btn:
        for k in ["measurements", "log", "snapshots", "worker_results"]:
            st.session_state[k] = []
        st.rerun()

    st.markdown("---")
    st.markdown("### 📡 ollama ps")
    for m in ollama_ps():
        vram = bytes_to_gb(m.get("size_vram", 0))
        total = bytes_to_gb(m.get("size", 0))
        st.markdown(
            f'<div class="rr"><span class="hi">{m["name"]}</span><br>'
            f'VRAM: <span class="ok">{vram} GB</span> | Total: {total} GB</div>',
            unsafe_allow_html=True,
        )

# ── Main panel ────────────────────────────────────────────────────────────────
with col_main:

    # ── START ─────────────────────────────────────────────────────────────────
    if not st.session_state.running and "run_btn" in dir() and run_btn:
        stop_event = threading.Event()
        rq = queue.Queue()
        st.session_state.update({
            "stop_event": stop_event,
            "result_queue": rq,
            "snapshots": [],
            "worker_results": [],
            "running": True,
            "_bench_start": time.time(),
            "_bench_model": model,
            "_bench_n": n_parallel,
            "_bench_ctx": num_ctx,
            "_bench_interval": sample_interval,
            "_bench_think": think_mode,
            "_last_snap_t": time.time(),
        })
        vram_b, total_b = get_vram(model)
        st.session_state.snapshots.append({"t": 0, "vram_gb": vram_b, "active": 0})

        threads = []
        for i in range(n_parallel):
            t = threading.Thread(
                target=stream_worker,
                args=(model, prompt, num_ctx, num_predict, think_mode,
                      stop_event, rq, i),
                daemon=True,
            )
            t.start()
            threads.append(t)
        st.session_state.threads = threads
        st.rerun()

    # ── STOP ──────────────────────────────────────────────────────────────────
    if st.session_state.running and "stop_btn" in dir() and stop_btn:
        st.session_state.stop_event.set()
        st.session_state.running = False
        for t in st.session_state.threads:
            t.join(timeout=5)
        _finalize = True
    else:
        _finalize = False

    # Auto-stop quand tous les threads sont morts
    if st.session_state.running:
        alive = sum(1 for t in st.session_state.threads if t.is_alive())
        if alive == 0:
            st.session_state.stop_event.set()
            st.session_state.running = False
            _finalize = True

    if _finalize and st.session_state.snapshots:
        rq = st.session_state.result_queue
        results = []
        while not rq.empty():
            results.append(rq.get_nowait())
        st.session_state.worker_results = results

        snaps = st.session_state.snapshots
        vram_peak = max(s["vram_gb"] for s in snaps)
        vram_base = snaps[0]["vram_gb"]
        delta = round(vram_peak - vram_base, 3)
        n = st.session_state._bench_n
        per_inst = round(delta / n, 3) if n > 0 and delta > 0 else None
        elapsed = round(time.time() - st.session_state._bench_start, 1)

        # Agrégation des stats workers
        valid = [r for r in results if not r.get("error")]
        avg_tps    = round(sum(r["tps"] for r in valid) / len(valid), 1) if valid else None
        avg_ttft   = round(sum(r["ttft_s"] for r in valid if r["ttft_s"]) / len(valid), 3) if valid else None
        avg_think  = round(sum(r["think_tokens"] for r in valid) / len(valid)) if valid else 0
        avg_resp   = round(sum(r["response_tokens"] for r in valid) / len(valid)) if valid else 0

        think_label = "think=ON (/think)" if st.session_state._bench_think else "think=OFF (/no_think)"
        measurement = {
            "timestamp":       datetime.now().strftime("%H:%M:%S"),
            "model":           st.session_state._bench_model,
            "think":           st.session_state._bench_think,
            "think_label":     think_label,
            "num_ctx":         st.session_state._bench_ctx,
            "n_parallel":      n,
            "vram_base_gb":    vram_base,
            "vram_peak_gb":    vram_peak,
            "delta_vram_gb":   delta,
            "per_instance_gb": per_inst,
            "avg_tps":         avg_tps,
            "avg_ttft_s":      avg_ttft,
            "avg_think_tok":   avg_think,
            "avg_resp_tok":    avg_resp,
            "elapsed_s":       elapsed,
            "errors":          len(results) - len(valid),
            "snapshots":       snaps.copy(),
        }
        st.session_state.measurements.append(measurement)
        st.session_state.log.append(
            f"[{measurement['timestamp']}] {measurement['model']} | "
            f"{'🧠' if measurement['think'] else '⚡'} | ctx={measurement['num_ctx']} | "
            f"x{n} → pic {vram_peak} GB (+{delta} GB) | "
            f"{avg_tps} tok/s | TTFT {avg_ttft}s | think_tok≈{avg_think}"
        )
        st.rerun()

    # ── POLLING live ──────────────────────────────────────────────────────────
    if st.session_state.running:
        now = time.time()
        elapsed_run = round(now - st.session_state._bench_start, 1)
        interval = st.session_state._bench_interval

        if now - st.session_state._last_snap_t >= interval:
            vram_now, _ = get_vram(st.session_state._bench_model)
            alive = sum(1 for t in st.session_state.threads if t.is_alive())
            st.session_state.snapshots.append({"t": elapsed_run, "vram_gb": vram_now, "active": alive})
            st.session_state._last_snap_t = now

        snaps = st.session_state.snapshots
        vram_cur = snaps[-1]["vram_gb"] if snaps else 0
        vram_base = snaps[0]["vram_gb"] if snaps else 0
        delta_live = round(vram_cur - vram_base, 3)
        alive_now = sum(1 for t in st.session_state.threads if t.is_alive())
        think_str = "🧠 think ON" if st.session_state._bench_think else "⚡ think OFF"

        st.markdown(f"### 🔴 En cours — {think_str}")
        c1, c2, c3, c4 = st.columns(4)
        for col, lbl, val, sub in [
            (c1, "VRAM actuelle",    f"{vram_cur} GB",    "ollama ps"),
            (c2, "Delta KV cache",   f"+{delta_live} GB", f"base {vram_base} GB"),
            (c3, "Instances actives",f"{alive_now}/{st.session_state._bench_n}", "threads vivants"),
            (c4, "Durée",            f"{elapsed_run}s",   "depuis le lancement"),
        ]:
            with col:
                st.markdown(
                    f'<div class="mc"><div class="ml">{lbl}</div>'
                    f'<div class="mv">{val}</div><div class="ms">{sub}</div></div>',
                    unsafe_allow_html=True,
                )

        if len(snaps) > 1:
            df_live = pd.DataFrame(snaps)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_live["t"], y=df_live["vram_gb"],
                mode="lines+markers", line=dict(color="#7c3aed", width=2),
                marker=dict(size=5, color="#a78bfa"),
                fill="tozeroy", fillcolor="rgba(124,58,237,0.08)",
            ))
            fig.update_layout(
                paper_bgcolor="#0a0a0f", plot_bgcolor="#0a0a0f",
                font=dict(color="#888", family="JetBrains Mono"),
                xaxis=dict(showgrid=False, title="Temps (s)"),
                yaxis=dict(gridcolor="#1e1e30", title="VRAM (GB)"),
                margin=dict(l=40, r=20, t=10, b=40), height=200, showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        time.sleep(interval)
        st.rerun()

    # ── Résultats ─────────────────────────────────────────────────────────────
    if st.session_state.measurements:
        last = st.session_state.measurements[-1]
        think_badge = (
            '<span class="think-badge">🧠 think ON</span>'
            if last["think"] else
            '<span class="nothink-badge">⚡ think OFF</span>'
        )
        st.markdown(f"### 📊 Dernière mesure &nbsp; {think_badge}", unsafe_allow_html=True)

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        metrics = [
            (c1, "VRAM pic",       f"{last['vram_peak_gb']} GB",    "max pendant le run"),
            (c2, "ΔKV cache",      f"+{last['delta_vram_gb']} GB",  f"base {last['vram_base_gb']} GB"),
            (c3, "GB/instance",    f"{last['per_instance_gb'] or 'n/a'} GB", f"x{last['n_parallel']} inst."),
            (c4, "Tokens/s",       f"{last['avg_tps'] or '—'}",     "moy. instances"),
            (c5, "TTFT",           f"{last['avg_ttft_s'] or '—'}s", "time-to-first-token"),
            (c6, "Think tokens",   f"{last['avg_think_tok']}",       f"réponse: {last['avg_resp_tok']}"),
        ]
        for col, lbl, val, sub in metrics:
            with col:
                st.markdown(
                    f'<div class="mc"><div class="ml">{lbl}</div>'
                    f'<div class="mv">{val}</div><div class="ms">{sub}</div></div>',
                    unsafe_allow_html=True,
                )

        # Courbe VRAM dernière mesure
        if last["snapshots"]:
            st.markdown("### 📈 Profil VRAM")
            df_s = pd.DataFrame(last["snapshots"])
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=df_s["t"], y=df_s["vram_gb"],
                mode="lines+markers", line=dict(color="#7c3aed", width=2),
                marker=dict(size=6, color="#a78bfa"),
                fill="tozeroy", fillcolor="rgba(124,58,237,0.08)",
            ))
            fig2.update_layout(
                paper_bgcolor="#0a0a0f", plot_bgcolor="#0a0a0f",
                font=dict(color="#888", family="JetBrains Mono"),
                xaxis=dict(showgrid=False, title="Temps (s)"),
                yaxis=dict(gridcolor="#1e1e30", title="VRAM (GB)"),
                margin=dict(l=40, r=20, t=10, b=40), height=240, showlegend=False,
            )
            st.plotly_chart(fig2, use_container_width=True)

        # ── Comparaison think ON vs OFF ───────────────────────────────────────
        df_all = pd.DataFrame([
            {
                "Heure":         m["timestamp"],
                "Modèle":        m["model"],
                "Mode":          "🧠 think" if m["think"] else "⚡ no_think",
                "num_ctx":       m["num_ctx"],
                "Instances":     m["n_parallel"],
                "VRAM pic (GB)": m["vram_peak_gb"],
                "ΔKV (GB)":      m["delta_vram_gb"],
                "GB/inst":       m["per_instance_gb"],
                "tok/s":         m["avg_tps"],
                "TTFT (s)":      m["avg_ttft_s"],
                "Think tok":     m["avg_think_tok"],
                "Resp tok":      m["avg_resp_tok"],
                "Durée (s)":     m["elapsed_s"],
            }
            for m in st.session_state.measurements
        ])

        # Graphique comparatif think ON vs OFF si les deux existent
        if df_all["Mode"].nunique() > 1:
            st.markdown("### 🔬 Comparaison think ON vs OFF")
            fig3 = go.Figure()
            colors = {"🧠 think": "#7c3aed", "⚡ no_think": "#34d399"}
            for mode, grp in df_all.groupby("Mode"):
                fig3.add_trace(go.Bar(
                    name=mode, x=grp.index.astype(str),
                    y=grp["tok/s"],
                    marker_color=colors.get(mode, "#888"),
                    text=grp["tok/s"], textposition="outside",
                ))
            fig3.update_layout(
                title="Tokens/s par mode",
                paper_bgcolor="#0a0a0f", plot_bgcolor="#0a0a0f",
                font=dict(color="#888", family="JetBrains Mono"),
                barmode="group",
                xaxis=dict(showgrid=False, title="Mesure #"),
                yaxis=dict(gridcolor="#1e1e30", title="tok/s"),
                legend=dict(bgcolor="#0f0f1a", bordercolor="#1e1e30", borderwidth=1),
                margin=dict(l=40, r=20, t=40, b=40), height=280,
            )
            st.plotly_chart(fig3, use_container_width=True)

            fig4 = go.Figure()
            for mode, grp in df_all.groupby("Mode"):
                fig4.add_trace(go.Bar(
                    name=mode, x=grp.index.astype(str),
                    y=grp["TTFT (s)"],
                    marker_color=colors.get(mode, "#888"),
                    text=grp["TTFT (s)"], textposition="outside",
                ))
            fig4.update_layout(
                title="Time-to-first-token (s)",
                paper_bgcolor="#0a0a0f", plot_bgcolor="#0a0a0f",
                font=dict(color="#888", family="JetBrains Mono"),
                barmode="group",
                xaxis=dict(showgrid=False, title="Mesure #"),
                yaxis=dict(gridcolor="#1e1e30", title="TTFT (s)"),
                legend=dict(bgcolor="#0f0f1a", bordercolor="#1e1e30", borderwidth=1),
                margin=dict(l=40, r=20, t=40, b=40), height=280,
            )
            st.plotly_chart(fig4, use_container_width=True)

        st.markdown("### 🗂 Historique")
        st.dataframe(df_all, use_container_width=True, hide_index=True)

        with st.expander("📋 Log"):
            for line in reversed(st.session_state.log):
                st.markdown(f'<div class="rr">{line}</div>', unsafe_allow_html=True)

    elif not st.session_state.running:
        st.markdown("""
        <div style="text-align:center;padding:80px 40px;">
            <div style="font-size:48px;margin-bottom:16px;">⚡</div>
            <div style="font-size:18px;font-family:'Syne',sans-serif;color:#444;">Lance un benchmark</div>
            <div style="font-size:13px;color:#333;margin-top:8px;">
                Alterne think ON / think OFF avec le même modèle et num_ctx<br>
                pour comparer VRAM, tokens/s et time-to-first-token.
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<hr style="border-color:#1e1e30;margin-top:40px;">
<p style="text-align:center;color:#333;font-size:11px;font-family:'JetBrains Mono',monospace;">
    /think · /no_think · VRAM via ollama ps · streaming maintenu pendant la mesure
</p>
""", unsafe_allow_html=True)