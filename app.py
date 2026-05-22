"""
app.py — Deep Visual Retrieval · Streamlit UI
Run:
  uvicorn api.search:app --host 0.0.0.0 --port 8000
  streamlit run app.py
"""

import json
from pathlib import Path

import httpx
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from PIL import Image

API_BASE = "http://localhost:8000"
BASE_DIR = Path(__file__).resolve().parent

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SnapMatch · Image-Based Product Retrieval System",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: #1a1a2e;
  }
  [data-testid="stSidebar"] * {
    color: #e0e0e0 !important;
  }
  [data-testid="stSidebar"] .stRadio label {
    font-size: 15px;
    padding: 6px 0;
  }

  /* ── Product card ── */
  .product-card {
    border: 1px solid #e8e8e8;
    border-radius: 8px;
    padding: 10px;
    background: #fff;
    transition: box-shadow 0.2s;
    margin-bottom: 12px;
  }
  .product-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.10); }

  .rank-badge {
    display: inline-block;
    background: #232f3e;
    color: #fff;
    border-radius: 4px;
    padding: 2px 7px;
    font-size: 11px;
    font-weight: 700;
    margin-right: 5px;
  }
  .cat-pill {
    display: inline-block;
    background: #f0f2f6;
    color: #333;
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 11px;
    border: 1px solid #ddd;
  }
  .score-line {
    font-size: 12px;
    color: #e47911;
    font-weight: 600;
    margin-top: 4px;
  }

  /* ── Category buttons — equal size grid ── */
  .cat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin: 12px 0 20px 0;
  }
  .cat-btn {
    border: 1.5px solid #e0e0e0;
    border-radius: 8px;
    padding: 14px 10px;
    text-align: center;
    background: #fff;
    cursor: pointer;
    transition: all 0.18s;
    font-size: 14px;
    font-weight: 600;
    color: #232f3e;
    line-height: 1.4;
  }
  .cat-btn:hover {
    border-color: #e47911;
    background: #fff8f0;
    color: #e47911;
  }
  .cat-btn.active {
    border-color: #232f3e;
    background: #232f3e;
    color: #fff;
  }
  .cat-count {
    font-size: 11px;
    font-weight: 400;
    color: #888;
    display: block;
    margin-top: 3px;
  }
  .cat-btn.active .cat-count { color: #ccc; }

  /* ── Page banner ── */
  .page-banner {
    background: linear-gradient(90deg, #232f3e 0%, #37475a 100%);
    border-radius: 10px;
    padding: 18px 28px;
    margin-bottom: 20px;
  }
  .page-banner h1 {
    color: #fff !important;
    font-size: 24px !important;
    font-weight: 700 !important;
    margin: 0 0 4px 0 !important;
  }
  .page-banner p {
    color: #f0c97a !important;
    font-size: 13px !important;
    margin: 0 !important;
  }
  /* ── Subsection label ── */
  .sub-header {
    font-size: 17px;
    font-weight: 700;
    color: #232f3e;
    border-left: 4px solid #e47911;
    padding-left: 10px;
    margin: 16px 0 10px 0;
    display: block;
  }

  /* ── Metric note ── */
  .metric-note {
    font-size: 12px;
    color: #888;
    margin-top: -6px;
  }

  /* ── Search bar area ── */
  .search-box input {
    border-radius: 4px 0 0 4px !important;
    border: 2px solid #e47911 !important;
    font-size: 15px !important;
  }

  /* ── Hide Streamlit chrome ── */
  #MainMenu, footer { visibility: hidden; }
  .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_client():
    return httpx.Client(base_url=API_BASE, timeout=30.0)


def api_get(path: str) -> dict | None:
    try:
        r = get_client().get(path)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def load_eval_report() -> dict | None:
    for fname in ["evaluation_report_relaxed.json", "evaluation_report.json"]:
        p = BASE_DIR / "outputs" / fname
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return None


def load_img(path: str) -> Image.Image | None:
    p = Path(path)
    if p.exists():
        return Image.open(p).convert("RGB")
    return None


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛍️ SnapMatch")
    st.markdown("Image-Based Product Retrieval System")
    st.markdown("**Stanford Product Dataset**")
    st.markdown("*ResNet50 · FAISS · MySQL*")
    st.divider()

    page = st.radio(
        "nav",
        ["🔍  Visual Search", "📊  Eval Metrics", "🗂  Browse by Category"],
        label_visibility="collapsed",
    )

    st.divider()
    health = api_get("/health")
    if health:
        st.success("● API online", icon=None)
        st.caption(f"**{health.get('total_vectors', 0):,}** vectors · **{health.get('total_images', 0):,}** images")
    else:
        st.error("● API offline")
        st.caption("Run: `uvicorn api.search:app --port 8000`")

    st.divider()
    st.caption("Indupriya Chidambararaj · 2026")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — VISUAL SEARCH
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔍  Visual Search":
    st.markdown('''<div class="page-banner"><h1>🔍 Visual Product Search</h1><p>Upload a product image — find the most visually similar items from the Stanford dataset.</p></div>''', unsafe_allow_html=True)
    st.write("")

    col_upload, col_gap, col_controls = st.columns([3, 0.3, 1.5])

    with col_controls:
        st.markdown("**Search settings**")
        top_k = st.slider("Results (Top-K)", min_value=4, max_value=20, value=8, step=2)
        st.caption("Cosine similarity via L2-normalised ResNet50 embeddings + FAISS IndexFlatIP.")

    with col_upload:
        uploaded = st.file_uploader(
            "Drop a product image here",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            help="JPEG or PNG · max 10 MB",
        )
        if uploaded:
            query_img = Image.open(uploaded).convert("RGB")
            st.image(query_img, caption="Query image", width=240)

    st.write("")
    search_clicked = uploaded and st.button("🔍  Search similar products", type="primary")

    if search_clicked:
        uploaded.seek(0)
        image_bytes = uploaded.read()

        with st.spinner("Searching..."):
            try:
                response = get_client().post(
                    "/search",
                    files={"file": (uploaded.name, image_bytes, uploaded.type)},
                    params={"top_k": top_k},
                )
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                st.error(f"Search failed: {e}")
                data = None

        if data:
            results      = data.get("results", [])
            search_ms    = data.get("search_time_ms", 0)
            top_score    = results[0]["similarity_score"] if results else 0

            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("Results", len(results))
            m2.metric("Search time", f"{search_ms:.1f} ms")
            m3.metric("Top similarity", f"{top_score:.4f}")
            st.write("")

            if not results:
                st.info("No results found. Try a different image.")
            else:
                st.markdown(f'<span class="sub-header">Top {len(results)} Similar Products</span>', unsafe_allow_html=True)
                st.write("")
                cols_per_row = 4
                for row_start in range(0, len(results), cols_per_row):
                    row_items = results[row_start: row_start + cols_per_row]
                    cols = st.columns(cols_per_row)
                    for col, item in zip(cols, row_items):
                        with col:
                            img_path = item.get("image_url", "").replace("/image?path=", "")
                            img = load_img(img_path)
                            if img:
                                st.image(img, use_container_width=True)
                            else:
                                st.image(f"{API_BASE}{item['image_url']}", use_container_width=True)
                            st.markdown(
                                f'<span class="rank-badge">#{item["rank"]}</span>'
                                f'<span class="cat-pill">{item["display_name"]}</span>'
                                f'<div class="score-line">Score: {item["similarity_score"]:.4f}</div>',
                                unsafe_allow_html=True,
                            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EVAL METRICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Eval Metrics":
    st.markdown('''<div class="page-banner"><h1>📊 Evaluation Metrics</h1><p>Relaxed evaluation — same super_class_id (same product type) · 200 test queries.</p></div>''', unsafe_allow_html=True)
    st.write("")

    report = load_eval_report()
    if not report:
        st.warning("No evaluation report found. Run `python src/evaluate.py` first.")
        st.stop()

    pk     = report.get("mean_precision_at_k", {})
    rk     = report.get("mean_recall_at_k", {})
    cat_m  = report.get("category_metrics", {})
    avg_ms = report.get("mean_search_time_ms", 0)
    max_ms = report.get("max_search_time_ms", 0)
    n_q    = report.get("n_queries", 200)

    # ── Top metrics ──
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Queries",        n_q)
    c2.metric("P@1",            f"{float(pk.get('1', 0)):.1%}")
    c3.metric("P@5",            f"{float(pk.get('5', 0)):.1%}")
    c4.metric("Avg search",     f"{avg_ms:.1f} ms")
    c5.metric("Max search",     f"{max_ms:.1f} ms")

    st.write("")
    st.divider()
    col_left, col_right = st.columns(2)
    ks = [1, 5, 10, 20]

    # ── Precision@K ──
    with col_left:
        st.markdown("**Precision@K**")
        p_vals = [float(pk.get(str(k), 0)) * 100 for k in ks]
        fig = go.Figure(go.Bar(
            x=[f"K={k}" for k in ks], y=p_vals,
            marker_color="#e47911",
            text=[f"{v:.1f}%" for v in p_vals], textposition="outside",
        ))
        fig.update_layout(
            yaxis=dict(title="Precision (%)", range=[0, 110]),
            xaxis_title="K", height=280,
            margin=dict(t=10, b=10, l=10, r=10),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<p class="metric-note">Fraction of Top-K belonging to the same product type.</p>', unsafe_allow_html=True)

    # ── Recall@K ──
    with col_right:
        st.markdown("**Recall@K**")
        r_vals = [float(rk.get(str(k), 0)) * 100 for k in ks]
        fig2 = go.Figure(go.Bar(
            x=[f"K={k}" for k in ks], y=r_vals,
            marker_color="#232f3e",
            text=[f"{v:.3f}%" for v in r_vals], textposition="outside",
        ))
        fig2.update_layout(
            yaxis=dict(title="Recall (%)"),
            xaxis_title="K", height=280,
            margin=dict(t=10, b=10, l=10, r=10),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('<p class="metric-note">Low recall is expected — each category has ~800 relevant images; K ≤ 20.</p>', unsafe_allow_html=True)

    st.divider()

    # ── Per-category precision ──
    st.markdown("**Per-category Precision**")
    k_choice = st.radio("K =", [1, 5, 10, 20], index=1, horizontal=True)
    pk_key   = f"precision@{k_choice}"

    if cat_m:
        df = pd.DataFrame([
            {
                "Category":      k.replace("_final", "").replace("_", " ").title(),
                "Precision (%)": round(v.get(pk_key, 0) * 100, 1),
            }
            for k, v in sorted(cat_m.items(), key=lambda x: -x[1].get(pk_key, 0))
        ])
        fig3 = px.bar(
            df, x="Precision (%)", y="Category", orientation="h",
            text="Precision (%)",
            color="Precision (%)",
            color_continuous_scale=["#D85A30", "#EF9F27", "#1D9E75"],
            range_color=[0, 100],
        )
        fig3.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig3.update_layout(
            margin=dict(t=10, b=10, l=10, r=70),
            height=400, coloraxis_showscale=False,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — BROWSE BY CATEGORY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗂  Browse by Category":
    st.markdown('''<div class="page-banner"><h1>🗂 Browse by Category</h1><p>Search a category and explore sample product images from the dataset.</p></div>''', unsafe_allow_html=True)
    st.write("")

    cat_data = api_get("/categories")
    if not cat_data:
        st.error("Could not load categories — is the API running?")
        st.stop()

    categories = cat_data.get("categories", [])

    # ── Search box ────────────────────────────────────────────────────────────
    search_query = st.text_input(
        "search",
        placeholder="🔎  Search a category — e.g. mug, chair, bicycle ...",
        label_visibility="collapsed",
    )

    matched = [
        c for c in categories
        if not search_query
        or search_query.lower() in c["display_name"].lower()
        or search_query.lower() in c["name"].lower()
    ]

    if not matched:
        st.warning(f"No category matched **{search_query}**. Try: mug, chair, fan, lamp ...")
        st.stop()

    st.caption(f"{len(matched)} categor{'y' if len(matched)==1 else 'ies'}")
    st.write("")

    # ── Category tiles — equal size, 4-per-row grid ───────────────────────────
    # Build HTML tile grid (equal size, hover effect, no Streamlit button sizing issues)
    ICONS = {
        "bicycle": "🚲", "cabinet": "🗄️", "chair": "🪑",
        "coffee": "☕", "fan": "💨",   "kettle": "🫖",
        "lamp":   "🪔", "mug":    "🍵", "sofa":  "🛋️",
        "stapler":"📎", "table":  "🪵", "toaster":"🍞",
    }
    def get_icon(name: str) -> str:
        for k, v in ICONS.items():
            if k in name.lower():
                return v
        return "📦"

    cols_per_row = 4
    rows = [matched[i:i+cols_per_row] for i in range(0, len(matched), cols_per_row)]

    for row in rows:
        cols = st.columns(cols_per_row)
        for col, cat in zip(cols, row):
            label   = cat["display_name"].title()
            count   = cat.get("total_images", "")
            icon    = get_icon(cat["name"])
            is_sel  = st.session_state.get("selected_cat_name") == cat["name"]
            btn_lbl = f"{icon} {label}\n{count} images"
            with col:
                if st.button(
                    btn_lbl,
                    key=f"cat_{cat['name']}",
                    use_container_width=True,
                    type="primary" if is_sel else "secondary",
                ):
                    st.session_state["selected_cat_name"]    = cat["name"]
                    st.session_state["selected_cat_display"] = label
                    st.rerun()

    # ── Sample image grid ─────────────────────────────────────────────────────
    if "selected_cat_name" in st.session_state:
        cat_name    = st.session_state["selected_cat_name"]
        cat_display = st.session_state["selected_cat_display"]
        cat_info    = next((c for c in categories if c["name"] == cat_name), {})
        total       = cat_info.get("total_images", "?")

        st.divider()
        left, right = st.columns([3, 1])
        with left:
            st.markdown(f'<span class="sub-header">{get_icon(cat_name)} {cat_display}</span>', unsafe_allow_html=True)
            st.caption(f"{total} total images")
        with right:
            num_samples = st.select_slider("Show", options=[4, 8, 12, 16, 20], value=8, key="browse_k")

        st.write("")

        with st.spinner(f"Loading {cat_display} images..."):
            browse_data = api_get(f"/categories/{cat_name}/samples?limit={num_samples}")

        products = browse_data.get("products", []) if browse_data else []

        # Fallback: load from disk
        if not products:
            cat_dir = BASE_DIR / "data" / cat_name
            if cat_dir.exists():
                all_imgs = list(cat_dir.rglob("*.jpg")) + list(cat_dir.rglob("*.JPG"))
                all_imgs = sorted(all_imgs)[:num_samples]
                products = [{"image_path": str(p), "class_id": p.parent.name, "split": "", "display_name": cat_display} for p in all_imgs]

        if products:
            cols_per_row = 4
            for row_start in range(0, len(products), cols_per_row):
                row_items = products[row_start: row_start + cols_per_row]
                rcols = st.columns(cols_per_row)
                for col, product in zip(rcols, row_items):
                    with col:
                        img = load_img(product.get("image_path", ""))
                        if img:
                            st.image(img, use_container_width=True)
                        display = product.get("display_name", cat_display)
                        split   = product.get("split", "")
                        st.markdown(
                            f'<div style="font-size:12px;font-weight:600;color:#232f3e;margin-top:4px">{display}</div>'
                            f'<div style="font-size:11px;color:#888">{split}</div>',
                            unsafe_allow_html=True,
                        )
        else:
            st.info("No images found. Check data/ folder or the API /categories endpoint.")