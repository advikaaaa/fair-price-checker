"""Fair Price Checker: is this Bengaluru listing's asking price fair?

Run:  streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import EXAMPLES, PROCESSED_CSV  # noqa: E402
from predict import FAIR, GOOD_DEAL, OVERPRICED, SUSPICIOUS, fmt_price, load_bundle, predict, similar_listings  # noqa: E402

st.set_page_config(page_title="Fair Price Checker", page_icon="🏠", layout="wide")

# Status colours: each verdict also has an icon and a text label, so colour is never the only cue.
VERDICT_STYLE = {
    GOOD_DEAL: {"color": "#0ca30c", "icon": "✅", "text": "#ffffff"},
    FAIR: {"color": "#2a78d6", "icon": "👍", "text": "#ffffff"},
    OVERPRICED: {"color": "#ec835a", "icon": "⚠️", "text": "#0b0b0b"},
    SUSPICIOUS: {"color": "#d03b3b", "icon": "🚩", "text": "#ffffff"},
}
RAISES, LOWERS = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#ecebe8"
OTHER_OPTION = "Other (not in list)"
AREA_TYPES = ["Super built-up", "Built-up", "Carpet", "Plot"]


# ---------------------------------------------------------------- cached loaders

@st.cache_resource
def get_model():
    return load_bundle()  # model + SHAP explainer, built once per server


@st.cache_data
def get_listings() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_CSV)


bundle = get_model()
listings = get_listings()
localities = sorted(l for l in bundle["localities"] if l != "other")
th = bundle["thresholds"]

# ---------------------------------------------------------------- session state & examples

DEFAULTS = {"location": "Whitefield", "other_location": "", "area_type": "Super built-up", "area_sqft": 1200,
            "bhk": 2, "bathrooms": 2, "balconies": 1, "status": "Ready to move", "asking_lakhs": 70.0}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)
st.session_state.setdefault("result", None)


def load_example(name: str):
    """Prefill the form from an example and run the check straight away (smooth live demo)."""
    ex = EXAMPLES[name]
    l = ex["listing"]
    st.session_state.update(location=l["location"], other_location="", area_type=l["area_type"],
                            area_sqft=l["area_sqft"], bhk=l["bhk"], bathrooms=l["bathrooms"],
                            balconies=l["balconies"], asking_lakhs=float(ex["asking_lakhs"]),
                            status="Ready to move" if l["ready_to_move"] else "Under construction")
    run_check()


def current_listing() -> dict:
    s = st.session_state
    location = s.other_location.strip() if s.location == OTHER_OPTION else s.location
    return {"location": location, "area_type": s.area_type, "area_sqft": float(s.area_sqft),
            "bhk": int(s.bhk), "bathrooms": int(s.bathrooms), "balconies": int(s.balconies),
            "ready_to_move": int(s.status == "Ready to move")}


def run_check():
    listing = current_listing()
    if st.session_state.location == OTHER_OPTION and not st.session_state.other_location.strip():
        st.session_state.result = (listing, None, "Please type the locality name (or pick one from the list).")
        return
    try:
        st.session_state.result = (listing, predict(listing, float(st.session_state.asking_lakhs)), None)
    except ValueError as e:
        st.session_state.result = (listing, None, str(e))


# Optional deep link, e.g. ?example=fair, opens the app with that example already checked.
EXAMPLE_LINKS = {name.split()[-2]: name for name in EXAMPLES}  # "fair", "overpriced", "suspicious"
if st.session_state.result is None and st.query_params.get("example") in EXAMPLE_LINKS:
    load_example(EXAMPLE_LINKS[st.query_params["example"]])

# ---------------------------------------------------------------- sidebar form

with st.sidebar:
    st.header("Listing details")
    st.selectbox("Locality", localities + [OTHER_OPTION], key="location",
                 help="Type to search. Pick 'Other' if the locality isn't listed.")
    if st.session_state.location == OTHER_OPTION:
        st.text_input("Locality name", key="other_location", placeholder="e.g. Varthur")
    st.selectbox("Area type", AREA_TYPES, key="area_type",
                 help="Super built-up includes common areas; Plot means an independent house with land.")
    st.number_input("Area (sq ft)", min_value=300, max_value=30000, step=50, key="area_sqft")
    c1, c2, c3 = st.columns(3)
    c1.number_input("BHK", min_value=1, max_value=10, step=1, key="bhk")
    c2.number_input("Baths", min_value=1, max_value=12, step=1, key="bathrooms")
    c3.number_input("Balconies", min_value=0, max_value=3, step=1, key="balconies")
    st.radio("Status", ["Ready to move", "Under construction"], key="status", horizontal=True)
    st.number_input("Asking price (₹ lakhs)", min_value=1.0, max_value=10000.0, step=1.0, key="asking_lakhs", format="%.1f",
                    help="100 lakhs = 1 crore")
    st.caption(f"= {fmt_price(st.session_state.asking_lakhs)}")
    st.button("Check price", type="primary", width="stretch", on_click=run_check)

# ---------------------------------------------------------------- header

st.title("🏠 Fair Price Checker")
st.markdown("Is this asking price fair? Get an estimated fair price for a Bengaluru property, "
            "a verdict, the reasons behind it, and how it compares with similar listings.")

cols = st.columns(len(EXAMPLES))
for col, name in zip(cols, EXAMPLES):
    col.button(name, width="stretch", on_click=load_example, args=(name,))


# ---------------------------------------------------------------- result sections

def verdict_badge(r):
    style = VERDICT_STYLE[r.verdict]
    lo, hi = r.range_lakhs
    st.markdown(
        f"""
<div style="background:{style['color']};color:{style['text']};border-radius:12px;padding:20px 24px;margin:8px 0 4px">
  <div style="font-size:0.95rem;opacity:0.9">Verdict</div>
  <div style="font-size:2rem;font-weight:700;line-height:1.2">{style['icon']} {r.verdict}</div>
  <div style="font-size:1.05rem;margin-top:6px">{r.reason}</div>
</div>""",
        unsafe_allow_html=True,
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("Asking price", fmt_price(r.asking_lakhs))
    m2.metric("Estimated fair price", fmt_price(r.estimate_lakhs))
    m3.metric("Fair range", f"{fmt_price(lo)} – {fmt_price(hi)}")


def factors_chart(r):
    f = pd.DataFrame(r.factors).iloc[::-1]  # biggest at the top of a horizontal bar chart
    signs = np.where(f.effect_lakhs > 0, "+", "−")
    fig = go.Figure(go.Bar(
        x=f.effect_lakhs, y=f.label, orientation="h",
        marker_color=np.where(f.effect_lakhs > 0, RAISES, LOWERS),
        text=[f"{s}{fmt_price(abs(v))}" for s, v in zip(signs, f.effect_lakhs)],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>%{text}<extra></extra>",
    ))
    span = f.effect_lakhs.abs().max() * 1.7  # room for the value labels on both sides
    fig.update_layout(height=70 + 48 * len(f), margin=dict(l=10, r=10, t=10, b=30), bargap=0.35,
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(size=14, color=INK), showlegend=False)
    fig.update_yaxes(tickfont=dict(size=14, color=INK))
    fig.update_xaxes(range=[-span, span], zeroline=True, zerolinecolor=MUTED, gridcolor=GRID,
                     showticklabels=False, title=dict(text="◀ lowers estimate      raises estimate ▶",
                                                      font=dict(size=12, color=MUTED)))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def nice_ticks(lo: float, hi: float, target: int = 6) -> np.ndarray:
    """Round tick values (1, 2 or 5 x a power of ten) so the price axis reads ₹20L, ₹40L, ..."""
    raw = (hi - lo) / target
    magnitude = 10 ** np.floor(np.log10(raw))
    step = min((m * magnitude for m in (1, 2, 5, 10)), key=lambda s: abs(s - raw))
    return np.arange(np.ceil(lo / step) * step, hi, step)


def context_chart(r, listing):
    # r.location is the matched locality ("other" if unknown, which falls back to city-wide listings).
    loc_for_data = r.location if r.location in localities else "__none__"
    similar, desc = similar_listings(listings, {**listing, "location": loc_for_data})
    if loc_for_data == "__none__":
        desc = desc.replace(" in __none__", "")
    prices = similar.price_lakhs
    lo_r, hi_r = r.range_lakhs

    # Bin prices ourselves so the chart looks the same everywhere.
    left = min(prices.quantile(0.01), r.asking_lakhs, lo_r) * 0.9
    right = max(prices.quantile(0.99), r.asking_lakhs, hi_r) * 1.1
    counts, edges = np.histogram(prices.clip(left, right), bins=30, range=(left, right))
    fig = go.Figure()
    fig.add_vrect(x0=lo_r, x1=hi_r, fillcolor=VERDICT_STYLE[FAIR]["color"], opacity=0.10, line_width=0)
    fig.add_trace(go.Bar(
        x=(edges[:-1] + edges[1:]) / 2, y=counts, width=np.diff(edges) * 0.9, marker_color="#9ec5f4",
        customdata=np.c_[edges[:-1], edges[1:]], name="Similar listings",
        hovertemplate="₹%{customdata[0]:.0f}L to ₹%{customdata[1]:.0f}L: %{y} listings<extra></extra>"))
    ymax = max(counts.max(), 1) * 1.35
    # Two labelled vertical lines. Labels sit at different heights and point away from each other,
    # so they never overlap even when asking price and estimate are close.
    asking_right = r.asking_lakhs >= r.estimate_lakhs
    lines = [
        (r.estimate_lakhs, f"Estimate {fmt_price(r.estimate_lakhs)}", "#1c5cab", "dash", ymax * 0.86,
         "right" if asking_right else "left"),
        (r.asking_lakhs, f"Asking {fmt_price(r.asking_lakhs)}", VERDICT_STYLE[r.verdict]["color"], "solid", ymax,
         "left" if asking_right else "right"),
    ]
    for x, label, color, dash, y, side in lines:
        fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=y, line=dict(color=color, width=3, dash=dash))
        fig.add_annotation(x=x, y=y, text=f"<b>{label}</b>", showarrow=False, yanchor="middle",
                           xanchor=side, xshift=6 if side == "left" else -6,
                           font=dict(color=INK, size=13), bgcolor="rgba(252,252,251,0.9)")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=40), showlegend=False,
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(size=13, color=INK))
    ticks = nice_ticks(left, right)
    fig.update_xaxes(title="Price", tickvals=ticks, ticktext=[f"₹{t / 100:g} Cr" if t >= 100 else f"₹{t:g}L" for t in ticks], gridcolor=GRID)
    fig.update_yaxes(title="Listings", gridcolor=GRID, range=[0, ymax * 1.12])
    st.caption(f"{len(similar):,} real listings: {desc}. Median {fmt_price(prices.median())}. "
               "Shaded band = fair range.")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def how_to_read():
    with st.expander("How to read this"):
        st.markdown(f"""
The model estimates a fair price from locality, area, BHK, bathrooms, balconies, area type and
construction status, learned from about 8,000 real Bengaluru listings. We then compare the asking
price with that estimate (**asking ÷ estimate**):

| Verdict | Asking ÷ estimate | Meaning |
|---|---|---|
| ✅ Good Deal | {th['suspicious_low']:.2f} – {th['fair_low']:.2f} | Cheaper than most similar listings |
| 👍 Fairly Priced | {th['fair_low']:.2f} – {th['fair_high']:.2f} | Within the normal spread (the "fair range") |
| ⚠️ Overpriced | {th['fair_high']:.2f} – {th['suspicious_high']:.2f} | Pricier than most similar listings |
| 🚩 Suspicious | below {th['suspicious_low']:.2f} or above {th['suspicious_high']:.2f} | Far outside normal spread: possible typo, missing information, or a listing too good to be true |

These bands were set from real listings the model had not seen: about **80%** of them fall in
"Fairly Priced", and about **5%** are far enough out to be flagged as suspicious.

**Why this estimate** shows how each detail moves the estimate compared with a typical Bengaluru
listing (about {fmt_price(r_typical)}), using SHAP values.
""")


# ---------------------------------------------------------------- main area

r_typical = np.exp(bundle["explainer"].expected_value[0])
result = st.session_state.result

if result is None:
    st.info("Enter the listing details in the sidebar and click **Check price**, or try one of the examples above.")
else:
    listing, r, error = result
    if error:
        st.error(f"Please check the inputs: {error}")
    else:
        verdict_badge(r)
        for note in r.notes:
            st.warning(note)
        left, right = st.columns([1, 1.25], gap="large")
        with left:
            st.subheader("Why this estimate")
            st.caption(f"Compared with a typical Bengaluru listing ({fmt_price(r_typical)}).")
            factors_chart(r)
        with right:
            st.subheader("Market context")
            context_chart(r, listing)
how_to_read()

# ---------------------------------------------------------------- footer

mape = bundle["test_metrics"]["MAPE_pct"]
st.divider()
st.caption(
    f"**Data:** [Bengaluru House Price Data](https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data) "
    f"(Kaggle), {len(listings):,} listings after cleaning; prices reflect when the data was collected (around 2017–2020).  \n"
    f"**Model:** {bundle['model_name']} on log(price); typical error about {mape:.0f}% on held-out listings.  \n"
    "**Disclaimer:** an educational estimate, not financial or legal advice. It cannot see photos, exact location, "
    "floor, condition or amenities."
)
