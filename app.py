import streamlit as st
import pandas as pd
from pymongo import MongoClient
import plotly.graph_objects as go
import seaborn as sns
import numpy as np

st.set_page_config(page_title="Vessel Power vs FOC", layout="wide")

def add_trendline(fig, x, y, label, color, line_style='dash'):
    # Convert to NumPy and clean
    x = np.array(x)
    y = np.array(y)

    # Remove NaNs, Infs, and check length
    mask = (
        np.isfinite(x) &
        np.isfinite(y)
    )
    x = x[mask]
    y = y[mask]

    if len(x) < 2 or np.all(x == x[0]):  # All x values same = no fit possible
        return

    try:
        coeffs = np.polyfit(x, y, 1)  # Linear fit
        x_vals = np.linspace(min(x), max(x), 100)
        y_vals = coeffs[0] * x_vals + coeffs[1]

        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode="lines",
            name=label,
            line=dict(color=color, width=5),
            showlegend=True
        ))
    except np.linalg.LinAlgError:
        # Skip trendline if fitting fails
        return


vessel_names = {1023 : "MH Perseus" ,
1005 : "PISCES",
1007 : "CAPELLA",
1017 : "CETUS",
1004 : "CASSIOPEIA",
1021 : "PYXIS",
1032 : "Cenataurus",
1016 : "CHARA",
1018 : "CARINA"}

name_to_id = {v: k for k, v in vessel_names.items()}

@st.cache_data(ttl=600)  # cache for 10 minutes
def load_data(collection_name):
    # connect and query once
    mongo_uri = st.secrets["mongo"]["uri"]
    client = MongoClient(mongo_uri)
    db = client["seaker_data"]
    collection = db[collection_name]

    data = list(collection.find())
    df = pd.DataFrame(data)
    df.drop(columns=['_id'], inplace=True)

    return df

df = load_data("Updated_autolog_complete_input_ideal_power_foc_7000series_except1004")
# df = pd.read_csv("../Data/Updated_autolog_complete_input_ideal_power_foc_7000series_except1004.csv")

df["ideal_foc"] = df["ideal_foc_hr"]

# Compute derived fields
df["MeanDraft"] = (df["DraftAftTele"] + df["DraftFwdTele"]) / 2
df["LCVCorrectedFOC"] = (((df["MEFuelMassCons"] / 1000)  * df["MEFuelLCV"] /40.6)/df["ME1RunningHoursMinute"])*1440

df["ideal_foc"] = (df["ideal_foc"]/df["ME1RunningHoursMinute"]) * 1440


# Streamlit setup

st.title("Power vs FOC")

# Vessel filter
vessels = sorted(df["VesselId"].unique())

base_colors = sns.color_palette("Set1", n_colors=8)
color_map = {}
for i, vessel in enumerate(vessels):
    dark = base_colors[i % len(base_colors)]
    light = tuple(min(1, c + 0.4) for c in dark)
    color_map[vessel] = {
        "actual": f"rgba({int(dark[0]*255)}, {int(dark[1]*255)}, {int(dark[2]*255)}, 1)",
        "ideal": f"rgba({int(light[0]*255)}, {int(light[1]*255)}, {int(light[2]*255)}, 0.6)"
    }

# Divide screen: 1 column for filters, 3 for plot (adjust ratio as needed)
filter_col, plot_col = st.columns([1, 3])

# ---- LEFT COLUMN: FILTERS ----
with filter_col:
    st.markdown("### Filter Inputs")

    display_mode = st.selectbox(
    "Select Display Mode",
    options=["Scatter Only", "Trendline Only", "Scatter + Trendline"],
    index=0  # default to Scatter Only
)

    # Prepare list of vessel names for selection
    available_ids = sorted(df["VesselId"].unique())
    available_names = [vessel_names.get(v, str(v)) for v in available_ids]

    # Show vessel names to the user
    selected_names = st.multiselect(
        "Select Vessels",
        options=available_names,
        default=[available_names[0]]
    )

    # Convert selected names back to IDs
    selected_vessels = [name_to_id[name] for name in selected_names]

    min_draft = st.number_input("Min Draft", value=float(df["MeanDraft"].min()), step=0.1, format="%.2f")
    max_draft = st.number_input("Max Draft", value=float(df["MeanDraft"].max()), step=0.1, format="%.2f")
    min_speed = st.number_input("Min SpeedOG", value=float(df["SpeedOG"].min()), step=0.1, format="%.2f")
    max_speed = st.number_input("Max SpeedOG", value=float(df["SpeedOG"].max()), step=0.1, format="%.2f")

# ---- Filter data based on inputs (no change) ----
filtered_df = df[
    (df["VesselId"].isin(selected_vessels)) &
    (df["MeanDraft"] >= min_draft) & (df["MeanDraft"] <= max_draft) &
    (df["SpeedOG"] >= min_speed) & (df["SpeedOG"] <= max_speed)
]

# ---- RIGHT COLUMN: PLOT ----
with plot_col:
    # Create and show plot (same code you already have)
    fig = go.Figure()

    for vessel in selected_vessels:
        vessel_data = filtered_df[filtered_df["VesselId"] == vessel]
        if vessel_data.empty:
            continue

        vessel_label = vessel_names.get(vessel, str(vessel))

        color_actual = color_map[vessel]["actual"]
        color_ideal = color_map[vessel]["ideal"]

        # --- SCATTERS ---
        if display_mode in ["Scatter Only", "Scatter + Trendline"]:
            fig.add_trace(go.Scatter(
                x=vessel_data["ME1ShaftPower"],
                y=vessel_data["LCVCorrectedFOC"],
                mode='markers',
                name=f"{vessel_label} - Actual",
                marker=dict(symbol='circle', size=10, color=color_actual),
                legendgroup=str(vessel),
            ))

            fig.add_trace(go.Scatter(
                x=vessel_data["ideal_power"],
                y=vessel_data["ideal_foc"],
                mode='markers',
                name=f"{vessel_label} - Ideal",
                marker=dict(symbol='diamond', size=10, color=color_ideal),
                legendgroup=str(vessel),
            ))

        # --- TRENDLINES ---
        if display_mode in ["Trendline Only", "Scatter + Trendline"]:
            add_trendline(fig,
                        vessel_data["ME1ShaftPower"],
                        vessel_data["LCVCorrectedFOC"],
                        label=f"{vessel_label} - Actual Trend",
                        color=color_ideal,
                        line_style="dot")
            add_trendline(fig,
                        vessel_data["ideal_power"],
                        vessel_data["ideal_foc"],
                        label=f"{vessel_label} - Ideal Trend",
                        color=color_actual,
                        line_style="dot")


    fig.update_layout(
        title="FOC vs Power: Actual ● vs Ideal ◆",
        xaxis_title="Power (kW)",
        yaxis_title="Fuel Oil Consumption (MT/day)",
        legend_title="Vessel ID + Type",
        height=700
    )

    st.plotly_chart(fig, use_container_width=True)
