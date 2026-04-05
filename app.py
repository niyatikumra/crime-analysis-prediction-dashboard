import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
import base64
# -------------------------------
# Page Config & UI Enhancement
# -------------------------------
st.set_page_config(page_title="Crime Sentinel AI - India", layout="wide")

# Enhanced Modern UI Styling
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #31333f;
        box-shadow: 0 8px 16px rgba(0,0,0,0.4);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent;
        font-size: 16px;
    }
    .stTabs [aria-selected="true"] {
        border-bottom: 3px solid #ff4b4b !important;
        font-weight: bold;
        color: #ff4b4b;
    }
    div[data-testid="stExpander"] {
        border-radius: 12px;
        border: 1px solid #31333f;
        background-color: #161b22;
    }
    </style>
    """, unsafe_allow_html=True)

# -------------------------------
# Load Data & Population Reference
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("01_District_wise_crimes_committed_IPC_2001_2012.csv")
    # Store wide format for Correlation Analysis before melting
    df_wide = df.copy()
    df_long = df.melt(
        id_vars=["STATE/UT", "DISTRICT", "YEAR"],
        var_name="Crime_Type",
        value_name="Cases"
    )
    df_long["Cases"] = pd.to_numeric(df_long["Cases"], errors="coerce").fillna(0)
    return df_long[df_long["Cases"] >= 0], df_wide

df_long, df_wide = load_data()

# Census 2011 Population Data (Millions) for Per-Capita Feature
pop_dict = {
    'UTTAR PRADESH': 199.8, 'MAHARASHTRA': 112.3, 'BIHAR': 104.0, 'WEST BENGAL': 91.2,
    'ANDHRA PRADESH': 84.5, 'MADHYA PRADESH': 72.6, 'TAMIL NADU': 72.1, 'RAJASTHAN': 68.5,
    'KARNATAKA': 61.0, 'GUJARAT': 60.4, 'ODISHA': 41.9, 'KERALA': 33.4, 'JHARKHAND': 32.9,
    'ASSAM': 31.2, 'PUNJAB': 27.7, 'HARYANA': 25.3, 'DELHI UT': 16.7, 'CHHATTISGARH': 25.5,
    'JAMMU & KASHMIR': 12.5, 'UTTARAKHAND': 10.0, 'HIMACHAL PRADESH': 6.8, 'TRIPURA': 3.6,
    'MEGHALAYA': 2.9, 'MANIPUR': 2.5, 'NAGALAND': 1.9, 'GOA': 1.4, 'ARUNACHAL PRADESH': 1.3
}
# -------------------------------
# Logic: Safety Score & PDF
# -------------------------------
def get_safety_score(current_state):
    state_total = df_long[df_long["STATE/UT"] == current_state]["Cases"].sum()
    pop = pop_dict.get(current_state, 10.0)
    rate = state_total / pop
    
    # Simple benchmark logic
    all_rates = [df_long[df_long["STATE/UT"] == s]["Cases"].sum() / p for s, p in pop_dict.items()]
    max_r = max(all_rates)
    score = 100 - ((rate / max_r) * 100)
    return round(max(0, min(100, score)), 1)

def create_pdf(state_name, score, total_cases):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 24)
    pdf.cell(200, 20, "Crime Analysis Executive Report", ln=True, align='C')
    pdf.set_font("Arial", '', 14)
    pdf.cell(200, 10, f"Target State: {state_name}", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, f"Safety Score: {score}/100", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 10, f"The state of {state_name} has reported a total of {total_cases:,} IPC cases between 2001-2012. Based on the 2011 Census population, the calculated safety index is {score}.")
    return pdf.output(dest='S').encode('latin-1')

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("🚔 Crime Sentinel AI")
state = st.sidebar.selectbox("Select Primary State", sorted(df_long["STATE/UT"].unique()))
filtered_data = df_long[df_long["STATE/UT"] == state]

# -------------------------------
# TABS (Expanded to 6 Tabs)
# -------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Dashboard", 
    "🗺️ Geospatial",
    "🔍 Deep Dive", 
    "🔮 Prediction", 
    "🚨 Anomalies",
    "📊 Recent Data"
])

# ===============================
# 📊 TAB 1: DASHBOARD (Enhanced with Per-Capita)
# ===============================
# ===============================
# 📊 TAB 1: DASHBOARD (Updated with Safety Score & PDF)
# ===============================
with tab1:
    total_cases = int(filtered_data["Cases"].sum())
    avg_cases = int(df_long.groupby("STATE/UT")["Cases"].sum().mean())
    total_india = int(df_long["Cases"].sum())
    
    # New Per Capita Logic
    state_pop_m = pop_dict.get(state, 10.0) 
    per_capita = (total_cases / (state_pop_m * 1_000_000)) * 100_000

    col1, col2, col3 = st.columns(3)
    col1.metric("State Total Crimes", f"{total_cases:,}")
    col2.metric("Crime Rate (Per 100k)", f"{round(per_capita, 2)}")
    col3.metric("India Total Crimes", f"{total_india:,}")

    st.divider()

    c1, c2 = st.columns([1, 2])
    with c1:
        # --- FIX: CALLING THE SAFETY SCORE ---
        s_score = get_safety_score(state)
        
        # Determine color based on score
        if s_score > 70: s_color = "#2ecc71" # Green
        elif s_score > 40: s_color = "#f1c40f" # Yellow
        else: s_color = "#e74c3c" # Red

        # Display the Safety Score Card
        st.markdown(f"""
            <div style="background-color: {s_color}; padding: 20px; border-radius: 15px; text-align: center; color: white;">
                <h3 style="margin:0;">SAFETY SCORE</h3>
                <h1 style="margin:0; font-size: 50px;">{s_score}</h1>
            </div>
        """, unsafe_allow_html=True)
        
        st.write("") # Spacer

        # --- FIX: ADDING THE PDF DOWNLOAD BUTTON ---
        try:
            pdf_data = create_pdf(state, s_score, total_cases)
            st.download_button(
                label="📥 Download Executive Report (PDF)",
                data=pdf_data,
                file_name=f"{state}_Crime_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"PDF Error: {e}")

        st.divider()

        st.subheader("⚠️ Risk Analysis")
        if total_cases > 500000:
            st.error("High Risk 🚨")
        elif total_cases > 200000:
            st.warning("Moderate Risk ⚠️")
        else:
            st.success("Low Risk ✅")
        
        risk_score = total_cases / avg_cases
        st.metric("Risk Score Index", round(risk_score, 2))
        st.info(f"💡 Based on {state_pop_m}M Population (Census 2011)")

    with c2:
        st.subheader("⚡ State Risk Ranking (Top 10)")
        risk_rank = df_long.groupby("STATE/UT")["Cases"].sum().sort_values(ascending=False).reset_index()
        fig_rank = px.bar(risk_rank.head(10), x="STATE/UT", y="Cases", color="Cases", color_continuous_scale='Reds')
        fig_rank.update_layout(template="plotly_dark", height=450)
        st.plotly_chart(fig_rank, use_container_width=True)

# ===============================
# 🗺️ TAB 2: GEOSPATIAL (New Feature)
# ===============================

    # ===============================
# 🗺️ TAB 2: GEOSPATIAL (Fixed with Coordinates)
# ===============================
with tab2:
    st.subheader("📍 India Crime Intensity Map")
    
    # Coordinates for Indian States to fix the "Black Map" issue
    state_coords = {
        'ANDHRA PRADESH': [15.91, 79.74], 'ARUNACHAL PRADESH': [28.21, 94.72], 'ASSAM': [26.20, 92.93],
        'BIHAR': [25.09, 85.31], 'CHHATTISGARH': [21.27, 81.86], 'GOA': [15.29, 74.12],
        'GUJARAT': [22.25, 71.19], 'HARYANA': [29.05, 76.08], 'HIMACHAL PRADESH': [31.10, 77.17],
        'JAMMU & KASHMIR': [33.77, 76.57], 'JHARKHAND': [23.61, 85.27], 'KARNATAKA': [15.31, 75.71],
        'KERALA': [10.85, 76.27], 'MADHYA PRADESH': [22.97, 78.65], 'MAHARASHTRA': [19.75, 75.71],
        'MANIPUR': [24.66, 93.90], 'MEGHALAYA': [25.46, 91.36], 'MIZORAM': [23.16, 92.93],
        'NAGALAND': [26.15, 94.56], 'ODISHA': [20.95, 85.09], 'PUNJAB': [31.14, 75.34],
        'RAJASTHAN': [27.02, 74.21], 'SIKKIM': [27.53, 88.51], 'TAMIL NADU': [11.12, 78.65],
        'TRIPURA': [23.94, 91.98], 'UTTAR PRADESH': [26.84, 80.94], 'UTTARAKHAND': [30.06, 79.01],
        'WEST BENGAL': [22.98, 87.85], 'DELHI UT': [28.61, 77.20]
    }

    state_map_data = df_long.groupby("STATE/UT")["Cases"].sum().reset_index()
    
    # Mapping the coordinates to our dataframe
    state_map_data['lat'] = state_map_data['STATE/UT'].map(lambda x: state_coords.get(x, [0,0])[0])
    state_map_data['lon'] = state_map_data['STATE/UT'].map(lambda x: state_coords.get(x, [0,0])[1])
    
    # Filter out states where we don't have coordinates
    state_map_data = state_map_data[state_map_data['lat'] != 0]

    fig_map = px.scatter_geo(
        state_map_data, 
        lat="lat", 
        lon="lon", 
        size="Cases", 
        color="Cases",
        hover_name="STATE/UT", 
        scope='asia',
        color_continuous_scale="Reds",
        template="plotly_dark"
    )

    # Adjusting the zoom to focus purely on India
    fig_map.update_geos(
        visible=True, 
        resolution=50,
        showcountries=True, 
        countrycolor="RebeccaPurple",
        lataxis_range=[6, 38], 
        lonaxis_range=[68, 98]
    )
    
    fig_map.update_layout(height=600, margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)

# ===============================
# 🔍 TAB 3: DEEP DIVE (Original restored + Correlation)
# ===============================
with tab3:
    st.subheader("🔄 State Comparison")
    state2 = st.sidebar.selectbox("Compare with State", [s for s in df_long["STATE/UT"].unique() if s != state], key="comp_state")

    c_data1 = df_long[df_long["STATE/UT"] == state].groupby("YEAR")["Cases"].sum().reset_index()
    c_data1["State Name"] = state
    c_data2 = df_long[df_long["STATE/UT"] == state2].groupby("YEAR")["Cases"].sum().reset_index()
    c_data2["State Name"] = state2

    combined_compare = pd.concat([c_data1, c_data2])
    fig_comp = px.line(combined_compare, x="YEAR", y="Cases", color="State Name", markers=True,
                       color_discrete_map={state: "#00BFFF", state2: "#FF4B4B"})
    fig_comp.update_layout(template="plotly_dark", hovermode="x unified")
    st.plotly_chart(fig_comp, use_container_width=True)

    st.divider()

    # Correlation Heatmap Feature added inside Deep Dive
    st.subheader("🔗 Crime Type Correlation Heatmap")
    crime_cols = [c for c in df_wide.columns if c not in ["STATE/UT", "DISTRICT", "YEAR"]]
    corr_matrix = df_wide[df_wide["STATE/UT"] == state][crime_cols].corr()
    fig_corr = px.imshow(corr_matrix, text_auto=False, color_continuous_scale='RdBu_r', aspect="auto")
    st.plotly_chart(fig_corr, use_container_width=True)

    st.divider()

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("📈 Year-wise Crime Trend")
        year_data = filtered_data.groupby("YEAR")["Cases"].sum().reset_index()
        st.plotly_chart(px.line(year_data, x="YEAR", y="Cases", markers=True), use_container_width=True)
    with col_right:
        st.subheader("🏙️ Top Districts")
        district_data = filtered_data.groupby("DISTRICT")["Cases"].sum().sort_values(ascending=False).head(10).reset_index()
        st.plotly_chart(px.bar(district_data, x="DISTRICT", y="Cases", color="Cases"), use_container_width=True)

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("🔍 Crime Type Analysis")
        crime_data = filtered_data.groupby("Crime_Type")["Cases"].sum().sort_values(ascending=False).head(10).reset_index()
        st.plotly_chart(px.bar(crime_data, x="Crime_Type", y="Cases", color="Crime_Type"), use_container_width=True)
    with col_b:
        st.subheader("🧩 Crime Distribution")
        fig_pie = px.pie(crime_data.head(5), names="Crime_Type", values="Cases", hole=0.3)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()
    st.subheader("🎯 District Level Analysis")
    district_dd = st.selectbox("Select District for Deep Dive", sorted(filtered_data["DISTRICT"].unique()))
    district_crime = filtered_data[filtered_data["DISTRICT"] == district_dd].groupby("Crime_Type")["Cases"].sum().reset_index()
    st.plotly_chart(px.bar(district_crime.head(10), x="Crime_Type", y="Cases"), use_container_width=True)

# ===============================
# 🔮 TAB 4: PREDICTION (Restored)
# ===============================
with tab4:
    st.header("🔮 Crime Forecasting (Target: 2030)")
    future_years = np.array(range(2013, 2031)).reshape(-1, 1)

    # 1. NATIONAL PREDICTION
    st.subheader("🇮🇳 National Crime Projection")
    yearly_data = df_long.groupby("YEAR")["Cases"].sum().reset_index()
    Xn = yearly_data["YEAR"].values.reshape(-1, 1)
    yn = yearly_data["Cases"].values
    model_n = LinearRegression().fit(Xn, yn)
    preds_n = model_n.predict(future_years)

    act_n = pd.DataFrame({"Year": yearly_data["YEAR"], "Cases": yn, "Type": "Actual"})
    pr_n = pd.DataFrame({"Year": np.insert(future_years.flatten(), 0, 2012), 
                         "Cases": np.insert(preds_n, 0, yn[-1]), "Type": "Predicted"})
    
    fig_n = px.line(pd.concat([act_n, pr_n]), x="Year", y="Cases", color="Type",
                    color_discrete_map={"Actual": "#00BFFF", "Predicted": "#FF4B4B"})
    fig_n.update_traces(line=dict(dash='dash', width=3), selector=dict(name='Predicted'))
    st.plotly_chart(fig_n, use_container_width=True)

    st.divider()

    # 2. STATE & DISTRICT SIDE-BY-SIDE
    col_s, col_d = st.columns(2)
    with col_s:
        st.subheader(f"📍 {state} Prediction")
        state_yearly = filtered_data.groupby("YEAR")["Cases"].sum().reset_index()
        Xs = state_yearly["YEAR"].values.reshape(-1, 1)
        ys = state_yearly["Cases"].values
        model_s = LinearRegression().fit(Xs, ys)
        preds_s = model_s.predict(future_years)
        act_s = pd.DataFrame({"Year": state_yearly["YEAR"], "Cases": ys, "Type": "Actual"})
        pr_s = pd.DataFrame({"Year": np.insert(future_years.flatten(), 0, 2012), 
                             "Cases": np.insert(preds_s, 0, ys[-1]), "Type": "Predicted"})
        st.plotly_chart(px.line(pd.concat([act_s, pr_s]), x="Year", y="Cases", color="Type"), use_container_width=True)

    with col_d:
        dist_p = st.selectbox("Select District for Prediction", sorted(filtered_data["DISTRICT"].unique()))
        st.subheader(f"🏘️ {dist_p} Projection")
        dist_yearly = filtered_data[filtered_data["DISTRICT"] == dist_p].groupby("YEAR")["Cases"].sum().reset_index()
        if not dist_yearly.empty:
            Xd = dist_yearly["YEAR"].values.reshape(-1, 1)
            yd = dist_yearly["Cases"].values
            model_d = LinearRegression().fit(Xd, yd)
            preds_d = model_d.predict(future_years)
            act_d = pd.DataFrame({"Year": dist_yearly["YEAR"], "Cases": yd, "Type": "Actual"})
            pr_d = pd.DataFrame({"Year": np.insert(future_years.flatten(), 0, 2012), 
                                 "Cases": np.insert(preds_d, 0, yd[-1]), "Type": "Predicted"})
            st.plotly_chart(px.line(pd.concat([act_d, pr_d]), x="Year", y="Cases", color="Type"), use_container_width=True)

# ===============================
# 🚨 TAB 5: ANOMALIES (New Feature)
# ===============================
with tab5:
    st.subheader("🚨 Statistical Anomaly Detection")
    st.write("Detecting years where crime volume was significantly higher/lower than the statistical norm.")
    
    anom_data = filtered_data.groupby("YEAR")["Cases"].sum().reset_index()
    # Z-Score Calculation
    mean_v = anom_data["Cases"].mean()
    std_v = anom_data["Cases"].std()
    anom_data['Z-Score'] = (anom_data['Cases'] - mean_v) / std_v
    anom_data['Is_Anomaly'] = anom_data['Z-Score'].abs() > 1.5 # 1.5 Std Dev Threshold

    fig_anom = go.Figure()
    fig_anom.add_trace(go.Scatter(x=anom_data['YEAR'], y=anom_data['Cases'], name="Normal Trend", mode='lines+markers'))
    
    anomalies = anom_data[anom_data['Is_Anomaly']]
    fig_anom.add_trace(go.Scatter(x=anomalies['YEAR'], y=anomalies['Cases'], mode='markers', 
                                  marker=dict(color='red', size=15, symbol='star'), name="Anomaly Detected"))
    
    fig_anom.update_layout(template="plotly_dark", title="Unusual Crime Patterns Identified")
    st.plotly_chart(fig_anom, use_container_width=True)
    
    if not anomalies.empty:
        st.error(f"⚠️ Red Alert: Irregular activity detected in: {list(anomalies['YEAR'])}")
    else:
        st.success("✅ Statistical trend appears stable for this state.")

# ===============================
# 📊 TAB 6: RECENT DATA (Restored)
# ===============================
with tab6:
    st.subheader("📊 Recent Crime Snapshot (2020-2022)")
    try:
        recent_df = pd.read_csv("recent_crime_data.csv")
        recent_df = recent_df[~recent_df["State/UT"].str.contains("Total", na=False)]
        recent_long = recent_df.melt(id_vars=["State/UT"], var_name="YEAR", value_name="Cases")
        recent_long["YEAR"] = pd.to_numeric(recent_long["YEAR"], errors="coerce")
        recent_long["Cases"] = pd.to_numeric(recent_long["Cases"], errors="coerce")
        recent_yr = recent_long.groupby("YEAR")["Cases"].sum().reset_index()
        st.plotly_chart(px.line(recent_yr, x="YEAR", y="Cases", markers=True, color_discrete_sequence=['#00FFCC']), use_container_width=True)
    except:
        st.info("Upload 'recent_crime_data.csv' for post-2012 trends.")