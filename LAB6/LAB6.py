# ================================================================
# LAB6.py - การวิเคราะห์ Titanic Dataset
# Beautiful Night Sky Theme Dashboard
# คำสั่งสำหรับรัน: streamlit run LAB6.py
# ================================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
from plotly.subplots import make_subplots

# ================================================================
# กำหนด Custom CSS สำหรับธีมสีน้ำเงินท้องฟ้ายามค่ำคืน
# ================================================================
def apply_custom_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main app styling */
    .stApp {
        background: linear-gradient(135deg, #0c1445 0%, #1e3a8a 30%, #1e40af 60%, #3b82f6 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, rgba(30, 58, 138, 0.8), rgba(59, 130, 246, 0.8));
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(147, 197, 253, 0.3);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    .main-title {
        color: #f8fafc;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-subtitle {
        color: #cbd5e1;
        font-size: 1.2rem;
        font-weight: 300;
        margin-bottom: 1rem;
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(45deg, #1e40af, #3b82f6);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 2rem 0 1rem 0;
        font-weight: 600;
        font-size: 1.3rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
        border: 1px solid rgba(147, 197, 253, 0.3);
    }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(30, 64, 175, 0.8), rgba(59, 130, 246, 0.6));
        border: 1px solid rgba(147, 197, 253, 0.3);
        padding: 1rem;
        border-radius: 15px;
        color: white;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4);
    }
    
    [data-testid="metric-container"] > div > div > div > div {
        color: #f8fafc !important;
    }
    
    /* DataFrame styling */
    .stDataFrame {
        background: rgba(15, 23, 42, 0.8);
        border-radius: 15px;
        padding: 1rem;
        border: 1px solid rgba(147, 197, 253, 0.3);
        backdrop-filter: blur(10px);
    }
    
    /* Success/Info boxes */
    .stAlert > div {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.2), rgba(16, 185, 129, 0.2));
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 10px;
        color: #f0f9ff;
        backdrop-filter: blur(10px);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #0f172a, #1e293b);
        border-right: 1px solid rgba(147, 197, 253, 0.3);
    }
    
    /* Feature card */
    .feature-card {
        background: linear-gradient(135deg, rgba(30, 64, 175, 0.7), rgba(59, 130, 246, 0.5));
        border: 1px solid rgba(147, 197, 253, 0.3);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: white;
        text-align: center;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(59, 130, 246, 0.2);
    }
    
    .feature-rank {
        font-size: 2rem;
        font-weight: 700;
        color: #fbbf24;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .feature-name {
        font-size: 1.3rem;
        font-weight: 600;
        margin: 0.5rem 0;
    }
    
    .feature-distance {
        font-size: 1.1rem;
        color: #cbd5e1;
        font-family: 'Courier New', monospace;
    }
    
    /* Animation for loading */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.6s ease-out;
    }
    
    /* Stars background effect */
    .stars-bg {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: -1;
    }
    
    .star {
        position: absolute;
        background: white;
        border-radius: 50%;
        opacity: 0.8;
        animation: twinkle 2s infinite;
    }
    
    @keyframes twinkle {
        0%, 100% { opacity: 0.8; }
        50% { opacity: 0.3; }
    }
    </style>
    
    <!-- Stars background -->
    <div class="stars-bg">
        <div class="star" style="top: 20%; left: 10%; width: 2px; height: 2px; animation-delay: 0s;"></div>
        <div class="star" style="top: 30%; left: 80%; width: 1px; height: 1px; animation-delay: 1s;"></div>
        <div class="star" style="top: 60%; left: 20%; width: 2px; height: 2px; animation-delay: 2s;"></div>
        <div class="star" style="top: 80%; left: 70%; width: 1px; height: 1px; animation-delay: 0.5s;"></div>
        <div class="star" style="top: 15%; left: 60%; width: 1px; height: 1px; animation-delay: 1.5s;"></div>
        <div class="star" style="top: 70%; left: 90%; width: 2px; height: 2px; animation-delay: 0.8s;"></div>
    </div>
    """, unsafe_allow_html=True)

# ================================================================
# ตั้งค่า Streamlit
# ================================================================
st.set_page_config(
    page_title="🚢 Titanic Analysis Dashboard",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
apply_custom_css()

# ================================================================
# Header Section
# ================================================================
st.markdown("""
<div class="main-header fade-in">
    <div class="main-title">🚢 Titanic Dataset Analysis</div>
    <div class="main-subtitle">✨ Beautiful Night Sky Dashboard ✨</div>
    <div style="color: #94a3b8; font-size: 1rem;">
        การวิเคราะห์ข้อมูลผู้โดยสารไททานิค พร้อมการคำนวณ Distance ระหว่างคลาส
    </div>
</div>
""", unsafe_allow_html=True)

# ================================================================
# กำหนด path และโหลดข้อมูล
# ================================================================
csv_file = os.path.join(os.path.dirname(__file__), "Titanic-Dataset.csv")

if not os.path.exists(csv_file):
    st.error("🚨 ไม่พบไฟล์ Titanic-Dataset.csv กรุณาอัปโหลดไฟล์ไปที่โฟลเดอร์เดียวกับ LAB6.py")
    st.stop()

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(csv_file)
        return df
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการโหลดไฟล์: {e}")
        st.stop()

# โหลดข้อมูล
with st.spinner("🔄 กำลังโหลดข้อมูล..."):
    df = load_data()

# ================================================================
# เตรียมข้อมูล
# ================================================================
features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
df_clean = df[features + ["Survived"]].dropna()

# ข้อมูลทั่วไป
total_samples = len(df_clean)
num_features = len(features)
classes = df_clean["Survived"].nunique()
class_distribution = df_clean["Survived"].value_counts().sort_index().tolist()

# แยกข้อมูลตามคลาส
class0 = df_clean[df_clean["Survived"] == 0]
class1 = df_clean[df_clean["Survived"] == 1]
class0_samples = len(class0)
class1_samples = len(class1)

# คำนวณ mean และ variance
mean0 = class0[features].mean()
mean1 = class1[features].mean()
var0 = class0[features].var(ddof=1)
var1 = class1[features].var(ddof=1)

# คำนวณ Distance
results = {}
for feature in features:
    mu_i, mu_j = mean0[feature], mean1[feature]
    var_i, var_j = var0[feature], var1[feature]
    try:
        d = 0.5 * ((var_j / var_i) + (var_i / var_j) - 2) + 0.5 * ((mu_i - mu_j) ** 2) * ((1 / var_i) + (1 / var_j))
        results[feature] = d
    except ZeroDivisionError:
        st.warning(f"⚠️ ไม่สามารถคำนวณ Distance สำหรับฟีเจอร์ {feature}")
        results[feature] = 0.0

df_result = pd.DataFrame(list(results.items()), columns=["Feature", "Distance"])
df_result = df_result.sort_values(by="Distance", ascending=False)

# ================================================================
# Main Dashboard
# ================================================================

# Dataset Overview
st.markdown('<div class="section-header">📊 ข้อมูลทั่วไปของ Dataset</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        label="👥 จำนวนตัวอย่าง", 
        value=f"{total_samples:,}",
        help="จำนวนแถวข้อมูลที่ใช้ในการวิเคราะห์"
    )
with col2:
    st.metric(
        label="🎯 จำนวนฟีเจอร์", 
        value=num_features,
        help="จำนวนตัวแปรที่ใช้ในการวิเคราะห์"
    )
with col3:
    st.metric(
        label="🏷️ จำนวนคลาส", 
        value=classes,
        help="รอด (1) และไม่รอด (0)"
    )
with col4:
    st.metric(
        label="⚖️ การกระจายคลาส", 
        value=f"{class_distribution[0]}:{class_distribution[1]}",
        help="อัตราส่วนระหว่างไม่รอด:รอด"
    )

# Class Analysis
st.markdown('<div class="section-header">🎭 การวิเคราะห์ Class Survival</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    st.metric(
        label="💀 Class 0 (ไม่รอด)", 
        value=f"{class0_samples:,}",
        delta=f"{(class0_samples/total_samples*100):.1f}%"
    )
with col2:
    st.metric(
        label="🎉 Class 1 (รอด)", 
        value=f"{class1_samples:,}",
        delta=f"{(class1_samples/total_samples*100):.1f}%"
    )
with col3:
    # Survival Pie Chart
    fig_pie = go.Figure(data=[go.Pie(
        labels=['ไม่รอด 💀', 'รอด 🎉'],
        values=[class0_samples, class1_samples],
        hole=0.4,
        marker_colors=['#ef4444', '#22c55e'],
        textinfo='label+percent+value',
        textfont=dict(size=14, color='white')
    )])
    
    fig_pie.update_layout(
        title=dict(text="📈 อัตราการรอดชีวิต", font=dict(color='white', size=18)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=300,
        showlegend=False
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# Feature Distance Analysis
st.markdown('<div class="section-header">🔍 การวิเคราะห์ Distance ของฟีเจอร์</div>', unsafe_allow_html=True)

# Enhanced Distance Bar Chart
fig_bar = go.Figure()

# Add gradient bars
colors = ['#fbbf24', '#f59e0b', '#d97706', '#b45309', '#92400e']
for i, (feature, distance) in enumerate(zip(df_result['Feature'], df_result['Distance'])):
    fig_bar.add_trace(go.Bar(
        x=[feature],
        y=[distance],
        name=feature,
        marker_color=colors[i % len(colors)],
        text=f'{distance:.6f}',
        textposition='auto',
        textfont=dict(color='white', size=12, family='Courier New'),
        hovertemplate=f'<b>{feature}</b><br>Distance: {distance:.6f}<br><extra></extra>',
        showlegend=False
    ))

fig_bar.update_layout(
    title=dict(text="🎯 Distance Score ของแต่ละฟีเจอร์", font=dict(color='white', size=20)),
    xaxis=dict(
        title=dict(text="ฟีเจอร์", font=dict(color='white', size=14)),
        tickfont=dict(color='white', size=12),
        gridcolor='rgba(255,255,255,0.1)'
    ),
    yaxis=dict(
        title=dict(text="Distance Score", font=dict(color='white', size=14)),
        tickfont=dict(color='white', size=12),
        gridcolor='rgba(255,255,255,0.1)'
    ),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=500,
    title_x=0.5
)

st.plotly_chart(fig_bar, use_container_width=True)

# Results Table
st.markdown('<div class="section-header">📋 ตารางสรุปผลลัพธ์</div>', unsafe_allow_html=True)

# Format the dataframe for better display
df_display = df_result.copy()
df_display['Distance'] = df_display['Distance'].apply(lambda x: f"{x:.6f}")
df_display['Rank'] = range(1, len(df_display) + 1)
df_display = df_display[['Rank', 'Feature', 'Distance']]

st.dataframe(
    df_display,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Rank": st.column_config.NumberColumn(
            "🏆 อันดับ",
            width="small"
        ),
        "Feature": st.column_config.TextColumn(
            "🎯 ฟีเจอร์",
            width="medium"
        ),
        "Distance": st.column_config.TextColumn(
            "📊 Distance Score",
            width="medium"
        )
    }
)

# Best Feature Highlight
st.markdown('<div class="section-header">🏆 ฟีเจอร์ที่ดีที่สุด</div>', unsafe_allow_html=True)

best_feature = df_result.iloc[0]["Feature"]
best_distance = df_result.iloc[0]["Distance"]

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown(f"""
    <div class="feature-card">
        <div class="feature-rank">🥇 #1</div>
        <div class="feature-name">{best_feature}</div>
        <div class="feature-distance">Distance: {best_distance:.6f}</div>
        <div style="margin-top: 1rem; font-size: 0.9rem; color: #cbd5e1;">
            ฟีเจอร์นี้มีความสามารถในการแยกแยะระหว่างคลาสได้ดีที่สุด
        </div>
    </div>
    """, unsafe_allow_html=True)

# Top Features Ranking
st.markdown('<div class="section-header">🎖️ อันดับฟีเจอร์ทั้งหมด</div>', unsafe_allow_html=True)

cols = st.columns(len(df_result))
medal_emojis = ['🥇', '🥈', '🥉', '🏅', '🏅']

for idx, (col, (_, row)) in enumerate(zip(cols, df_result.iterrows())):
    with col:
        medal = medal_emojis[idx] if idx < len(medal_emojis) else '🏅'
        st.markdown(f"""
        <div style="
            text-align: center; 
            padding: 1rem; 
            background: linear-gradient(135deg, rgba(30, 64, 175, 0.6), rgba(59, 130, 246, 0.4));
            border-radius: 10px; 
            border: 1px solid rgba(147, 197, 253, 0.3);
            color: white;
            backdrop-filter: blur(10px);
        ">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">{medal}</div>
            <div style="font-weight: 600; margin-bottom: 0.5rem;">{row['Feature']}</div>
            <div style="font-size: 0.9rem; color: #cbd5e1; font-family: 'Courier New';">{row['Distance']:.6f}</div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; margin-top: 2rem;">
    <p>✨ สร้างด้วย Streamlit และความรักในการวิเคราะห์ข้อมูล ✨</p>
    <p>🚢 Titanic Dataset Analysis Dashboard | Night Sky Theme 🌙</p>
</div>
""", unsafe_allow_html=True)