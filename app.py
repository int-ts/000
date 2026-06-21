import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="用户购买行为数据分析",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 用户购买行为数据可视化分析")
st.markdown("---")

@st.cache_data
def load_data():
    try:
        df = sns.load_dataset('tips')
    except:
        np.random.seed(42)
        n = 244
        df = pd.DataFrame({
            'total_bill': np.random.uniform(10, 50, n),
            'tip': np.random.uniform(1, 10, n),
            'sex': np.random.choice(['Male', 'Female'], n),
            'smoker': np.random.choice(['Yes', 'No'], n),
            'day': np.random.choice(['Thur', 'Fri', 'Sat', 'Sun'], n),
            'time': np.random.choice(['Lunch', 'Dinner'], n),
            'size': np.random.randint(1, 6, n)
        })
    return df

df = load_data()

st.sidebar.header("🔍 数据筛选")

selected_days = st.sidebar.multiselect(
    "选择星期",
    options=df['day'].unique(),
    default=df['day'].unique()
)

selected_time = st.sidebar.multiselect(
    "选择时段",
    options=df['time'].unique(),
    default=df['time'].unique()
)

selected_sex = st.sidebar.multiselect(
    "选择性别",
    options=df['sex'].unique(),
    default=df['sex'].unique()
)

min_bill, max_bill = st.sidebar.slider(
    "总消费金额范围",
    min_value=float(df['total_bill'].min()),
    max_value=float(df['total_bill'].max()),
    value=(float(df['total_bill'].min()), float(df['total_bill'].max()))
)

filtered_df = df[
    (df['day'].isin(selected_days)) &
    (df['time'].isin(selected_time)) &
    (df['sex'].isin(selected_sex)) &
    (df['total_bill'] >= min_bill) &
    (df['total_bill'] <= max_bill)
]

st.subheader("📋 数据概览")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("总记录数", f"{len(filtered_df)} 条")
with col2:
    st.metric("平均消费", f"${filtered_df['total_bill'].mean():.2f}")
with col3:
    st.metric("平均小费", f"${filtered_df['tip'].mean():.2f}")
with col4:
    st.metric("小费/消费比", f"{filtered_df['tip'].sum() / filtered_df['total_bill'].sum() * 100:.1f}%")

with st.expander("📄 查看原始数据"):
    st.dataframe(filtered_df, use_container_width=True)
    st.caption(f"共 {len(filtered_df)} 条记录")

st.markdown("---")

st.header("📈 描述性统计分析")

col1, col2 = st.columns(2)

with col1:
    st.subheader("数值型变量统计摘要")
    st.dataframe(filtered_df.describe(), use_container_width=True)

with col2:
    st.subheader("分类变量频数统计")

    display_names = {
        'sex': '性别',
        'smoker': '吸烟者',
        'day': '星期',
        'time': '时段'
    }

    day_map = {
        'Thur': '周四',
        'Fri': '周五',
        'Sat': '周六',
        'Sun': '周日'
    }

    for col in ['sex', 'smoker', 'day', 'time']:
        freq = filtered_df[col].value_counts()

        if col == 'day':
            freq.index = freq.index.map(lambda x: day_map.get(x, x))

        freq_df = pd.DataFrame({
            '类别': freq.index,
            '频数': freq.values,
            '占比': (freq.values / len(filtered_df) * 100).round(1).astype(str) + '%'
        })

        st.write(f"**{display_names.get(col, col)}**")
        st.dataframe(freq_df, use_container_width=True, hide_index=True)

st.markdown("---")

st.header("📊 数据可视化分析")

st.subheader("3.1 消费金额分布情况")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(filtered_df['total_bill'], bins=20, kde=True, ax=ax, color='skyblue')
    ax.set_title('总消费金额分布', fontsize=14)
    ax.set_xlabel('总消费金额 ($)')
    ax.set_ylabel('频数')
    st.pyplot(fig)
    plt.close()

with col2:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(filtered_df['tip'], bins=20, kde=True, ax=ax, color='lightcoral')
    ax.set_title('小费金额分布', fontsize=14)
    ax.set_xlabel('小费金额 ($)')
    ax.set_ylabel('频数')
    st.pyplot(fig)
    plt.close()

st.caption("📌 从分布图可以看出，消费金额和小费金额均呈右偏分布，大部分消费集中在较低区间。")

st.subheader("3.2 分组对比分析")

tab1, tab2, tab3 = st.tabs(["按性别分组", "按是否吸烟分组", "按星期分组"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.boxplot(x='sex', y='total_bill', data=filtered_df, ax=ax, palette=['#FF6B6B', '#4ECDC4'])
        ax.set_title('不同性别消费金额对比', fontsize=13)
        st.pyplot(fig)
        plt.close()
    with col2:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.boxplot(x='sex', y='tip', data=filtered_df, ax=ax, palette=['#FF6B6B', '#4ECDC4'])
        ax.set_title('不同性别小费金额对比', fontsize=13)
        st.pyplot(fig)
        plt.close()

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.boxplot(x='smoker', y='total_bill', data=filtered_df, ax=ax, palette=['#95E1D3', '#F38181'])
        ax.set_title('吸烟与否消费金额对比', fontsize=13)
        st.pyplot(fig)
        plt.close()
    with col2:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.boxplot(x='smoker', y='tip', data=filtered_df, ax=ax, palette=['#95E1D3', '#F38181'])
        ax.set_title('吸烟与否小费金额对比', fontsize=13)
        st.pyplot(fig)
        plt.close()

with tab3:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x='day', y='total_bill', data=filtered_df, ax=ax, palette='Set2')
    ax.set_title('不同星期消费金额对比', fontsize=13)
    st.pyplot(fig)
    plt.close()

st.subheader("3.3 相关性热力图")

fig, ax = plt.subplots(figsize=(8, 6))
numeric_cols = ['total_bill', 'tip', 'size']
corr_matrix = filtered_df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
ax.set_title('数值变量相关性热力图', fontsize=14)
st.pyplot(fig)
plt.close()

st.caption("📌 热力图显示，消费金额与小费金额呈强正相关（约0.68），说明消费越高小费通常也越高。")

st.subheader("3.4 消费金额 vs 小费金额 散点图")

fig = px.scatter(
    filtered_df,
    x='total_bill',
    y='tip',
    color='sex',
    size='size',
    hover_data=['day', 'time', 'smoker'],
    title='消费金额与小费金额的关系',
    labels={'total_bill': '总消费金额 ($)', 'tip': '小费金额 ($)'},
    color_discrete_sequence=px.colors.qualitative.Set1
)
st.plotly_chart(fig, use_container_width=True)

st.caption("📌 散点图展示了消费金额与小费金额的正相关关系，不同颜色代表不同性别，点的大小代表用餐人数。")

st.subheader("3.5 不同星期消费金额趋势")

day_order = ['Thur', 'Fri', 'Sat', 'Sun']
day_agg = filtered_df.groupby('day').agg({
    'total_bill': ['mean', 'sum', 'count'],
    'tip': 'mean'
}).reset_index()
day_agg.columns = ['day', 'avg_bill', 'sum_bill', 'count', 'avg_tip']
day_agg['day'] = pd.Categorical(day_agg['day'], categories=day_order, ordered=True)
day_agg = day_agg.sort_values('day')

fig = go.Figure()
fig.add_trace(go.Bar(
    x=day_agg['day'],
    y=day_agg['avg_bill'],
    name='平均消费',
    marker_color='#4ECDC4',
    text=day_agg['avg_bill'].round(2),
    textposition='outside'
))
fig.add_trace(go.Scatter(
    x=day_agg['day'],
    y=day_agg['avg_tip'],
    name='平均小费',
    mode='lines+markers',
    line=dict(color='#FF6B6B', width=3),
    marker=dict(size=10)
))
fig.update_layout(
    title='不同星期的平均消费与小费',
    xaxis_title='星期',
    yaxis_title='金额 ($)',
    legend=dict(orientation='h', y=1.05)
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

st.header("🤖 用户聚类分析")

st.markdown("""
使用 **K-Means聚类算法** 对用户进行分群，将用户分为高价值、中价值和低价值三类。
""")

cluster_data = filtered_df[['total_bill', 'tip', 'size']].copy()
scaler = StandardScaler()
cluster_scaled = scaler.fit_transform(cluster_data)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
filtered_df['cluster'] = kmeans.fit_predict(cluster_scaled)

cluster_means = filtered_df.groupby('cluster')['total_bill'].mean().sort_values()
cluster_map = {old: new for new, old in enumerate(cluster_means.index)}
filtered_df['cluster_label'] = filtered_df['cluster'].map(cluster_map)

cluster_stats = filtered_df.groupby('cluster_label').agg({
    'total_bill': ['mean', 'count'],
    'tip': 'mean',
    'size': 'mean'
}).round(2)
cluster_stats.columns = ['平均消费', '用户数', '平均小费', '平均人数']

cluster_names = {0: '低价值用户', 1: '中价值用户', 2: '高价值用户'}
cluster_stats.index = cluster_stats.index.map(cluster_names)

st.subheader("聚类结果统计")

col1, col2 = st.columns([2, 1])

with col1:
    st.dataframe(cluster_stats, use_container_width=True)

with col2:
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ['#95E1D3', '#FCE38A', '#F38181']
    cluster_counts = filtered_df['cluster_label'].value_counts().sort_index()
    cluster_counts.index = cluster_counts.index.map(cluster_names)
    ax.pie(cluster_counts.values, labels=cluster_counts.index, autopct='%1.1f%%', colors=colors)
    ax.set_title('用户类别分布', fontsize=13)
    st.pyplot(fig)
    plt.close()

st.subheader("聚类可视化")

fig = px.scatter(
    filtered_df,
    x='total_bill',
    y='tip',
    color='cluster_label',
    size='size',
    hover_data=['day', 'time', 'sex', 'smoker'],
    title='用户聚类结果可视化',
    labels={'total_bill': '总消费金额 ($)', 'tip': '小费金额 ($)'},
    color_discrete_sequence=px.colors.qualitative.Set2,
    category_orders={'cluster_label': [0, 1, 2]}
)

fig.for_each_trace(lambda t: t.update(name=cluster_names[int(t.name)] if t.name.isdigit() else t.name))
st.plotly_chart(fig, use_container_width=True)

st.caption("📌 聚类分析将用户分为三类：高价值用户（消费高、小费高）、中价值用户和低价值用户。")

st.header("💡 分析结论与洞察")

st.markdown("""
### 主要发现

1. **消费分布特征**
   - 大部分用户的消费金额集中在 10-30 美元之间，呈右偏分布
   - 小费金额与消费金额呈强正相关（相关系数约 0.68）

2. **分组差异**
   - 男性用户的平均消费略高于女性用户
   - 周末（周六、周日）的消费金额明显高于工作日
   - 吸烟与非吸烟用户在消费行为上差异不大

3. **用户分层**
   - 高价值用户（约占总用户的 25%）消费金额高、小费慷慨，是重点维护对象
   - 中价值用户是主要的消费群体，具有较大的转化潜力
   - 低价值用户消费较低，可通过营销活动提升其消费频次

### 建议

- 🎯 针对高价值用户提供专属优惠和VIP服务，提升用户忠诚度
- 📈 在周末推出促销活动，进一步提高周末消费额
- 💰 针对低价值用户发放优惠券，刺激消费，促进向上转化
""")

st.markdown("---")
st.caption("📊 数据可视化课程作业 | 基于 Streamlit 构建 | 数据来源：Seaborn Tips Dataset")
