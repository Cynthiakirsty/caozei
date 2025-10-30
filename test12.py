import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="每日指标 + 盈余率智能预测", layout="wide")
st.title("曹贼版每日指标分析与盈余率预测模型")

# =========================
# 多文件上传 + 合并
# =========================
uploaded_files = st.file_uploader(
    "上传每日指标文件（可多选 Excel/CSV）", 
    type=["xlsx", "xls", "csv"], 
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("👆 请上传一个或多个包含历史每日指标的文件。")
    st.stop()

dfs = []
for uploaded_file in uploaded_files:
    fname = uploaded_file.name.lower()
    try:
        if fname.endswith(".csv"):
            df_temp = pd.read_csv(uploaded_file)
        else:
            df_temp = pd.read_excel(uploaded_file)
        
        # 清理列名
        df_temp.columns = df_temp.columns.str.strip().str.replace("\n", "").str.replace(" ", "")
        
        # 将每个文件的数据添加到dfs列表中
        dfs.append(df_temp)
    except Exception as e:
        st.error(f"读取文件 {uploaded_file.name} 失败：{e}")
        st.stop()

# 合并所有数据并去重列
df = pd.concat(dfs, axis=0, ignore_index=True)
df = df.loc[:, ~df.columns.duplicated()]  # 去重列名
df.columns = df.columns.str.strip().str.replace("\n", "").str.replace(" ", "")  # 再次清理列名

st.success("✅ 文件上传成功并已合并！")
st.subheader("📊 数据预览")
st.dataframe(df.head(10))

# =========================
# 基础字段校验与清洗
# =========================
if "日期" not in df.columns:
    st.error("❌ 文件必须包含【日期】列。")
    st.stop()

try:
    df["日期"] = pd.to_datetime(df["日期"])
except Exception:
    st.error("日期格式无法解析，请确保【日期】列为可识别格式（例如 YYYY-MM-DD）。")
    st.stop()

# 打印列名查看
st.write(f"数据的列名：{df.columns.tolist()}")

# 检查“新增人数”，“首充人数”，“首充次留”列是否存在
required_cols = ["新增人数", "首充人数", "首充次留"]

missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    st.warning(f"⚠️ 数据中缺少以下列：{missing_cols}，无法绘制漏斗图。")
else:
    # 计算次日登录人数
    df["次日登录人数"] = (df["首充人数"] * df["首充次留"]).round().astype(int)


    # 日期选择
    min_date, max_date = df["日期"].min(), df["日期"].max()
    date_range = st.date_input(
        "选择日期范围（默认展示全部）",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # 筛选
    mask = (df["日期"] >= pd.to_datetime(date_range[0])) & (df["日期"] <= pd.to_datetime(date_range[1]))
    df_filtered = df.loc[mask]

    # 汇总
    funnel_data = pd.DataFrame({
        "阶段": ["新增人数", "首充人数", "次日登录人数"],
        "人数": [
            df_filtered["新增人数"].sum(),
            df_filtered["首充人数"].sum(),
            df_filtered["次日登录人数"].sum()
        ]
    })

    # 转化率标注
    add_rate = df_filtered["首充人数"].sum() / df_filtered["新增人数"].sum() if df_filtered["新增人数"].sum() > 0 else 0
    next_rate = df_filtered["次日登录人数"].sum() / df_filtered["首充人数"].sum() if df_filtered["首充人数"].sum() > 0 else 0

    st.markdown(f"""
    **转化率分析：**
    - 新增 → 首充：{add_rate:.2%}  
    - 首充 → 次日登录：{next_rate:.2%}
    """)

    # 漏斗图
    fig_funnel = px.funnel(
        funnel_data,
        x="人数",
        y="阶段",
        title=f"用户转化漏斗图（{date_range[0]} 至 {date_range[1]}）"
    )
    st.plotly_chart(fig_funnel, use_container_width=True)

# =========================
# 原有趋势图（保留 + 扩展）
# =========================
st.header("📈 指标趋势图")

# 盈余率趋势
if "盈余率" in df.columns:
    fig2 = px.line(df, x="日期", y="盈余率", markers=True, title="盈余率趋势")
    st.plotly_chart(fig2, use_container_width=True)

# 充值 vs 提现趋势
if all(col in df.columns for col in ["充值金额", "提现金额"]):
    fig3 = px.line(df, x="日期", y=["充值金额", "提现金额"], markers=True, title="充值金额 & 提现金额趋势")
    st.plotly_chart(fig3, use_container_width=True)

# 其他字段趋势图（新增）
numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ["盈余率"]]
sel_fields = st.multiselect(
    "选择要额外展示趋势的字段（可多选）",
    numeric_cols,
    default=[c for c in ["充值人数", "提现人数", "日活跃玩家数", "新增用户数"] if c in numeric_cols],
)
for col in sel_fields:
    fig = px.line(df, x="日期", y=col, markers=True, title=f"{col} 趋势")
    st.plotly_chart(fig, use_container_width=True)

# =========================
# 盈余率预测模块（保留原逻辑）
# =========================
st.header("🔮 盈余率预测模型")

all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
features_candidates = [c for c in all_numeric if c not in ["充值金额", "提现金额", "盈余率"]]

selected_features = st.multiselect(
    "选择用于预测充值/提现的特征", features_candidates,
    default=[c for c in ["日活跃玩家数", "新增用户数", "整体ARPPU"] if c in features_candidates]
)

if len(selected_features) < 1:
    st.info("请选择至少一个特征以继续预测。")
    st.stop()

col_a, col_b, col_c = st.columns(3)
with col_a:
    future_days = st.number_input("预测未来天数", min_value=1, max_value=90, value=14)
with col_b:
    model_choice = st.selectbox("选择模型", ["LinearRegression", "RandomForestRegressor"])
with col_c:
    use_scenario = st.checkbox("使用情景增长率", value=True)

if use_scenario:
    col1, col2, col3 = st.columns(3)
    with col1:
        rech_daily_pct = st.number_input("充值每日增长率 %", value=1.0, step=0.1)
    with col2:
        wd_daily_pct = st.number_input("提现每日增长率 %", value=0.5, step=0.1)
    with col3:
        add_noise_pct = st.slider("每日波动 ±%", 0.0, 10.0, 2.0, step=0.1)
else:
    col1, col2 = st.columns(2)
    with col1:
        noise_pct = st.slider("模型外推随机波动 ±%", 0.0, 10.0, 2.0, step=0.1)
    with col2:
        recent_window = st.number_input("趋势窗口天数", min_value=3, max_value=30, value=7)

# 训练数据准备
df_train = df.dropna(subset=selected_features + ["充值金额", "提现金额"]).copy()
X = df_train[selected_features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

if model_choice == "LinearRegression":
    model_rech = LinearRegression().fit(X_scaled, df_train["充值金额"])
    model_wd = LinearRegression().fit(X_scaled, df_train["提现金额"])
else:
    from sklearn.ensemble import RandomForestRegressor
    model_rech = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_scaled, df_train["充值金额"])
    model_wd = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_scaled, df_train["提现金额"])

# 生成未来日期并预测
last_date = df["日期"].max()
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days, freq="D")
future_rows = []

recharge_base = df_train["充值金额"].iloc[-1] if len(df_train) > 0 else 1.0
withdraw_base = df_train["提现金额"].iloc[-1] if len(df_train) > 0 else 0.0

for i, d in enumerate(future_dates, start=1):
    if use_scenario:
        pred_rech = recharge_base * ((1 + rech_daily_pct / 100) ** i)
        pred_wd = withdraw_base * ((1 + wd_daily_pct / 100) ** i)
        noise_r = np.random.normal(0, add_noise_pct / 100)
        noise_w = np.random.normal(0, add_noise_pct / 100)
        pred_rech = max(1.0, pred_rech * (1 + noise_r))
        pred_wd = max(0.0, pred_wd * (1 + noise_w))
    else:
        base_feat = df_train[selected_features].iloc[-1]
        daily_delta = (df_train[selected_features].iloc[-1] - df_train[selected_features].iloc[-recent_window]) / recent_window
        day_feat = base_feat + daily_delta * i
        noise = np.random.normal(0, noise_pct / 100, size=day_feat.shape[0])
        day_feat = day_feat * (1 + noise)
        day_feat_scaled = scaler.transform([day_feat])
        pred_rech = float(model_rech.predict(day_feat_scaled)[0])
        pred_wd = float(model_wd.predict(day_feat_scaled)[0])
        if pred_rech <= 0: pred_rech = recharge_base

    pred_rate = (pred_rech - pred_wd) / pred_rech if pred_rech != 0 else 0
    future_rows.append({"日期": d, "预测充值金额": pred_rech, "预测提现金额": pred_wd, "预测盈余率": pred_rate})

future_df = pd.DataFrame(future_rows)

# 展示结果
st.subheader("📅 未来每日预测结果")
st.dataframe(future_df)

# 绘制历史+预测盈余率
hist = df[["日期", "盈余率"]].rename(columns={"盈余率": "实际盈余率"}).set_index("日期")
fut = future_df.set_index("日期")[["预测盈余率"]]
comb = pd.concat([hist, fut], axis=0).reset_index()
fig_pred = px.line(comb, x="日期", y=["实际盈余率", "预测盈余率"], title="实际 vs 预测盈余率", markers=True)
st.plotly_chart(fig_pred, use_container_width=True)

# 下载
output = BytesIO()
with pd.ExcelWriter(output, engine="openpyxl") as writer:
    future_df.to_excel(writer, index=False, sheet_name="预测结果")
st.download_button(
    "💾 下载预测结果（Excel）",
    data=output.getvalue(),
    file_name="future_profit_predictions.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

# 相关性展示
st.header("📌 影响盈余率的关键特征")
if "盈余率" in df.columns:
    corr_with_target = df[selected_features + ["盈余率"]].corr()["盈余率"].drop("盈余率").sort_values(ascending=False)
    st.dataframe(corr_with_target.to_frame("相关系数"))

try:
    coef_df = pd.DataFrame({
        "特征": selected_features,
        "充值模型系数": getattr(model_rech, "coef_", [np.nan]*len(selected_features)),
        "提现模型系数": getattr(model_wd, "coef_", [np.nan]*len(selected_features)),
    }).set_index("特征")
    st.subheader("模型系数（仅线性模型时有效）")
    st.dataframe(coef_df)
except Exception:
    pass

# ===== 补充说明 =====
st.markdown("""
---
🔎 **说明小结：**

你可选择「情景法」（手动设定增长率）或「模型法」（基于所选特征由回归器预测充值/提现并外推）。  
若需要更稳健的非线性预测（推荐），可切换到 **RandomForestRegressor** 并调参；  
同时可加入「发放占比」与「RTP」等因素，它们会显著影响盈余率变化。
""")