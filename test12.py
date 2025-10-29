import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from io import BytesIO

# 页面配置
st.set_page_config(page_title="每日指标 + 盈余率智能预测（保留原图表）", layout="wide")
st.title("曹贼版每日指标分析与按充提预测的盈余率模型")

# =========================
# 上传与读取（保留原样）
# =========================
uploaded_file = st.file_uploader("上传每日指标文件（Excel/CSV）", type=["xlsx", "xls", "csv"])

if not uploaded_file:
    st.info("👆 请先上传包含历史每日指标的文件（至少包含：日期、充值金额、提现金额）。")
    st.stop()

try:
    fname = uploaded_file.name.lower()
    if fname.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"读取文件失败：{e}")
    st.stop()

# 清理与标准化（保留原样）
df.columns = df.columns.str.strip()
st.success("✅ 数据读取成功！")
st.subheader("📅 数据预览")
st.dataframe(df.head(10))

# 必要列检查
expected_cols = ["日期", "新增人数", "日活跃人数", "盈余率", "充值人数", "充值金额", "提现人数", "提现金额", "首充次留", "首充3留", "新ARPPU", "老ARPPU"]
missing = [c for c in expected_cols if c not in df.columns]
if missing:
    st.warning(f"⚠️ 注意：文件中缺少以下列（部分功能受限）：{missing}")

# 格式化日期与数值（保留原样）
if "日期" in df.columns:
    try:
        df["日期"] = pd.to_datetime(df["日期"])
    except Exception:
        st.error("日期格式无法解析，请确保【日期】列为可识别格式（例如 YYYY-MM-DD）。")
        st.stop()
else:
    st.error("文件必须包含【日期】列。")
    st.stop()

# 计算或标准化部分列（保留原样）
if "充值金额" in df.columns and "充值人数" in df.columns:
    df["整体ARPPU"] = df["充值金额"] / df["充值人数"].replace(0, np.nan)
else:
    df["整体ARPPU"] = np.nan

# 如果盈余率没有但有充提则计算
if "盈余率" not in df.columns and "充值金额" in df.columns and "提现金额" in df.columns:
    df["盈余率"] = (df["充值金额"] - df["提现金额"]) / df["充值金额"]
    df["盈余率"] = df["盈余率"].replace([np.inf, -np.inf], np.nan).fillna(0)

# =========================
# 原有图表（保持并展示）
# =========================
st.header("📈 原有指标趋势")

# 新增 & 日活
if all(col in df.columns for col in ["新增人数", "日活跃人数"]):
    fig1 = px.line(df, x="日期", y=["新增人数", "日活跃人数"], markers=True, title="新增人数 & 日活趋势")
    st.plotly_chart(fig1, use_container_width=True)

# 盈余率趋势
if "盈余率" in df.columns:
    fig2 = px.line(df, x="日期", y="盈余率", markers=True, title="盈余率趋势")
    st.plotly_chart(fig2, use_container_width=True)

# 充值 vs 提现
if all(col in df.columns for col in ["充值金额", "提现金额"]):
    fig3 = px.line(df, x="日期", y=["充值金额", "提现金额"], markers=True, title="充值金额 & 提现金额趋势")
    st.plotly_chart(fig3, use_container_width=True)

# ARPPU 图
arppu_cols = [c for c in ["新ARPPU", "老ARPPU", "整体ARPPU"] if c in df.columns]
if arppu_cols:
    fig4 = px.line(df, x="日期", y=arppu_cols, markers=True, title="ARPPU 趋势对比")
    st.plotly_chart(fig4, use_container_width=True)

# 相关性热力图与特征重要性（保留原样）
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# 如果包含目标列则显示热力图
if "盈余率" in df.columns and len(numeric_cols) >= 2:
    st.subheader("🔥 特征相关性热力图")
    corr = df[numeric_cols].corr()
    heatmap = ff.create_annotated_heatmap(
        z=np.round(corr.values, 2),
        x=list(corr.columns),
        y=list(corr.index),
        annotation_text=np.round(corr.values, 2),
        showscale=True,
        colorscale="RdBu",
        reversescale=True
    )
    heatmap.update_layout(title="相关性矩阵（数值列）", width=900, height=700)
    st.plotly_chart(heatmap, use_container_width=True)

# 如果你之前有特征重要性展示，这里也保留类似展示（当模型可建时）
# =========================
# 新增模块：按充值/提现分别建模，再按公式计算未来每日盈余率（**仅新增**）
# =========================
st.header("🔮 按充值/提现预测未来每日盈余率")
st.markdown("说明：先分别预测未来每日的 **充值金额** 与 **提现金额**，再按公式 `(充值-提现)/充值` 计算每日盈余率。你可以选择回归模型或情景增长率。")

# 选择用于预测的特征（排除充值/提现自身）
all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
features_candidates = [c for c in all_numeric if c not in ["充值金额", "提现金额", "盈余率"]]
selected_features = st.multiselect("选择用于预测充值/提现的特征（至少1个）", features_candidates,
                                   default=[c for c in ["日活跃人数", "新增人数", "整体ARPPU"] if c in features_candidates])

if len(selected_features) < 1:
    st.info("请选择至少一个特征以继续预测。")
    st.stop()

# 预测参数：天数、模型类型、情景开关
col_a, col_b, col_c = st.columns([2,2,2])
with col_a:
    future_days = st.number_input("预测未来天数 (天)", min_value=1, max_value=90, value=14)
with col_b:
    model_choice = st.selectbox("回归模型（用于训练充值/提现）", ["LinearRegression", "RandomForest（更鲁棒）"])
with col_c:
    use_scenario = st.checkbox("使用情景（手动设定每日增长率），否则使用模型预测 + 可选外推", value=True)

# 情景参数（仅当 use_scenario True）
if use_scenario:
    st.markdown("情景设置：每日增长率为复利（可为负数），也可选择是否加入随机波动。")
    col1, col2, col3 = st.columns(3)
    with col1:
        rech_daily_pct = st.number_input("情景：充值每日增长率 %", value=1.0, step=0.1)
    with col2:
        wd_daily_pct = st.number_input("情景：提现每日增长率 %", value=0.5, step=0.1)
    with col3:
        add_noise_pct = st.slider("每日随机波动幅度 ±% (用于情景模拟)", 0.0, 10.0, 2.0, step=0.1)
else:
    st.markdown("不使用情景：将基于模型预测并用最近趋势外推（可带小幅随机扰动）。")
    coln1, coln2 = st.columns(2)
    with coln1:
        noise_pct = st.slider("模型外推每日随机波动 ±%", 0.0, 10.0, 2.0, step=0.1)
    with coln2:
        recent_window = st.number_input("用于计算最近趋势的窗口天数", min_value=3, max_value=30, value=7)

# 训练样本检查
df_train = df.dropna(subset=selected_features + ["充值金额", "提现金额"]).copy()
if df_train.shape[0] < max(10, len(selected_features)*5):
    st.warning("历史样本较少，模型预测可能不稳定（建议增加更多历史日期样本）。")

# 训练模型（两个模型：充值、提现）
X = df_train[selected_features].values
scaler_amt = StandardScaler()
X_scaled = scaler_amt.fit_transform(X)

if model_choice == "LinearRegression":
    model_rech = LinearRegression()
    model_wd = LinearRegression()
    model_rech.fit(X_scaled, df_train["充值金额"].values)
    model_wd.fit(X_scaled, df_train["提现金额"].values)
else:
    # 随机森林（非必须，按需安装 sklearn）
    try:
        from sklearn.ensemble import RandomForestRegressor
        model_rech = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_scaled, df_train["充值金额"].values)
        model_wd = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_scaled, df_train["提现金额"].values)
    except Exception as e:
        st.error("RandomForest 未安装或出错，回退使用 LinearRegression。")
        model_rech = LinearRegression().fit(X_scaled, df_train["充值金额"].values)
        model_wd = LinearRegression().fit(X_scaled, df_train["提现金额"].values)

# 生成未来日期与基准特征外推（基于最近 N 天线性增量）
last_date = df["日期"].max()
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days, freq='D')

# 计算最近趋势基准
if not use_scenario:
    window = min(len(df_train), recent_window if 'recent_window' in locals() else 7)
    recent = df_train.tail(window).reset_index(drop=True)
    if len(recent) >= 2:
        # 日均增量（用线性差）
        daily_delta = (recent[selected_features].iloc[-1] - recent[selected_features].iloc[0]) / max(1, len(recent)-1)
    else:
        daily_delta = np.zeros(len(selected_features))
    base_feat = recent[selected_features].iloc[-1] if len(recent) > 0 else df_train[selected_features].iloc[-1]
else:
    # 情景也可以基于最后一天特征为基准（但我们用情景增长直接对充值/提现金额计算）
    if len(df_train) > 0:
        base_feat = df_train[selected_features].iloc[-1]
    else:
        st.error("历史数据不足以外推情景。")
        st.stop()

# 开始逐日预测（保持原图不变，仅新增结果区）
future_rows = []
recharge_base = df_train["充值金额"].iloc[-1] if len(df_train)>0 else 1.0
withdraw_base = df_train["提现金额"].iloc[-1] if len(df_train)>0 else 0.0

for i, d in enumerate(future_dates, start=1):
    if use_scenario:
        # 1) 用用户情景直接按增长率推充值与提现（复利），再加入随机波动
        pred_rech = recharge_base * ((1 + rech_daily_pct/100) ** i)
        pred_wd = withdraw_base * ((1 + wd_daily_pct/100) ** i)
        # 加噪声
        noise_r = np.random.normal(0, add_noise_pct/100)
        noise_w = np.random.normal(0, add_noise_pct/100)
        pred_rech = max(1.0, pred_rech * (1 + noise_r))
        pred_wd = max(0.0, pred_wd * (1 + noise_w))
    else:
        # 2) 用模型预测：先外推特征，然后模型预测充值/提现
        day_feat = base_feat + daily_delta * i
        # 可选小幅随机扰动
        noise = np.random.normal(0, noise_pct/100, size=day_feat.shape[0])
        day_feat = day_feat * (1 + noise)
        day_feat_df = pd.DataFrame([day_feat.values], columns=selected_features)
        day_feat_scaled = scaler_amt.transform(day_feat_df.values)
        pred_rech = float(model_rech.predict(day_feat_scaled)[0])
        pred_wd = float(model_wd.predict(day_feat_scaled)[0])
        # 防护
        if pred_rech <= 0:
            pred_rech = recharge_base * (1 + 0.001*i)
        pred_wd = max(0.0, pred_wd)

    # 计算盈余率（按公式）
    if pred_rech == 0:
        pred_rate = 0.0
    else:
        pred_rate = (pred_rech - pred_wd) / pred_rech
        pred_rate = float(np.clip(pred_rate, -1.0, 1.0))

    future_rows.append({
        "日期": d,
        "预测充值金额": pred_rech,
        "预测提现金额": pred_wd,
        "预测盈余率": pred_rate
    })

future_df = pd.DataFrame(future_rows)

# 显示预测结果（新增，不删除原图）
st.subheader("📅 未来逐日预测结果")
st.dataframe(future_df)

# 绘制历史实际盈余率与预测盈余率（新增）
display_hist = df[["日期", "盈余率"]].rename(columns={"盈余率": "实际盈余率"}).set_index("日期")
display_future = future_df.set_index("日期")[["预测盈余率"]]
display_combined = pd.concat([display_hist, display_future], axis=0).reset_index()
fig_pred = px.line(display_combined, x="日期", y=["实际盈余率", "预测盈余率"],
                   title="历史实际盈余率 vs 未来预测盈余率", markers=True)
st.plotly_chart(fig_pred, use_container_width=True)

# 允许导出预测结果（新增）
output = BytesIO()
with pd.ExcelWriter(output, engine="openpyxl") as writer:
    future_df.to_excel(writer, index=False, sheet_name="预测结果")
st.download_button("💾 下载预测结果（Excel）", data=output.getvalue(), file_name="future_profit_predictions.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# =========================
# 保留原有的特征重要性展示（若有模型）
# =========================
st.header("📌 影响盈余率的关键因素")
# 如果原先训练了模型（我们之前训练过 model），展示系数（注意：在新增模块我们训练了 model_rech/model_wd）
# 若 df 中存在 selected_features，则计算简单相关性与展示系数（如线性回归的系数不可直接对应充提影响）
if "盈余率" in df.columns and len(selected_features) > 0:
    corr_with_target = df[selected_features + ["盈余率"]].corr()["盈余率"].drop("盈余率").sort_values(ascending=False)
    st.write("与盈余率相关性（Pearson）:")
    st.dataframe(corr_with_target.to_frame("相关系数"))

# 如果你想看模型系数（充值/提现模型），我们也展示
try:
    rech_coefs = model_rech.coef_
    wd_coefs = model_wd.coef_
    coef_df = pd.DataFrame({
        "特征": selected_features,
        "充值模型系数": rech_coefs,
        "提现模型系数": wd_coefs
    }).set_index("特征")
    st.subheader("充值/提现模型系数（用于参考，需标准化后更可比）")
    st.dataframe(coef_df)
except Exception:
    pass

st.markdown("""
---
🔎 说明小结：
- 你可选择「情景法」（手动设定增长率）或「模型法」（基于所选特征由回归器预测充值/提现并外推）。  
- 若需要更稳健的非线性预测（推荐），可切换到 RandomForest 并调参；也可增加置信区间与多情景对比（我可以继续加）。  
""")
