# app.py
import streamlit as st
import pandas as pd
import numpy as np
from urllib.parse import urlparse

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

# --------------------- Page Config ---------------------
st.set_page_config(
    page_title="AI ค่าใช้จ่ายนักศึกษา (Decision Tree)",
    page_icon="💸",
    layout="wide"
)

# --------------------- Global Styles ---------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
html, body, [class*="css"]  { font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, 'Helvetica Neue', Arial; }
:root { --brand:#4f46e5; --brand-2:#a78bfa; --soft:#f5f7ff; --ok:#10b981; --warn:#f59e0b; --bad:#ef4444;}
.block-container { padding-top: 1.2rem !important; }
.hero { padding: 1.2rem 1.4rem; border-radius: 16px; background: linear-gradient(135deg, var(--soft), #ffffff);
        border: 1px solid #eaeefe; display:flex; gap:1rem; align-items:center;}
.hero .emoji { font-size: 1.8rem; }
.hero .title { font-size: 1.2rem; font-weight: 800; letter-spacing: .2px;}
.hero .sub { color:#5b6170; margin-top: 2px; }
.card { border: 1px solid #eaeefe; border-radius: 14px; padding: 1rem 1rem; background: #fff; }
.pill { display:inline-flex; align-items:center; gap:.4rem; padding:.3rem .6rem; border-radius:999px; font-size:.8rem;
        background:#eef2ff; color:#3730a3; border:1px solid #eaeefe; }
.param-pill{ background:#ecfeff; color:#155e75; border-color:#cffafe; }
.kpill{ background:#fef9c3; color:#854d0e; border-color:#fde68a; }
.best { background:#f0fdf4; border:1px solid #bbf7d0; color:#166534; border-radius: 12px; padding:.6rem .8rem; font-weight:600; }
hr.soft { border: none; border-top:1px solid #eef1ff; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# --------------------- Header ---------------------
colA, colB = st.columns([1,4])
with colA:
    st.markdown('<div class="hero"><div class="emoji">💸</div><div><div class="title">AI ผู้ช่วยค่าใช้จ่ายนักศึกษา</div><div class="sub">เว็บแอป Decision Tree • ใช้ข้อมูลจริงเท่านั้น</div></div></div>', unsafe_allow_html=True)
with colB:
    st.markdown("""
    <div class="card" style="display:flex;gap:.6rem;justify-content:flex-end;align-items:center;">
      <span class="pill">✅ ข้อมูลจริง (Kaggle/CSV)</span>
      <span class="pill">🌐 เว็บพร้อมส่งลิงก์</span>
      <span class="pill">🌳 Decision Tree</span>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# --------------------- Sidebar: Data ---------------------
st.sidebar.header("① ข้อมูลจริง (CSV จาก Kaggle/ลิงก์สาธารณะ)")
src = st.sidebar.radio("วิธีนำเข้าข้อมูล", ["อัปโหลดไฟล์ CSV", "วางลิงก์ CSV (สาธารณะ)"])
df = None

def clean_after_read(_df: pd.DataFrame) -> pd.DataFrame:
    # ทำความสะอาดค่าที่เป็นสัญลักษณ์/ข้อความว่างให้เป็น NaN
    _df = _df.replace(["", " ", "NA", "N/A", "na", "n/a", "-", "--", "None", "none"], np.nan)
    # แปลงค่าที่เป็น +/- infinity เป็น NaN
    _df = _df.replace([np.inf, -np.inf], np.nan)
    return _df

if src == "อัปโหลดไฟล์ CSV":
    f = st.sidebar.file_uploader("อัปโหลดไฟล์ .csv", type=["csv"])
    if f is not None:
        df = pd.read_csv(f)
        df = clean_after_read(df)
else:
    url = st.sidebar.text_input("วางลิงก์ CSV (เข้าถึงได้โดยไม่ล็อกอิน)")
    if url:
        try:
            parsed = urlparse(url)
            if parsed.scheme in ["http", "https"]:
                df = pd.read_csv(url)
                df = clean_after_read(df)
            else:
                st.sidebar.error("ลิงก์ต้องขึ้นต้นด้วย http/https")
        except Exception as e:
            st.sidebar.error(f"โหลดลิงก์ไม่สำเร็จ: {e}")

if df is None:
    st.info("อัปโหลด CSV จาก Kaggle หรือวางลิงก์ CSV สาธารณะ (ต้องเป็นข้อมูลจริง)")
    st.stop()

# --------------------- Tabs ---------------------
tab_data, tab_train, tab_results, tab_about = st.tabs(["📊 ข้อมูล", "🧪 เทรน & เทียบพารามิเตอร์", "🏁 สรุปผลโมเดล", "ℹ️ เกี่ยวกับงาน"])

# --------------------- Tab: Data ---------------------
with tab_data:
    st.markdown("### ตัวอย่างข้อมูล")
    st.dataframe(df.head(20), use_container_width=True)
    st.caption(f"ทั้งหมด: **{df.shape[0]} แถว × {df.shape[1]} คอลัมน์**")

    st.markdown("#### เลือกคอลัมน์เป้าหมาย (ค่าใช้จ่ายรายเดือน)")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) == 0:
        st.error("ต้องมีคอลัมน์ตัวเลข (เช่น monthly_expense/total_expense)")
        st.stop()

    c1, c2 = st.columns([2,1])
    with c1:
        target_col = st.selectbox("คอลัมน์ค่าใช้จ่าย (ตัวเลข)", options=num_cols)
    with c2:
        thr_mode = st.radio("กำหนด Label", [">= มัธยฐาน = สูง", "กำหนด threshold เอง"], horizontal=False)

    if thr_mode == ">= มัธยฐาน = สูง":
        thr = float(np.median(df[target_col].dropna()))
    else:
        thr = st.number_input("Threshold (เช่น 10000)", min_value=0.0, value=float(np.median(df[target_col].dropna())), step=100.0)

    y = (df[target_col] >= thr).astype(int)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown(f"""<div class="card"><b>นิยาม Label</b><hr class="soft"/>
        1 = {target_col} ≥ {thr:.2f}<br/>0 = {target_col} < {thr:.2f}</div>""", unsafe_allow_html=True)
    with col_r:
        st.markdown(f"""<div class="card"><b>สัดส่วนคลาส</b><hr class="soft"/>
        1 → {int(y.sum())} แถว &nbsp;&nbsp;|&nbsp;&nbsp; 0 → {int((1-y).sum())} แถว</div>""", unsafe_allow_html=True)

    st.markdown("#### เลือกฟีเจอร์เริ่มต้น")
    candidates = [c for c in df.columns if c != target_col]
    if len(candidates) == 0:
        st.error("ต้องมีคอลัมน์อื่นสำหรับใช้เป็นฟีเจอร์")
        st.stop()

    features_selected = st.multiselect("ฟีเจอร์", options=candidates, default=candidates)
    if len(features_selected) == 0:
        st.error("ต้องเลือกอย่างน้อย 1 ฟีเจอร์")
        st.stop()

# --------------------- Common Preprocessor (with Imputer) ---------------------
X = df[features_selected].copy()
num_feats = X.select_dtypes(include=[np.number]).columns.tolist()
cat_feats = [c for c in X.columns if c not in num_feats]

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore"))
])

pre_all = ColumnTransformer(
    transformers=[
        ("num", num_pipe, num_feats),
        ("cat", cat_pipe, cat_feats),
    ],
    remainder="drop"
)

# --------------------- Sidebar: Train options ---------------------
st.sidebar.header("② การฝึกโมเดล (Decision Tree)")
test_size = st.sidebar.slider("Test size (hold-out)", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("random_state", min_value=0, value=42, step=1)
cv_folds = st.sidebar.slider("StratifiedKFold (CV folds)", 3, 10, 5)

criterion = st.sidebar.selectbox("criterion", ["gini", "entropy"])
max_depths = st.sidebar.multiselect("max_depth", [3,5,7,9,None], default=[3,5,7])
min_samples_splits = st.sidebar.multiselect("min_samples_split", [2,4,6,8], default=[2,4])
min_samples_leafs = st.sidebar.multiselect("min_samples_leaf", [1,2,4], default=[1])

st.sidebar.header("③ เลือกฟีเจอร์ขั้นสูง")
use_selectk = st.sidebar.checkbox("เปิดใช้ SelectKBest", value=True)
k_values = st.sidebar.multiselect("ค่า k", options=[5,7,9,11,13,15,20], default=[5,9,13])

# --------------------- Tab: Training ---------------------
with tab_train:
    st.markdown("### เทรน & เทียบพารามิเตอร์/ฟีเจอร์")
    st.markdown('<span class="pill">Stratified K-Fold CV</span> <span class="param-pill">criterion: '
                + criterion + '</span>', unsafe_allow_html=True)
    st.write("")

    def run_cv_table():
        results = []
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        feature_sets = [("all", None)]
        if use_selectk:
            feature_sets += [(f"SelectKBest(k={k})", k) for k in k_values]

        for tag, k in feature_sets:
            if tag.startswith("SelectKBest"):
                base = Pipeline([
                    ("pre", pre_all),
                    ("sel", SelectKBest(score_func=f_classif, k=max(1, min(k, len(features_selected))))),
                    ("clf", DecisionTreeClassifier(random_state=random_state, criterion=criterion))
                ])
            else:
                base = Pipeline([
                    ("pre", pre_all),
                    ("clf", DecisionTreeClassifier(random_state=random_state, criterion=criterion))
                ])

            for md in max_depths:
                for mss in min_samples_splits:
                    for msl in min_samples_leafs:
                        pipe = base.set_params(
                            clf__max_depth=md,
                            clf__min_samples_split=mss,
                            clf__min_samples_leaf=msl
                        )
                        acc = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy").mean()
                        results.append({
                            "features": tag,
                            "params": f"criterion={criterion}, max_depth={md}, min_samples_split={mss}, min_samples_leaf={msl}",
                            "cv_accuracy": acc
                        })
        return pd.DataFrame(results).sort_values("cv_accuracy", ascending=False).reset_index(drop=True)

    with st.spinner("กำลังประมวลผลหลายชุด (CV)..."):
        cv_table = run_cv_table()

    cv_table_display = cv_table.copy()
    cv_table_display.insert(0, "rank", cv_table_display.index + 1)
    st.dataframe(
        cv_table_display.rename(columns={
            "rank": "rank",
            "features": "features",
            "params": "parameters",
            "cv_accuracy": "cv_accuracy"
        }).style.format({"cv_accuracy": "{:.4f}"}),
        use_container_width=True
    )

# --------------------- Tab: Results ---------------------
with tab_results:
    st.markdown("### ผลลัพธ์ & โมเดลสุดท้าย")
    best = cv_table.iloc[0]
    c1, c2, c3 = st.columns([3,2,2])
    with c1:
        st.markdown(f"""<div class="card best">✅ ชุดที่ดีที่สุด<br/>
        <span class="kpill">Features: {best["features"]}</span><br/>
        <span class="param-pill">{best["params"]}</span></div>""", unsafe_allow_html=True)
    with c2:
        st.metric("CV Accuracy (ดีที่สุด)", f"{best['cv_accuracy']:.4f}")
    with c3:
        st.markdown(f'<div class="card">Hold-out test \u2022 test_size = {test_size:.2f} \u2022 random_state = {random_state}</div>', unsafe_allow_html=True)

    st.markdown("#### Training combinations")
    training_overview = cv_table_display.rename(columns={
        "rank": "Rank",
        "features": "Feature set",
        "params": "Parameters",
        "cv_accuracy": "CV Accuracy"
    })
    st.dataframe(
        training_overview.style.format({"CV Accuracy": "{:.4f}"}),
        use_container_width=True
    )

    # build best pipeline
    def build_best_pipeline(row):
        feat_tag = row["features"]
        md_txt = row["params"].split("max_depth=")[1].split(",")[0].strip()
        md = None if md_txt=="None" else int(md_txt)
        mss = int(row["params"].split("min_samples_split=")[1].split(",")[0].strip())
        msl = int(row["params"].split("min_samples_leaf=")[1].strip())
        if feat_tag.startswith("SelectKBest"):
            import re
            match = re.search(r"k=(\d+)", feat_tag)
            k = int(match.group(1)) if match else len(features_selected)
            pipe = Pipeline([
                ("pre", pre_all),
                ("sel", SelectKBest(score_func=f_classif, k=max(1, min(k, len(features_selected))))),
                ("clf", DecisionTreeClassifier(
                    random_state=random_state, criterion=criterion,
                    max_depth=md, min_samples_split=mss, min_samples_leaf=msl
                ))
            ])
        else:
            pipe = Pipeline([
                ("pre", pre_all),
                ("clf", DecisionTreeClassifier(
                    random_state=random_state, criterion=criterion,
                    max_depth=md, min_samples_split=mss, min_samples_leaf=msl
                ))
            ])
        return pipe

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    final_pipe = build_best_pipeline(best)
    final_pipe.fit(X_train, y_train)
    y_pred = final_pipe.predict(X_test)

    test_acc = accuracy_score(y_test, y_pred)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Test Accuracy (hold-out)", f"{test_acc:.4f}")
    with col2:
        st.markdown('<div class="card">โมเดลสุดท้าย = Decision Tree</div>', unsafe_allow_html=True)

    st.markdown("#### Classification Report (test)")
    st.text(classification_report(y_test, y_pred, digits=4))

    st.markdown("#### Confusion Matrix (test)")
    cm = confusion_matrix(y_test, y_pred)
    st.dataframe(pd.DataFrame(cm, index=["Actual 0","Actual 1"], columns=["Pred 0","Pred 1"]), use_container_width=True)

    # Feature importances (หลังเข้ารหัส)
    st.markdown("#### สำคัญที่สุด (Feature Importances)")
    try:
        pre = final_pipe.named_steps["pre"]
        num_names = pre.transformers_[0][2]
        cat_names = pre.transformers_[1][2]
        ohe = pre.named_transformers_["cat"]
        ohe_names = list(ohe.get_feature_names_out(cat_names)) if len(cat_names)>0 else []
        all_feat_names = list(num_names) + ohe_names
        importances = final_pipe.named_steps["clf"].feature_importances_
        imp_df = (pd.DataFrame({"feature": all_feat_names, "importance": importances})
                  .sort_values("importance", ascending=False)
                  .head(20))
        st.dataframe(imp_df, use_container_width=True)
    except Exception:
        st.info("ไม่สามารถดึง feature importances ได้ (ขึ้นกับการเข้ารหัส/การเลือกฟีเจอร์)")

    st.markdown(
        f"""
        <div class="card">
            <b>สรุปการเลือกโมเดลสุดท้าย</b><br/>
            Feature set: <code>{best['features']}</code><br/>
            Parameters: <code>{best['params']}</code><br/>
            CV accuracy: {best['cv_accuracy']:.4f}<br/>
            Hold-out accuracy: {test_acc:.4f}
        </div>
        """,
        unsafe_allow_html=True
    )

# --------------------- Tab: About ---------------------
with tab_about:
    st.markdown("### เกี่ยวกับงาน/การส่ง")
    st.markdown("""
- ✅ ใช้ข้อมูลจริงจาก Kaggle/CSV สาธารณะ (ไม่ใช้ข้อมูลที่สร้างเอง)  
- ✅ เว็บแอป Decision Tree เท่านั้น  
- ✅ แสดงตารางผลการเทรน (ฟีเจอร์ & พารามิเตอร์หลายแบบ) พร้อม Accuracy  
- ✅ สรุปชุดฟีเจอร์/พารามิเตอร์ที่เลือกใช้ + ทดสอบ hold-out  
- ✅ ส่ง **ลิงก์เว็บ** ได้เลย
    """)
