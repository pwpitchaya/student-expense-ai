import re

import numpy as np
import pandas as pd
import streamlit as st
from urllib.parse import urlparse

from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# --------------------- Page Config ---------------------
st.set_page_config(
    page_title="AI ผู้ช่วยค่าใช้จ่ายนักศึกษา (Decision Tree)",
    page_icon="💸",
    layout="wide",
)

# --------------------- Global Styles ---------------------
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)

# --------------------- Header ---------------------
colA, colB = st.columns([1, 4])
with colA:
    st.markdown(
        '<div class="hero"><div class="emoji">💸</div><div><div class="title">AI ผู้ช่วยค่าใช้จ่ายนักศึกษา</div><div class="sub">เวิร์กโฟลว์เทรน Decision Tree • อัปเดตผลบนหน้าเดียว</div></div></div>',
        unsafe_allow_html=True,
    )
with colB:
    st.markdown(
        """
    <div class="card" style="display:flex;gap:.6rem;justify-content:flex-end;align-items:center;">
      <span class="pill">✅ ข้อมูลจริง (Kaggle/CSV)</span>
      <span class="pill">🌐 หน้าเว็บพร้อมส่งงาน</span>
      <span class="pill">🌳 Decision Tree</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.write("")

# --------------------- Sidebar: Data ---------------------
st.sidebar.header("① ข้อมูลจาก CSV (Kaggle/แหล่งสาธารณะ)")
src = st.sidebar.radio("เลือกวิธีนำเข้าข้อมูล", ["อัปโหลดไฟล์ CSV", "วางลิงก์ CSV (สาธารณะ)"])
df = None


def clean_after_read(_df: pd.DataFrame) -> pd.DataFrame:
    """ทำความสะอาดค่าเบื้องต้น เช่น ช่องว่างหรือค่าไม่ระบุให้เป็น NaN"""
    _df = _df.replace(["", " ", "NA", "N/A", "na", "n/a", "-", "--", "None", "none"], np.nan)
    _df = _df.replace([np.inf, -np.inf], np.nan)
    return _df


if src == "อัปโหลดไฟล์ CSV":
    f = st.sidebar.file_uploader("เลือกไฟล์ .csv", type=["csv"])
    if f is not None:
        df = pd.read_csv(f)
        df = clean_after_read(df)
else:
    url = st.sidebar.text_input("ลิงก์ไฟล์ CSV (ต้องเป็นลิงก์ดาวน์โหลดได้เลย)")
    if url:
        try:
            parsed = urlparse(url)
            if parsed.scheme in ["http", "https"]:
                df = pd.read_csv(url)
                df = clean_after_read(df)
            else:
                st.sidebar.error("ลิงก์ต้องขึ้นต้นด้วย http หรือ https")
        except Exception as e:
            st.sidebar.error(f"โหลดลิงก์ไม่สำเร็จ: {e}")

if df is None:
    st.info("อัปโหลดไฟล์ CSV หรือใส่ลิงก์ CSV สาธารณะทางแถบด้านซ้าย เพื่อเริ่มต้นใช้งาน")
    st.stop()

st.caption("ปรับค่าทางด้านซ้ายแล้วระบบจะคำนวณผลใหม่ให้โดยอัตโนมัติ")

# --------------------- ขั้นตอนหลักบนหน้าเดียว ---------------------
st.markdown("### 1) ตรวจสอบข้อมูลเบื้องต้น")
st.dataframe(df.head(20), use_container_width=True)
st.caption(f"ชุดข้อมูลนี้มีทั้งหมด **{df.shape[0]} แถว × {df.shape[1]} คอลัมน์**")

# --------------------- เลือกเป้าหมายและฟีเจอร์ ---------------------
st.markdown("### 2) กำหนดเป้าหมาย (label) และเลือกฟีเจอร์")
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(num_cols) == 0:
    st.error("ต้องมีคอลัมน์ชนิดตัวเลขอย่างน้อย 1 คอลัมน์ (ตัวอย่างเช่น monthly_expense)")
    st.stop()

c1, c2 = st.columns([2, 1])
with c1:
    target_col = st.selectbox("เลือกคอลัมน์ที่ใช้สร้าง label (ค่าใช้จ่าย)", options=num_cols)
with c2:
    thr_mode = st.radio("วิธีแบ่งระดับค่าใช้จ่าย", [">= มัธยฐาน = สูง", "กำหนด threshold เอง"], horizontal=False)

if thr_mode == ">= มัธยฐาน = สูง":
    thr = float(np.median(df[target_col].dropna()))
else:
    thr = st.number_input(
        "ตั้งค่า threshold เอง (เช่น 10000)",
        min_value=0.0,
        value=float(np.median(df[target_col].dropna())),
        step=100.0,
    )

y = (df[target_col] >= thr).astype(int)

col_l, col_r = st.columns(2)
with col_l:
    st.markdown(
        f"""<div class="card"><b>นิยาม Label</b><hr class="soft"/>
        1 = {target_col} ≥ {thr:.2f}<br/>0 = {target_col} < {thr:.2f}</div>""",
        unsafe_allow_html=True,
    )
with col_r:
    st.markdown(
        f"""<div class="card"><b>สัดส่วนคลาส</b><hr class="soft"/>
        1 → {int(y.sum())} แถว &nbsp;&nbsp;|&nbsp;&nbsp; 0 → {int((1 - y).sum())} แถว</div>""",
        unsafe_allow_html=True,
    )

st.markdown("เลือกฟีเจอร์ที่จะใช้เป็นตัวแปรอิสระ (เลือกหลายค่าได้)")
candidates = [c for c in df.columns if c != target_col]
if len(candidates) == 0:
    st.error("ต้องมีคอลัมน์อื่นนอกจากคอลัมน์เป้าหมาย เพื่อใช้เป็นฟีเจอร์")
    st.stop()

features_selected = st.multiselect("ฟีเจอร์", options=candidates, default=candidates)
if len(features_selected) == 0:
    st.error("เลือกอย่างน้อย 1 ฟีเจอร์เพื่อสร้างโมเดล")
    st.stop()

# --------------------- เตรียม Pipeline การแปลงข้อมูล ---------------------
X = df[features_selected].copy()
num_feats = X.select_dtypes(include=[np.number]).columns.tolist()
cat_feats = [c for c in X.columns if c not in num_feats]

num_pipe = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

cat_pipe = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ]
)

pre_all = ColumnTransformer(
    transformers=[
        ("num", num_pipe, num_feats),
        ("cat", cat_pipe, cat_feats),
    ],
    remainder="drop",
)

# --------------------- Sidebar: ตัวเลือกการฝึก ---------------------
st.sidebar.header("② ตั้งค่าการฝึก Decision Tree")
test_size = st.sidebar.slider("ขนาดชุดทดสอบ (hold-out)", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("ค่า random_state", min_value=0, value=42, step=1)
cv_folds = st.sidebar.slider("จำนวน Stratified K-Fold", 3, 10, 5)

criterion = st.sidebar.selectbox("criterion", ["gini", "entropy"])
max_depths = st.sidebar.multiselect("max_depth", [3, 5, 7, 9, None], default=[3, 5, 7])
min_samples_splits = st.sidebar.multiselect("min_samples_split", [2, 4, 6, 8], default=[2, 4])
min_samples_leafs = st.sidebar.multiselect("min_samples_leaf", [1, 2, 4], default=[1])

st.sidebar.header("③ ตัวเลือกฟีเจอร์ขั้นสูง")
use_selectk = st.sidebar.checkbox("ลองใช้ SelectKBest", value=True)
k_values = st.sidebar.multiselect("ค่า k ที่ทดลอง", options=[5, 7, 9, 11, 13, 15, 20], default=[5, 9, 13])

# --------------------- ส่วนสรุปผลบนหน้าเดียว ---------------------
st.markdown("### 3) เทรน เปรียบเทียบ และเลือกโมเดลที่ดีที่สุด")
st.markdown(
    '<span class="pill">Stratified K-Fold CV</span> '
    f'<span class="param-pill">criterion: {criterion}</span>',
    unsafe_allow_html=True,
)


def run_cv_table() -> pd.DataFrame:
    """เทรนหลายชุดฟีเจอร์/พารามิเตอร์ แล้วสรุปค่าเฉลี่ย Accuracy จาก Cross-Validation"""
    results = []
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    feature_sets = [("all", None)]
    if use_selectk:
        feature_sets += [(f"SelectKBest(k={k})", k) for k in k_values]

    for tag, k in feature_sets:
        if tag.startswith("SelectKBest"):
            base = Pipeline(
                steps=[
                    ("pre", pre_all),
                    ("sel", SelectKBest(score_func=f_classif, k=max(1, min(k, len(features_selected))))),
                    ("clf", DecisionTreeClassifier(random_state=random_state, criterion=criterion)),
                ]
            )
        else:
            base = Pipeline(
                steps=[
                    ("pre", pre_all),
                    ("clf", DecisionTreeClassifier(random_state=random_state, criterion=criterion)),
                ]
            )

        for md in max_depths:
            for mss in min_samples_splits:
                for msl in min_samples_leafs:
                    pipe = base.set_params(
                        clf__max_depth=md,
                        clf__min_samples_split=mss,
                        clf__min_samples_leaf=msl,
                    )
                    acc = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy").mean()
                    results.append(
                        {
                            "features": tag,
                            "params": (
                                f"criterion={criterion}, max_depth={md}, "
                                f"min_samples_split={mss}, min_samples_leaf={msl}"
                            ),
                            "cv_accuracy": acc,
                        }
                    )

    return pd.DataFrame(results).sort_values("cv_accuracy", ascending=False).reset_index(drop=True)


with st.spinner("กำลังประมวลผลผลการเทรนทั้งหมด..."):
    cv_table = run_cv_table()

cv_table_display = cv_table.copy()
cv_table_display.insert(0, "ลำดับ", cv_table_display.index + 1)
cv_table_display = cv_table_display.rename(
    columns={
        "features": "ชุดฟีเจอร์",
        "params": "พารามิเตอร์",
        "cv_accuracy": "ความแม่นยำ (CV)",
    }
)

st.markdown("#### 3.1 ตารางสรุปผลการทดลอง (Cross-Validation)")
st.dataframe(cv_table_display.style.format({"ความแม่นยำ (CV)": "{:.4f}"}), use_container_width=True)

best = cv_table.iloc[0]
c1, c2, c3 = st.columns([3, 2, 2])
with c1:
    st.markdown(
        f"""<div class="card best">✅ โมเดลเด่นที่สุด<br/>
        <span class="kpill">ฟีเจอร์: {best["features"]}</span><br/>
        <span class="param-pill">{best["params"]}</span></div>""",
        unsafe_allow_html=True,
    )
with c2:
    st.metric("CV Accuracy (สูงสุด)", f"{best['cv_accuracy']:.4f}")
with c3:
    st.markdown(
        f'<div class="card">ตั้งค่า hold-out • test_size = {test_size:.2f} • random_state = {random_state}</div>',
        unsafe_allow_html=True,
    )


def build_best_pipeline(row: pd.Series) -> Pipeline:
    feat_tag = row["features"]
    md_txt = row["params"].split("max_depth=")[1].split(",")[0].strip()
    md = None if md_txt == "None" else int(md_txt)
    mss = int(row["params"].split("min_samples_split=")[1].split(",")[0].strip())
    msl = int(row["params"].split("min_samples_leaf=")[1].strip())

    if feat_tag.startswith("SelectKBest"):
        match = re.search(r"k=(\d+)", feat_tag)
        k = int(match.group(1)) if match else len(features_selected)
        return Pipeline(
            steps=[
                ("pre", pre_all),
                ("sel", SelectKBest(score_func=f_classif, k=max(1, min(k, len(features_selected))))),
                ("clf", DecisionTreeClassifier(
                    random_state=random_state,
                    criterion=criterion,
                    max_depth=md,
                    min_samples_split=mss,
                    min_samples_leaf=msl,
                )),
            ]
        )

    return Pipeline(
        steps=[
            ("pre", pre_all),
            ("clf", DecisionTreeClassifier(
                random_state=random_state,
                criterion=criterion,
                max_depth=md,
                min_samples_split=mss,
                min_samples_leaf=msl,
            )),
        ]
    )


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
)
final_pipe = build_best_pipeline(best)
final_pipe.fit(X_train, y_train)
y_pred = final_pipe.predict(X_test)

test_acc = accuracy_score(y_test, y_pred)
col1, col2 = st.columns(2)
with col1:
    st.metric("Hold-out Accuracy", f"{test_acc:.4f}")
with col2:
    st.markdown('<div class="card">โมเดลสุดท้าย: Decision Tree</div>', unsafe_allow_html=True)

with st.expander("ดูรายละเอียดรายงาน (Classification Report + Confusion Matrix)"):
    st.markdown("##### Classification Report")
    st.text(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            digits=4,
            zero_division=0,
        )
    )
    st.markdown("##### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    st.dataframe(
        pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]),
        use_container_width=True,
    )

with st.expander("ดูความสำคัญของฟีเจอร์ (Feature Importances)"):
    try:
        pre = final_pipe.named_steps["pre"]
        num_names = pre.transformers_[0][2]
        cat_names = pre.transformers_[1][2]
        ohe = pre.named_transformers_["cat"]
        ohe_names = list(ohe.get_feature_names_out(cat_names)) if len(cat_names) > 0 else []
        all_feat_names = list(num_names) + ohe_names
        importances = final_pipe.named_steps["clf"].feature_importances_
        imp_df = (
            pd.DataFrame({"feature": all_feat_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .head(20)
        )
        st.dataframe(imp_df, use_container_width=True)
    except Exception:
        st.info("ไม่สามารถคำนวณ feature importances ได้ (ขึ้นอยู่กับการเข้ารหัส/ฟีเจอร์ที่เลือก)")

st.markdown(
    f"""
    <div class="card">
        <b>สรุปโมเดลที่พร้อมใช้งาน</b><br/>
        ชุดฟีเจอร์: <code>{best['features']}</code><br/>
        พารามิเตอร์: <code>{best['params']}</code><br/>
        ความแม่นยำเฉลี่ย (CV): {best['cv_accuracy']:.4f}<br/>
        ความแม่นยำบน hold-out: {test_acc:.4f}
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("ข้อมูลเพิ่มเติมเกี่ยวกับงานนี้"):
    st.markdown(
        """
- ใช้ข้อมูลจากไฟล์ CSV จริง (ดาวน์โหลดได้หรืออัปโหลดเอง)
- สร้าง Decision Tree เพื่อจำแนกค่าใช้จ่ายสูง/ต่ำแบบอัตโนมัติ
- หน้าเดียวสำหรับเลือกฟีเจอร์ ปรับพารามิเตอร์ และดูผลลัพธ์
- มีทั้งค่าเฉลี่ย Cross-Validation และ Hold-out test ให้ตรวจสอบ
- โหลดผลใหม่ทันทีเมื่อมีการปรับค่าทางแถบด้านซ้าย
        """
    )
