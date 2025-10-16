<div align="center">
  <h1>AI ผู้ช่วยวางแผนและพยากรณ์ค่าใช้จ่ายรายเดือนของนักศึกษา</h1>
  <p>
    เว็บแอป Machine Learning (Decision Tree) เพื่อช่วยนักศึกษาวิเคราะห์และพยากรณ์พฤติกรรมการใช้จ่าย  
    <br/>
    ใช้ข้อมูลจริงจาก Kaggle พร้อมระบบเทียบพารามิเตอร์, Accuracy, Confusion Matrix และ Feature Importances
  </p>
  <a href="https://student-expense-ai-pwpitchaya.streamlit.app">
    🌐 เปิดใช้งานแอปบน Streamlit
  </a>
</div>

---

## 📘 รายละเอียดโครงงาน
| หัวข้อ | รายละเอียด |
|--------|-------------|
| **ชื่อโปรเจกต์** | AI ผู้ช่วยวางแผนและพยากรณ์ค่าใช้จ่ายรายเดือนของนักศึกษา |
| **เทคนิคหลัก** | Decision Tree Classifier |
| **เครื่องมือที่ใช้** | Python, Streamlit, scikit-learn, Pandas, Numpy |
| **ชุดข้อมูล** | Student Spending Dataset (Kaggle, 2023) |
| **ผู้จัดทำ** | นางสาวภวพิชญา คำวงษา  |

---

## 🧠 ฟีเจอร์หลักของเว็บแอป

- ✅ **อัปโหลดข้อมูลจริงจาก Kaggle หรือ CSV สาธารณะ**
- 🧩 **ปรับพารามิเตอร์ Decision Tree ได้ครบ (criterion, depth, split, leaf)**
- ⚙️ **เลือกจำนวนฟีเจอร์ด้วย SelectKBest**
- 📊 **เปรียบเทียบ Accuracy ของแต่ละชุดผ่าน Cross Validation**
- 🏁 **แสดงผล Test Accuracy, Confusion Matrix และ Feature Importances**
- 💬 **สรุปข้อความส่งงานอัตโนมัติสำหรับรายงาน**

---

## ⚙️ วิธีติดตั้งและรันบนเครื่อง

```bash
# 1️⃣ ติดตั้งแพ็กเกจที่จำเป็น
pip install -r requirements.txt

# 2️⃣ รันเว็บแอป
streamlit run app.py
```
## 📊 โครงสร้างโปรเจกต์
```bash
student-expense-ai/
├── app.py
├── requirements.txt
├── .streamlit/
│   └── config.toml
└── University Students Monthly Expenses.csv (optional)
```
## 💡 อธิบายพารามิเตอร์สำคัญ
พารามิเตอร์	คำอธิบาย
criterion	เกณฑ์วัดความไม่บริสุทธิ์ของโหนด (Gini/Entropy)
max_depth	ความลึกสูงสุดของต้นไม้ (None = ไม่จำกัด)
min_samples_split	จำนวนขั้นต่ำก่อนจะแตกกิ่ง
min_samples_leaf	จำนวนขั้นต่ำของตัวอย่างในแต่ละใบ
SelectKBest (k)	เลือกฟีเจอร์ K ตัวที่สัมพันธ์กับคลาสมากที่สุด
CV folds	จำนวนพับในการทำ Cross Validation

## ✨ ผู้พัฒนา

💻 **(นางสาวภวพิชญา คำวงษา  )**  
🎓 **สาขาวิทยาการคอมพิวเตอร์และสารสนเทศ**  
🏫 **คณะสหวิทยาการ**
