# -*- coding: utf-8 -*-
# app.py
# เว็บแอป Streamlit ทำนายโอกาสรอดชีวิตบนเรือ Titanic
# ต้องวางไฟล์ต่อไปนี้ไว้โฟลเดอร์เดียวกับ app.py
#   - titanic_scaler.joblib  (StandardScaler ที่ fit ไว้แล้ว)
#   - titanic_tree.joblib    (โมเดล Decision Tree)
# วิธีรัน:  streamlit run app.py

from pathlib import Path  # ใช้หาที่อยู่ไฟล์ (ไลบรารีมาตรฐานของ Python)

import joblib               # โหลดไฟล์ .joblib
import pandas as pd         # สร้างตารางข้อมูล 1 แถวจากค่าที่ผู้ใช้กรอก
import sklearn              # ใช้แสดงเวอร์ชัน scikit-learn
import streamlit as st      # สร้างหน้าเว็บ

# ---------------------------------------------------------------
# ส่วนที่ 1: ตั้งค่าหน้าเว็บและที่อยู่ไฟล์
# ---------------------------------------------------------------
st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")

BASE_DIR = Path(__file__).resolve().parent      # โฟลเดอร์ที่เก็บ app.py
SCALER_PATH = BASE_DIR / "titanic_scaler.joblib"
MODEL_PATH = BASE_DIR / "titanic_tree.joblib"


# ---------------------------------------------------------------
# ส่วนที่ 2: โหลด scaler และโมเดล (โหลดครั้งเดียว แล้วเก็บไว้ใน cache)
# ---------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(MODEL_PATH)
    return scaler, model


try:
    scaler, model = load_artifacts()
except Exception as error:
    # ถ้าโหลดไม่ได้ ให้บอกสิ่งที่ควรตรวจ แล้วหยุดการทำงานของแอป
    st.error("โหลดไฟล์โมเดลไม่สำเร็จ")
    st.write(f"ชนิดข้อผิดพลาด: `{type(error).__name__}` — {error}")
    st.markdown(
        f"""
**สิ่งที่ควรตรวจสอบ**
1. ไฟล์ `titanic_scaler.joblib` และ `titanic_tree.joblib` ต้องอยู่ในโฟลเดอร์เดียวกับ `app.py` (`{BASE_DIR}`)
2. ชื่อไฟล์ต้องตรงทุกตัวอักษร
3. เวอร์ชัน scikit-learn ในเครื่องนี้คือ **{sklearn.__version__}** ต้องตรงกับเวอร์ชันที่ใช้บันทึกไฟล์ใน Colab
"""
    )
    st.stop()


# ---------------------------------------------------------------
# ส่วนที่ 3: แปลงค่าที่ผู้ใช้กรอก ให้อยู่ในรูปเดียวกับตอนที่ fit scaler
# ---------------------------------------------------------------
# ข้อมูลใน scaler บอกว่าตอนเทรน ข้อมูลผ่านการแปลงมาก่อนเข้า scaler:
#   - Pclass ถูกลดลง 1 (คลาส 1,2,3 -> 0,1,2)
#   - Age ถูกทำ min-max ให้อยู่ช่วง 0-1 (ใช้ต่ำสุด 0.42 สูงสุด 80 ปี)
#   - Sex_female = 1 ถ้าเป็นหญิง, 0 ถ้าเป็นชาย
#   - FamilySize = SibSp + Parch + 1 (ไม่ได้แปลง)
#   - Fare ยังไม่ทราบวิธีแปลงที่ใช้ใน Colab  <-- ต้องแก้ฟังก์ชันด้านล่างให้ตรง
AGE_MIN, AGE_MAX = 0.42, 80.0
FARE_MAX = 512.3292


def transform_fare(fare):
    # TODO: แก้ให้ตรงกับที่ทำใน Colab ตอนเตรียมข้อมูล Fare
    # ตอนนี้ใช้ min-max (หารด้วยค่าสูงสุด) เป็นค่าสมมติ
    return fare / FARE_MAX


def build_raw_row(pclass, sex, age, fare, family_size):
    """สร้างตาราง 1 แถวที่มีคอลัมน์ตรงกับที่ scaler เคยเห็นตอน fit"""
    row = {
        "Pclass": pclass - 1,
        "Sex_female": 1 if sex == "หญิง" else 0,
        "Age": (age - AGE_MIN) / (AGE_MAX - AGE_MIN),
        "Fare": transform_fare(fare),
        "FamilySize": family_size,
    }
    return pd.DataFrame([row])[list(scaler.feature_names_in_)]


# ---------------------------------------------------------------
# ส่วนที่ 4: หน้าเว็บ
# ---------------------------------------------------------------
st.title("🚢 ทำนายโอกาสรอดชีวิตบนเรือ Titanic")
st.caption("กรอกข้อมูลผู้โดยสาร แล้วกดปุ่มทำนาย")

col1, col2 = st.columns(2)
with col1:
    pclass = st.selectbox("ชั้นโดยสาร (Pclass)", [1, 2, 3], index=2)
    sex = st.radio("เพศ", ["ชาย", "หญิง"], horizontal=True)
    age = st.number_input("อายุ (ปี)", min_value=0.42, max_value=80.0,
                          value=30.0, step=1.0)
with col2:
    fare = st.number_input("ค่าโดยสาร (Fare)", min_value=0.0,
                           max_value=FARE_MAX, value=32.0, step=1.0)
    family_size = st.number_input(
        "จำนวนสมาชิกครอบครัวบนเรือ (รวมตัวเอง)", min_value=1, max_value=11,
        value=1, step=1,
        help="นับตัวเอง + พี่น้อง/คู่สมรส + พ่อแม่/ลูก ที่เดินทางมาด้วย")

if st.button("ทำนาย", type="primary"):
    # 4.1 สร้างข้อมูลดิบ 1 แถว แล้วผ่าน scaler
    raw_row = build_raw_row(pclass, sex, age, fare, family_size)
    scaled = scaler.transform(raw_row)

    # 4.2 เตรียมข้อมูลเข้าโมเดล
    if model.n_features_in_ != scaled.shape[1]:
        st.error(f"โมเดลต้องการ {model.n_features_in_} ฟีเจอร์ "
                 f"แต่ scaler ให้ {scaled.shape[1]} ฟีเจอร์ "
                 "ไฟล์สองตัวนี้อาจมาจากคนละชุดการเทรน")
        st.stop()

    model_input = pd.DataFrame(scaled, columns=scaler.feature_names_in_)
    if hasattr(model, "feature_names_in_"):
        if set(model.feature_names_in_) != set(model_input.columns):
            st.error("ชื่อฟีเจอร์ของโมเดลไม่ตรงกับ scaler: "
                     f"โมเดล = {list(model.feature_names_in_)}, "
                     f"scaler = {list(model_input.columns)}")
            st.stop()
        model_input = model_input[list(model.feature_names_in_)]

    # 4.3 ทำนายและแสดงผล
    prediction = model.predict(model_input)[0]
    st.subheader("ผลการทำนาย")
    if prediction == 1:
        st.success("คาดว่า **รอดชีวิต** 🎉")
    else:
        st.error("คาดว่า **ไม่รอดชีวิต**")

    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        if 1 in classes:
            p_survive = model.predict_proba(model_input)[0][classes.index(1)]
            st.metric("ความน่าจะเป็นที่จะรอดชีวิต", f"{p_survive:.1%}")
            st.progress(float(p_survive))

    # 4.4 แสดงรายละเอียดกลางทาง ไว้ตรวจว่าค่าที่เข้าโมเดลสมเหตุสมผล
    with st.expander("รายละเอียดทางเทคนิค (สำหรับตรวจสอบ)"):
        st.write("ข้อมูลก่อนเข้า scaler:")
        st.dataframe(raw_row)
        st.write("ข้อมูลหลังผ่าน scaler:")
        st.dataframe(pd.DataFrame(scaled, columns=scaler.feature_names_in_))
        st.write(f"scikit-learn ในเครื่องนี้: {sklearn.__version__}")
