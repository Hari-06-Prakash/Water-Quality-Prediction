import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import oracledb
import hashlib
import datetime

# -------------------------------
# Oracle DB Connection
# -------------------------------
try:
    dsn_tns = oracledb.makedsn(
        'localhost',
        '1521',
        service_name='XEPDB1'
    )
    conn = oracledb.connect(
    user='water_app',
    password='nbk2005',
    dsn=dsn_tns
    )
    cursor = conn.cursor()
    db_connected = True
except Exception as e:
    db_connected = False
    st.warning(f"Database not connected: {e}")

# -------------------------------
# Password hashing
# -------------------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# -------------------------------
# Load ML Model & Feature Names
# -------------------------------
try:
    model, feature_names = joblib.load("model.pkl")  # saved with train_model.py
    model_loaded = True
except:
    model_loaded = False
    feature_names = []

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Smart Water Quality Prediction", layout="wide")
st.title("💧 Smart Water Quality Prediction using ML & XAI")

# -------------------------------
# Initialize session state
# -------------------------------
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
    st.session_state['username'] = None

# -------------------------------
# Sidebar Navigation
# -------------------------------
with st.sidebar:
    choice = option_menu(
        "📌 Navigation",
        ["Home", "Manual Input", "Upload Data", "Dashboard", "History", "Model Performance", "Logout"],
        icons=["house", "pencil-square", "upload", "bar-chart", "clock-history", "graph-up", "box-arrow-right"],
        menu_icon="cast",
        default_index=0
    )

# -------------------------------
# Home Page (Login/Signup)
# -------------------------------
if choice == "Home":
    st.subheader("🏠 Welcome to Smart Water Quality Prediction")
    st.write("""
    This system predicts **water quality** (Safe/Unsafe) using 
    advanced **Machine Learning** and explains results with **XAI**.
    """)

    st.markdown("---")
    st.subheader("🔐 User Login / Signup")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        username = st.text_input("👤 Username", max_chars=20)
        password = st.text_input("🔑 Password", type="password", max_chars=20)

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Login"):
                hashed_pw = hash_password(password)
                cursor.execute(
                    "SELECT * FROM users WHERE username=:u AND password=:p",
                    {'u': username, 'p': hashed_pw}
                )
                user = cursor.fetchone()
                if user:
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = username
                    st.success(f"✅ Logged in as {username}")
                else:
                    st.error("❌ Invalid credentials,Please Sign Up!")

        with col_btn2:
            if st.button("Signup"):
                hashed_pw = hash_password(password)
                try:
                    cursor.execute(
                        "INSERT INTO users (username, password) VALUES (:u, :p)",
                        {'u': username, 'p': hashed_pw}
                    )
                    conn.commit()
                    st.success(f"✅ Account created for {username}")
                except oracledb.IntegrityError:
                    st.error("❌ Username already exists")

# -------------------------------
# Protect pages
# -------------------------------
if choice != "Home" and not st.session_state['logged_in']:
    st.warning("⚠️ You must log in to access this page.")
    st.stop()

# -------------------------------
# Manual Input Page
# -------------------------------
elif choice == "Manual Input":
    st.subheader("📝 Enter Water Quality Parameters")

    if not model_loaded:
        st.warning("⚠️ Model not found. Train it first by running `train_model.py`.")
    else:
        input_data = {}
        for feature in feature_names:
            input_data[feature] = st.number_input(f"{feature}", min_value=0.0, step=0.1)

        input_df = pd.DataFrame([input_data])

        if st.button("🔮 Predict Water Quality"):
            st.write("### ✅ Input Sample Data")
            st.dataframe(input_df)

            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0]

            if prediction == 1:
                result_text = "Safe"
                st.success(f"✅ Predicted Result: **Safe Water** (Confidence: {probability[1]*100:.2f}%)")
            else:
                result_text = "Unsafe"
                st.error(f"❌ Predicted Result: **Unsafe Water** (Confidence: {probability[0]*100:.2f}%)")

            # Save prediction to DB
            cursor.execute(
                "INSERT INTO predictions (username, predicted_result, confidence, created_at) VALUES (:u, :r, :c, :t)",
                {
                    'u': st.session_state['username'],
                    'r': result_text,
                    'c': max(probability)*100,
                    't': datetime.datetime.now()
                }
            )
            conn.commit()

            # SHAP Explanation
            st.subheader("🔎 Explanation (XAI - Feature Impact)")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)

            if isinstance(shap_values, list):
                row_values = shap_values[1][0]
                expected_val = explainer.expected_value[1]
            else:
                row_values = shap_values[0]
                if row_values.ndim > 1:
                    row_values = row_values[0]
                row_values = row_values.flatten()
                expected_val = explainer.expected_value
                if isinstance(expected_val, (list, tuple)) or hasattr(expected_val, "__len__"):
                    expected_val = expected_val[0]

            fig, ax = plt.subplots(figsize=(8, 6))
            shap.plots._waterfall.waterfall_legacy(
                expected_val,
                row_values,
                feature_names=feature_names,
                max_display=10,
                show=False
            )
            st.pyplot(fig)

# -------------------------------
# Upload Data Page
# -------------------------------
elif choice == "Upload Data":
    st.subheader("📂 Upload Dataset for Bulk Prediction")

    uploaded_file = st.file_uploader("Upload CSV/Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.session_state["uploaded_df"] = df
            st.write("### 📊 Dataset Preview")
            st.dataframe(df)

            if model_loaded:
                df_features = df[feature_names]
                predictions = model.predict(df_features)
                proba = model.predict_proba(df_features)

                df_results = df.copy()
                df_results["Prediction"] = ["✅Safe" if p == 1 else "❌Unsafe" for p in predictions]
                df_results["Confidence"] = [f"{max(prob)*100:.2f}%" for prob in proba]

                st.write("### 🚀 Prediction Results")
                st.dataframe(df_results)

                # Save predictions to DB
                for i, row in df_results.iterrows():
                    cursor.execute(
                        "INSERT INTO predictions (username, predicted_result, confidence, created_at) VALUES (:u, :r, :c, :t)",
                        {
                            'u': st.session_state['username'],
                            'r': row["Prediction"],
                            'c': float(row["Confidence"].replace("%","")),
                            't': datetime.datetime.now()
                        }
                    )
                conn.commit()

        except Exception as e:
            st.error(f"❌ Error reading file: {e}")

# -------------------------------
# Dashboard Page
# -------------------------------
elif choice == "Dashboard":
    st.subheader("📊 Data Dashboard")
    try:
        if "uploaded_df" in st.session_state:
            df = st.session_state["uploaded_df"]
        else:
            df = pd.read_csv("water_potability.csv")  # fallback

        st.write("### Dataset Overview")
        st.dataframe(df.head())

        fig, ax = plt.subplots()
        df["Potability"].value_counts().plot(kind="bar", color=["red","green"], ax=ax)
        ax.set_xticklabels(["Unsafe","Safe"])
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(df.corr(), annot=False, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"❌ Could not load dataset: {e}")

# -------------------------------
# History Page
# -------------------------------
elif choice == "History":
    st.subheader("📜 Prediction History")
    cursor.execute(
        "SELECT predicted_result, confidence, created_at FROM predictions WHERE username=:u ORDER BY created_at DESC",
        {'u': st.session_state['username']}
    )
    history = cursor.fetchall()
    if history:
        df_history = pd.DataFrame(history, columns=["Result","Confidence","Time"])
        st.dataframe(df_history)
    else:
        st.info("No history found.")

# -------------------------------
# Model Performance Page
# -------------------------------
elif choice == "Model Performance":
    st.subheader("📈 Model Performance")
    try:
        if "uploaded_df" in st.session_state:
            df = st.session_state["uploaded_df"]
        else:
            df = pd.read_csv("water_potability.csv")  # fallback

        X = df.drop("Potability", axis=1).fillna(df.mean())
        y = df["Potability"]

        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        st.write(f"**✅ Accuracy: {acc:.2f}**")
        st.text("Classification Report:")
        st.text(classification_report(y, y_pred))

        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

        st.subheader("ROC CURVE")
        y_prob = model.predict_proba(X)[:,1]
        fpr,tpr,_ = roc_curve(y,y_prob)
        roc_auc = auc(fpr,tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr,tpr,label=f"AUC = {roc_auc:.2f}")
        ax.plot([0,1],[0,1], linestyle="--")
        ax.legend()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"❌ Could not evaluate model: {e}")

# -------------------------------
# Logout Page
# -------------------------------
elif choice == "Logout":
    st.session_state['logged_in'] = False
    st.session_state['username'] = None
    st.success("✅ You have been logged out.")
 