import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌱",
    layout="centered"
)

st.title("🌾 Sasya-Nirdesan Web App")
st.markdown(
    "<h5 style='color:red;'>Machine Learning-Based Crop Recommendation System</h5>",
    unsafe_allow_html=True
)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("crop_recommendation_no_outliers.csv")

df = load_data()

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# --------------------------------------------------
# Data Preprocessing
# --------------------------------------------------
le = LabelEncoder()
y = le.fit_transform(df["Crop"])
X = df.drop("Crop", axis=1)

# --------------------------------------------------
# Train-Test Split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# Model Training
# --------------------------------------------------
@st.cache_resource
def train_model():
    model = XGBClassifier(
        objective="multi:softprob",
        n_estimators=400,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss"
    )
    model.fit(X_train, y_train)
    return model

model = train_model()

# --------------------------------------------------
# Model Evaluation
# --------------------------------------------------
st.subheader("📈 Model Performance")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.success(f"✅ Model Accuracy: **{accuracy:.2f}**")

report = classification_report(
    y_test,
    y_pred,
    target_names=le.classes_,
    output_dict=True
)
st.dataframe(pd.DataFrame(report).transpose())

# --------------------------------------------------
# User Input Section
# --------------------------------------------------
st.subheader("🌱 Predict Suitable Crops")

input_data = {}
for col in X.columns:
    input_data[col] = st.number_input(
        f"Enter {col}",
        value=float(X[col].mean())
    )

input_df = pd.DataFrame([input_data])

# --------------------------------------------------
# Top-3 Crop Recommendation WITH CONFIDENCE (NO TABLE)
# --------------------------------------------------
if st.button("🔍 Predict Crop"):
    probabilities = model.predict_proba(input_df)[0]

    top3_indices = probabilities.argsort()[-3:][::-1]
    top3_crops = le.inverse_transform(top3_indices)
    top3_probs = probabilities[top3_indices]

    st.subheader("🌾 Top 3 Recommended Crops")

    for i, (crop, prob) in enumerate(zip(top3_crops, top3_probs), start=1):
        st.success(f"**{i}. {crop}** — Confidence: **{prob * 100:.2f}%**")
