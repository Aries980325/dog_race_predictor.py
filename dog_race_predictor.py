import streamlit as st
import pandas as pd
import numpy as np
import fitz  # PyMuPDF
import re
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Dog Race Predictor", layout="wide")
st.title("🐾 Dog Race Predictor")

# ----------- Helper Functions ---------------

@st.cache_data
def extract_form_data(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()

    dogs = []
    dog_blocks = re.split(r'\n(?=\d+\n[A-Za-z ]+\(\d\))', text)
    for block in dog_blocks:
        try:
            trap_match = re.search(r'^(\d+)', block)
            name_match = re.search(r'\n([A-Za-z ]+)\((\d)\)', block)
            trainer_match = re.search(r'Trainer: (.*?)\n', block)
            stats_match = re.search(r'Career\n(\d+): (\d+)-(\d+)-(\d+)', block)

            if trap_match and name_match and trainer_match and stats_match:
                trap = int(trap_match.group(1))
                name = name_match.group(1).strip()
                box = int(name_match.group(2))
                trainer = trainer_match.group(1).strip()
                total_runs = int(stats_match.group(1))
                wins = int(stats_match.group(2))
                seconds = int(stats_match.group(3))
                thirds = int(stats_match.group(4))

                dogs.append({
                    "Trap": trap,
                    "Name": name,
                    "Box": box,
                    "Trainer": trainer,
                    "Runs": total_runs,
                    "Wins": wins,
                    "Seconds": seconds,
                    "Thirds": thirds
                })
        except Exception as e:
            st.warning(f"Error parsing block:\n{block}\nError: {e}")
    return pd.DataFrame(dogs)

def create_features(df):
    df["WinRate"] = df["Wins"] / df["Runs"]
    df["PlaceRate"] = (df["Wins"] + df["Seconds"] + df["Thirds"]) / df["Runs"]
    df["BoxBias"] = df["Box"].apply(lambda x: 1.2 if x in [1, 2, 3] else 1.0)
    df["TrapAdjustedWin"] = df["WinRate"] * df["BoxBias"]
    return df

@st.cache_data
def train_dummy_model():
    np.random.seed(42)
    df = pd.DataFrame({
        "TrapAdjustedWin": np.random.rand(200),
        "PlaceRate": np.random.rand(200),
        "Target": np.random.randint(0, 2, 200)
    })
    X = df.drop("Target", axis=1)
    y = df["Target"]
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)
    return model

# ----------- App UI -------------------

uploaded_file = st.file_uploader("📄 Upload your Dog Race Form Guide (PDF)", type="pdf")

if uploaded_file:
    data = extract_form_data(uploaded_file)
    if data.empty:
        st.error("Could not extract any valid dog info from this PDF.")
    else:
        st.subheader("📋 Parsed Runners")
        st.dataframe(data)

        features = create_features(data)
        model = train_dummy_model()
        X = features[["TrapAdjustedWin", "PlaceRate"]]
        features["Win_Probability"] = model.predict_proba(X)[:, 1]
        prediction = features.sort_values(by="Win_Probability", ascending=False).reset_index(drop=True)

        st.subheader("📈 Predicted Finish Order")
        places = ["🥇 1st", "🥈 2nd", "🥉 3rd", "4th", "5th", "6th"]
        for i, row in prediction.iterrows():
            st.write(f"{places[i]} — {row['Name']} (Box {row['Box']}) | Win Chance: {row['Win_Probability']:.2%}")

        if len(prediction) >= 3:
            top1, top2, top3 = prediction.iloc[0], prediction.iloc[1], prediction.iloc[2]
            st.subheader("💡 Bet Recommendations")
            st.markdown(f"✅ **Win Bet:** {top1['Name']} (Box {top1['Box']})")
            st.markdown(f"✅ **Exacta:** {top1['Name']} > {top2['Name']}")
            st.markdown(f"✅ **Trifecta:** {top1['Name']} > {top2['Name']} > {top3['Name']}")

        # --- Actual Results Input ---
        st.subheader("📝 Enter Actual Race Results (Optional)")
        real_results = []
        for i in range(len(prediction)):
            dog = st.selectbox(f"{places[i]} place", prediction["Name"], key=f"actual_{i}")
            real_results.append(dog)

        # Accuracy Check
        if st.button("Compare Prediction to Actual Results"):
            predicted_top3 = prediction["Name"].head(3).tolist()
            actual_top3 = real_results[:3]
            correct = sum([1 for dog in predicted_top3 if dog in actual_top3])
            st.success(f"✅ Predicted {correct}/3 of top 3 finishers correctly.")
