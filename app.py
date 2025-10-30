
import re
import io
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from transformers import pipeline

st.set_page_config(page_title="Valorant Chat Toxicity Analyzer", layout="centered")

st.title("Valorant Chat Toxicity Analyzer (Improved Accuracy)")
st.markdown("Upload a chat log or paste text. Uses a refined model with better accuracy for chat messages.")

uploaded = st.file_uploader("Upload chat.txt", type=["txt"])
text_input = st.text_area("Or paste chat text here (each line = one message):", height=200)

data = ""
if uploaded is not None:
    data = uploaded.read().decode('utf-8')
elif text_input.strip():
    data = text_input

if not data:
    st.info("Upload a file or paste text to analyze.")
    st.stop()

def parse_chat(text):
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"\[(.*?)\]\s*(.*?):\s*(.*)", line)
        if m:
            time, player, msg = m.groups()
        else:
            parts = line.split(":", 1)
            if len(parts) == 2:
                player, msg = parts[0].strip(), parts[1].strip()
                time = ""
            else:
                time, player, msg = "", "", line
        rows.append({"time": time, "player": player, "message": msg})
    return pd.DataFrame(rows)

df = parse_chat(data)
st.subheader("Preview of parsed messages")
st.dataframe(df)

with st.spinner("Loading improved toxicity model..."):
    classifier = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-offensive", return_all_scores=True)

st.success("Model loaded successfully!")

if st.button("Analyze messages"):
    results = []
    for msg in df["message"].fillna("").tolist():
        msg_clean = re.sub(r'[^a-zA-Z ]', '', msg.lower()).strip()
        if not msg_clean:
            results.append({"label": None, "score": 0.0, "toxic": False})
            continue
        out = classifier(msg_clean)[0]
        best = max(out, key=lambda x: x["score"])
        label, score = best["label"], best["score"]
        toxic = (label.lower() in ["offensive", "toxic"]) and (score > 0.6)
        results.append({"label": label, "score": score, "toxic": toxic})

    res_df = pd.DataFrame(results)
    df2 = pd.concat([df.reset_index(drop=True), res_df], axis=1)

    st.subheader("Analysis Results")
    st.dataframe(df2)

    total = len(df2)
    tox = int(df2["toxic"].sum())
    st.markdown(f"**Total messages analyzed:** {total}")
    st.markdown(f"**Toxic messages:** {tox} ({(tox/total*100) if total>0 else 0:.1f}%)")

    if "player" in df2.columns:
        ranking = df2[df2["toxic"] & df2["player"].notna() & (df2["player"]!='')].groupby("player").size().sort_values(ascending=False)
        if not ranking.empty:
            st.subheader("Top Toxic Players")
            st.table(ranking.reset_index().rename(columns={0:"toxic_count", "player":"player"}))

    st.subheader("Toxic vs Non-toxic")
    fig, ax = plt.subplots(figsize=(5,3))
    df2['toxic'].value_counts().plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'])
    ax.set_xlabel("Toxic")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    csv = df2.to_csv(index=False, encoding='utf-8-sig')
    st.download_button("Download analysis CSV", csv, file_name="valorant_toxicity_results.csv", mime="text/csv")
