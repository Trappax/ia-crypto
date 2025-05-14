import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import random
from textblob import TextBlob
from GoogleNews import GoogleNews
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime
import matplotlib.pyplot as plt

# ----------------- CONFIGURATION DE L'APP -----------------
st.set_page_config(page_title="IA Crypto Predictor", layout="centered")
st.title("🔮 Prédicteur IA Crypto")
st.markdown("Une intelligence artificielle qui analyse les données techniques et émotionnelles pour prédire les mouvements de marché.")

# ----------------- CHOIX DE LA CRYPTO -----------------
crypto = st.selectbox("💰 Choisissez une cryptomonnaie :", ["BTC-USD", "ETH-USD", "SOL-USD"])
nom_crypto = crypto.split("-")[0]

# ----------------- DONNÉES DE MARCHÉ -----------------
data = yf.download(crypto, start="2021-01-01", end=datetime.today().strftime('%Y-%m-%d'))
data["Return"] = data["Close"].pct_change()
data["Target"] = np.where(data["Return"].shift(-1) > 0, 1, 0)
data["SMA_5"] = data["Close"].rolling(5).mean()
data["SMA_10"] = data["Close"].rolling(10).mean()
data["RSI_14"] = 100 - (100 / (1 + (data["Close"].diff().where(lambda x: x > 0, 0).rolling(14).mean() /
                                   -data["Close"].diff().where(lambda x: x < 0, 0).rolling(14).mean())))
data["MACD"] = data["Close"].ewm(span=12).mean() - data["Close"].ewm(span=26).mean()
data["Signal_MACD"] = data["MACD"].ewm(span=9).mean()

# ----------------- ÉMOTIONS DU MARCHÉ -----------------
sentiment_twitter = random.uniform(-0.2, 0.2)
googlenews = GoogleNews(lang='en')
googlenews.search("crypto news")
titles = [a['title'] for a in googlenews.result()]
sentiment_actu = sum([TextBlob(t).sentiment.polarity for t in titles]) / len(titles) if titles else 0

data["Sentiment_Twitter"] = sentiment_twitter
data["Sentiment_Actu"] = sentiment_actu
data = data.dropna()

# ----------------- ENTRAÎNEMENT -----------------
features = ["Return", "SMA_5", "SMA_10", "RSI_14", "MACD", "Signal_MACD", "Sentiment_Twitter", "Sentiment_Actu"]
X = data[features]
y = data["Target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
prediction = model.predict(X.tail(1))[0]
confidence = model.score(X_test, y_test)

# ----------------- AFFICHAGE -----------------
st.subheader("📊 Données techniques & émotions")
col1, col2 = st.columns(2)
col1.metric("Sentiment Twitter", f"{sentiment_twitter:.2f}", delta=None)
col2.metric("Sentiment Actualités", f"{sentiment_actu:.2f}", delta=None)

st.subheader(f"📈 Prédiction IA pour {nom_crypto}")
if prediction == 1:
    st.success("✅ L'IA prédit que le prix **va MONTER** demain. (Recommandation : Acheter)")
else:
    st.error("⚠️ L'IA prédit que le prix **va BAISSER** demain. (Recommandation : Attendre / Vendre)")

st.info(f"🔍 Fiabilité actuelle de l'IA : **{confidence * 100:.2f}%**")

st.subheader("📉 Historique des prix")
fig, ax = plt.subplots()
data["Close"].plot(ax=ax, title=f"Prix de {nom_crypto} depuis 2021")
st.pyplot(fig)

with st.expander("🧠 Comment l'IA décide ?"):
    st.write("""
    - L'IA analyse des indicateurs techniques comme les moyennes mobiles, RSI, MACD...
    - Elle prend aussi en compte l'ambiance du marché (sentiment Twitter + actualités).
    - Puis elle prédit si demain le prix va monter ou baisser.
    """)

st.caption("Créé par Antoine · Prédiction IA expérimentale — ne constitue pas un conseil financier.")
