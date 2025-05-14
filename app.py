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

st.title("🔮 IA Crypto/Bourse — Prédiction de demain")

# Télécharger les données
btc = yf.download("BTC-USD", start="2020-01-01", end=datetime.today().strftime('%Y-%m-%d'))
btc["Return"] = btc["Close"].pct_change()
btc["Target"] = np.where(btc["Return"].shift(-1) > 0, 1, 0)
btc["SMA_5"] = btc["Close"].rolling(5).mean()
btc["SMA_10"] = btc["Close"].rolling(10).mean()
btc["RSI_14"] = 100 - (100 / (1 + (btc["Close"].diff().where(lambda x: x > 0, 0).rolling(14).mean() /
                                   -btc["Close"].diff().where(lambda x: x < 0, 0).rolling(14).mean())))
btc["MACD"] = btc["Close"].ewm(span=12).mean() - btc["Close"].ewm(span=26).mean()
btc["Signal_MACD"] = btc["MACD"].ewm(span=9).mean()

# Sentiment Twitter (simulé)
sentiment_twitter = random.uniform(-0.2, 0.2)
btc["Sentiment_Twitter"] = sentiment_twitter

# Sentiment actualités économiques
googlenews = GoogleNews(lang='en')
googlenews.search("economic news")
titles = [a['title'] for a in googlenews.result()]
sentiment_actu = sum([TextBlob(t).sentiment.polarity for t in titles]) / len(titles) if titles else 0
btc["Sentiment_Actu"] = sentiment_actu

btc = btc.dropna()
features = ["Return", "SMA_5", "SMA_10", "RSI_14", "MACD", "Signal_MACD", "Sentiment_Twitter", "Sentiment_Actu"]
X = btc[features]
y = btc["Target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Prédiction
dernier = X.tail(1)
prediction = model.predict(dernier)[0]
confidence = model.score(X_test, y_test)

# Affichage
if prediction == 1:
    st.success("📈 L'IA prédit que le Bitcoin VA MONTER demain (Acheter).")
else:
    st.error("📉 L'IA prédit que le Bitcoin VA BAISSER demain (Attendre ou vendre).")

st.write(f"🔍 Confiance de l'IA basée sur l'historique : {confidence*100:.2f}%")
st.write(f"🗞️ Sentiment Twitter (simulé) : {sentiment_twitter:.2f}")
st.write(f"📰 Sentiment Actualités : {sentiment_actu:.2f}")
