import streamlit as st
import pandas as pd
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

model = pickle.load(open("logistic_model.pkl", "rb"))
le_color = pickle.load(open("label_encoder_color.pkl", "rb"))
le_cat = pickle.load(open("label_encoder_cat.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
cosine_sim = pickle.load(open("cosine_similarity_matrix.pkl", "rb"))
df = pd.read_csv("processed_handm.csv")

st.set_page_config(page_title="Sustainable Fashion Recommender", layout="wide")
st.title("H&M Sustainable Fashion Recommender")

st.sidebar.header("Filter Options")
price = st.sidebar.slider("Price", float(df.price.min()), float(df.price.max()), float(df.price.mean()))
color = st.sidebar.selectbox("Color", sorted(df.colorName.unique()))
cat = st.sidebar.selectbox("Category", sorted(df.mainCatCode.unique()))
top_n = st.sidebar.slider("Recommendations", 1, 10, 5)

color_enc = le_color.transform([color])[0]
cat_enc = le_cat.transform([cat])[0]

user_input = pd.DataFrame([[price, color_enc, cat_enc]], columns=['price', 'colorName', 'mainCatCode'])
prediction = model.predict(user_input)[0]

st.write("Sustainability Prediction:", "Yes" if prediction == 1 else "No")

df['combined_text'] = df['productName'] + " " + df['details']

def recommend(product_id, top_n=5, sustainable_only=True):
    if product_id not in df['productId'].values:
        return pd.DataFrame()

    idx = df[df['productId'] == product_id].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:]
    if sustainable_only:
        sim_scores = [i for i in sim_scores if df.iloc[i[0]]['predicted_sustainability'] == 1]
    indices = [i[0] for i in sim_scores[:top_n]]
    return df.iloc[indices][['productId', 'productName', 'price', 'details']]

st.subheader("🔎Similar Sustainable Products")
product_id = st.text_input("Enter a Product ID to get Recommendations", value="1258600003")
if st.button("Recommend"):
    recs = recommend(int(product_id), top_n=top_n)
    if not recs.empty:
        st.dataframe(recs)
    else:
        st.warning("Product not found or no similar sustainable items.")
st.subheader("📊 Data Insights")

tab1, tab2 = st.tabs(["Sustainability Ratio", "Price Distribution"])

with tab1:
    fig, ax = plt.subplots()
    sns.countplot(x="is_sustainable", data=df, ax=ax, palette="coolwarm")
    ax.set_xticklabels(['Not Sustainable', 'Sustainable'])
    st.pyplot(fig)

with tab2:
    fig2, ax2 = plt.subplots()
    sns.histplot(df['price'], bins=30, kde=True, ax=ax2)
    ax2.set_title("Price Distribution")
    st.pyplot(fig2)