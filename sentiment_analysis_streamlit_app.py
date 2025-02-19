import nltk
import streamlit as st
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nltk.download('punkt')
nltk.download('stopwords')

st.title("Text Analysis App")
st.write("Enter a paragraph of text, and the app will:")
st.write("- Show the frequency distribution of cleaned words")
st.write("- Generate a word cloud")
st.write("- Perform sentiment analysis using VADER")

text = st.text_area("Enter text here", height = 200)

if text.strip() == "":
    st.write("Please enter some text to analyze.")
else:
    #tokenize words
    words = word_tokenize(text)

    #remove punctuation and convert to lowercase
    words_nopunc = [w.lower() for w in words if w.isalpha()]

    #remove stopwords
    stopwords_list = stopwords.words('english')

    words_clean = [w for w in words_nopunc if w not in stopwords_list]

    #frequency distribution of cleaned words
    fdist_clean = FreqDist(words_clean)

    st.subheader("Frequency Distribution of Cleaned Words")

    top_words = fdist_clean.most_common(10) #get top 10 words

    words, frequencies = zip(*top_words) #unzip into two lists

    fig, ax = plt.subplots()

    ax.bar(words, frequencies)

    ax.set_xlabel('Words')

    ax.set_ylabel('Frequency')

    ax.set_title('Top 10 Words (No Punctuation, No Stopwords)')

    plt.xticks(rotation=45)

    st.pyplot(fig)

    st.subheader("Word Cloud (Cleaned Text):")
    
    cleaned_text = " ".join(words_clean)

    wordcloud_clean = WordCloud(width = 800, height = 400, background_color ='white').generate(cleaned_text)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.imshow(wordcloud_clean, interpolation='bilinear')

    ax.axis('off')

    st.pyplot(fig)

    st.subheader("Sentiment Analysis")

    analyzer = SentimentIntensityAnalyzer()
    
    sentiment_score = analyzer.polarity_scores(text)

    st.write("VADER Sentiment Scores:", sentiment_score)

    #interpret sentiment scores
    if sentiment_score['compound'] >= 0.05:
        st.write("Overall Sentiment: Positive🟢")
    elif sentiment_score['compound'] <= -0.05:
        st.write("Overall Sentiment: Negative🔴")
    else:
        st.write("Overall Sentiment: Neutral🟠")