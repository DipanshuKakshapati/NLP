import streamlit as st
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
import nltk

nltk.download('punkt')

st.title("Text Summarization App")
st.write("Enter a paragraph of text, and the app will generate a summary.")

input_text = st.text_area("Enter your text here", height=200)

selected_algorithm = st.selectbox("Choose a summarization algorithm", ["LSA (Latent Semantic Analysis)", "LexRank", "Luhn"])

summary_sentences = st.slider("Number of sentences in the summary", 1, 10, 3)

if st.button("Summarize"):
    if input_text.strip() == "":
        st.warning("Please enter some text to summarize.")
    
    else:
        #summarization logic
        
        #initialize the parser and tokenizer
        parser = PlaintextParser.from_string(input_text, Tokenizer("english"))
        
        #initiate the summarizer based on the selected algorithm
        if selected_algorithm == "LSA (Latent Semantic Analysis)":
            summarizer = LsaSummarizer()
        elif selected_algorithm == "LexRank":
            summarizer = LexRankSummarizer()
        elif selected_algorithm == "Luhn":
            summarizer = LuhnSummarizer()
    
        summary = summarizer(parser.document, summary_sentences)
        
        #display the summary
        st.subheader("Summary")
        for sentence in summary:
            st.write(sentence)