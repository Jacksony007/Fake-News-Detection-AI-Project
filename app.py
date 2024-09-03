import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import shap

# Load NLTK resources
import nltk
nltk.download('stopwords')

# Load models
port_stem = PorterStemmer()
vector_form = pickle.load(open('vector.pkl', 'rb'))
load_model = pickle.load(open('model.pkl', 'rb'))

# Function to preprocess text
def stemming(content):
    con = re.sub('[^a-zA-Z]', ' ', content)
    con = con.lower()
    con = con.split()
    con = [port_stem.stem(word) for word in con if not word in stopwords.words('english')]
    con = ' '.join(con)
    return con

# Function to predict fake news
def fake_news(news):
    news = stemming(news)
    input_data = [news]
    vector_form1 = vector_form.transform(input_data)
    prediction = load_model.predict(vector_form1)
    confidence = load_model.predict_proba(vector_form1).max()
    return prediction[0], confidence

# Modern CSS style
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

# Main function
if __name__ == '__main__':
    st.title('Social Media Scam Detector')
    st.subheader("Enter the News content below")
    
    # Text input for user to enter news content
    sentence = st.text_area("", height=200)

    # Prediction button
    predict_btt = st.button("Predict")

    # Clear instructions
    st.markdown("Please enter the news content in the box below and click 'Predict'.")

    # Loading animation
    if predict_btt:
        with st.spinner('Predicting...'):
            prediction_class, confidence = fake_news(sentence)
        st.success('Prediction complete!')

    # Error handling
    try:
        if predict_btt:
            prediction_class, confidence = fake_news(sentence)
            if prediction_class == 'REAL':
                st.success('Reliable')
            elif prediction_class == 'FAKE':
                st.error('Unreliable')
    except Exception as e:
        st.error(f"An error occurred: {e}")

    # Word Cloud Visualization
    if sentence:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(sentence)
        st.image(wordcloud.to_array(), caption='Word Cloud', use_column_width=True)

    # Confidence Score Display
    if predict_btt:
        st.write(f"Confidence Score: {confidence:.2f}")

    # Explanation of Prediction
    if predict_btt:
        st.markdown("### Explanation")
        if prediction_class == 'REAL':
            st.write("The prediction model indicates that the news content is considered reliable.")
        elif prediction_class == 'FAKE':
            st.write("The prediction model indicates that the news content is considered unreliable.")



    # Advanced Visualization: Prediction Results Bar Chart
    if predict_btt:
        st.write("### Prediction Results Bar Chart")
        prediction_counts = {'Real': 0, 'Fake': 0}  # Initialize both keys with zero count
        if prediction_class == 'REAL':
            prediction_counts['Real'] += 1
        elif prediction_class == 'FAKE':
            prediction_counts['Fake'] += 1
        fig, ax = plt.subplots()
        ax.bar(prediction_counts.keys(), prediction_counts.values())
        st.pyplot(fig)

    # Advanced Visualization: Prediction Results Line Chart
    if predict_btt:
        st.write("### Prediction Results Line Chart")
        prediction_counts = {'Real': 0, 'Fake': 0}  # Initialize both keys with zero count
        if prediction_class == 'REAL':
            prediction_counts['Real'] += 1
        elif prediction_class == 'FAKE':
            prediction_counts['Fake'] += 1
        fig, ax = plt.subplots()
        ax.plot(prediction_counts.keys(), prediction_counts.values(), marker='o', linestyle='-')
        st.pyplot(fig)


    # Social Media Share Button
    if predict_btt:
        st.write("### Share Prediction Result")
        share_message = f"I used the Social Media Scam Detector to analyze this news content. The prediction result is {prediction_class.lower()} with a confidence score of {confidence:.2f}. #ScamDetector"
        st.text_area("Share on Social Media", value=share_message, height=100)

    # Scam Prevention Tips
    if predict_btt:
        st.write("### Scam Prevention Tips")
        if prediction_class == 'FAKE':
            st.write("Here are some tips to prevent falling for scams:")
            st.write("- Always verify information from credible sources.")
            st.write("- Avoid clicking on suspicious links or downloading unknown files.")
            st.write("- Report scams to the relevant authorities or social media platforms.")

    # Language Selector (Placeholder)
    languages = ['English', 'Spanish', 'French']
    selected_language = st.selectbox("Select Language", languages)

    # Responsive Layout
    # (Streamlit automatically handles responsive layout)

    # Dark Mode Toggle (Placeholder)
    dark_mode = st.checkbox("Dark Mode")
    if dark_mode:
        st.markdown("<style>body {background-color: #1a1a1a; color: white;}</style>", unsafe_allow_html=True)
