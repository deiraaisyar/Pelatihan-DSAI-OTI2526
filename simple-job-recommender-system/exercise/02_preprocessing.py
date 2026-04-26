import pandas as pd
import re
import nltk
from deep_translator import GoogleTranslator
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

input_path = "./data/jobstreet_data_{TEMPAT}_{NAMA_JOB}.csv"
df = pd.read_csv(input_path)

translator = GoogleTranslator(source="auto", target="en")
stopwords_en = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def translate_text(text):
    if pd.isna(text):
        return ""
    try:
        return translator.translate(text)
    except:
        return text

def clean_text(text):
    if pd.isna(text):
        return ""
    
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    tokens = [word for word in tokens if word not in stopwords_en]
    tokens = [word for word in tokens if len(word) > 2]

    return " ".join(tokens)

print("Translating descriptions...")
df["description_translated"] = df["description"].apply(translate_text)

print("Cleaning descriptions...")
df["description_cleaned"] = df["description_translated"].apply(clean_text)

output_path = "./data/jo bstreet_data_{TEMPAT}_{NAMA_JOB}_cleaned  .csv"
df.to_csv(output_path, index=False)

print("\nORIGINAL:")
print(df["description"].iloc[0][:300])

print("\nTRANSLATED:")
print(df["description_translated"].iloc[0][:300])

print("\nCLEANED:")
print(df["description_cleaned"].iloc[0][:300])

print("\nCleaned dataset saved to:", output_path)
print("Total rows processed:", len(df))