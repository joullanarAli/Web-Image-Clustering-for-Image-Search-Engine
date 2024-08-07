import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation


class PreprocessData():

    def __init__(self):
        pass
    def tokenize_data(self,text):
        # Verify that the 'punkt' tokenizer data is available
        try:
            nltk.data.find('tokenizers/punkt')
            print("Punkt tokenizer data is available.")
        except LookupError:
            print("Punkt tokenizer data is not available. Please check your setup.")
        tokens = word_tokenize(text)
        return tokens

    def remove_punkt(self, tokens):
        # Remove punctuation from tokens
        tokens_without_punkt = [token for token in tokens if token not in punctuation]
        #print("Tokens without punctuation:", tokens_without_punkt)
        return tokens_without_punkt
    
    def to_lower(self, text):
        text = text.lower()
        return text

    def preprocess_text(self,text):


        # Add your local nltk_data path
        nltk_data_path = 'D:\\nltk_data'  
        if nltk_data_path not in nltk.data.path:
            nltk.data.path.append(nltk_data_path)
        # Convert to lowercase
        text = self.to_lower(text)

        # Tokenize the text
        tokens = word_tokenize(text)
        
        # Remove punctuation
        tokens = self.remove_punkt(tokens)
        
        # Remove stop words
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        
        try:
            nltk.data.find('corpora/omw-1.4')
        except LookupError:
            print(f"The resource omw-1.4 not found in {nltk_data_path}")
        else:
            print(f"The resource omw-1.4 was found in {nltk_data_path}")

        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        
        # Join tokens back to string
        preprocessed_text = ' '.join(tokens)
        
        return preprocessed_text



