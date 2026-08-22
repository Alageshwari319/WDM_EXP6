### EX6 Information Retrieval Using Vector Space Model in Python
### DATE: 18.08.2026
### AIM: To implement Information Retrieval Using Vector Space Model in Python.
### Description: 
<div align = "justify">
Implementing Information Retrieval using the Vector Space Model in Python involves several steps, including preprocessing text data, constructing a term-document matrix, 
calculating TF-IDF scores, and performing similarity calculations between queries and documents. Below is a basic example using Python and libraries like nltk and 
sklearn to demonstrate Information Retrieval using the Vector Space Model.

### Procedure:
1. Define sample documents.
2. Preprocess text data by tokenizing, removing stopwords, and punctuation.
3. Construct a TF-IDF matrix using TfidfVectorizer from sklearn.
4. Define a search function that calculates cosine similarity between a query and documents based on the TF-IDF matrix.
5. Execute a sample query and display the search results along with similarity scores.

### Program:
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import nltk

# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')


# Sample documents
documents = {
    "doc1": "This is the first document.",
    "doc2": "This document is the second document.",
    "doc3": "And this is the third one.",
    "doc4": "Is this the first document?",
}


# Preprocessing function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())

    tokens = [
        token for token in tokens
        if token not in stopwords.words("english")
        and token not in string.punctuation
    ]

    return " ".join(tokens)


# Preprocess documents
preprocessed_docs = {
    doc_id: preprocess_text(doc)
    for doc_id, doc in documents.items()
}


# Create TF-IDF matrix
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(
    preprocessed_docs.values()
)


# Search function
def search(query, tfidf_matrix, tfidf_vectorizer):

    # Preprocess query
    processed_query = preprocess_text(query)

    # Convert query into TF-IDF vector
    query_vector = tfidf_vectorizer.transform([processed_query])

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    # Store document IDs
    doc_ids = list(documents.keys())

    # Create results
    results = []

    for i, score in enumerate(similarity_scores):
        results.append(
            (
                doc_ids[i],
                documents[doc_ids[i]],
                score
            )
        )

    # Sort according to similarity score
    results.sort(key=lambda x: x[2], reverse=True)

    return results


# Get query from user
query = input("Enter your query: ")


# Perform search
search_results = search(
    query,
    tfidf_matrix,
    tfidf_vectorizer
)


# Display results
print("\nQuery:", query)

for i, result in enumerate(search_results, start=1):

    print(f"\nRank: {i}")
    print("Document ID:", result[0])
    print("Document:", result[1])
    print("Similarity Score:", result[2])
    print("----------------------")


# Get highest cosine similarity score
highest_rank_score = max(
    result[2] for result in search_results
)

print(
    "\nThe highest rank cosine score is:",
    highest_rank_score
)
```

### Output:
<img width="1145" height="632" alt="image" src="https://github.com/user-attachments/assets/690e7984-6e01-4d6e-9f93-e3cfa024908a" />

### Result:
Thus, the implementation of Information Retrieval Using Vector Space Model in Python is executed successfully.
