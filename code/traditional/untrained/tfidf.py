import re
import csv
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import jieba

def tokenize_chinese(text):
    return " ".join(jieba.cut(text))

#law index: id / context
with open("law.json", "r") as f:
    law_index = json.load(f)
article_tokens = [tokenize_chinese(article["context"]).split() for article in law_index]
contexts = [article["context"] for article in law_index]

# query: id / title / question
with open("test_data.json", "r") as f:
    queries = json.load(f)

thresh = 0.2

def retrieve(title, question, top_n=5):
    title_tok = tokenize_chinese(title.strip())
    if question:
        que_tok = tokenize_chinese(question.strip())

    full = " ".join(title_tok)
    if question:
        full += " " + " ".join(que_tok)
    
    candidate = []
    # exact match
    match = re.search(r"(\w+法第\d+-?\d*條)", title)
    if match:
        article_id = match.group(1)
        exact_matches = [article for article in law_index if article["id"] == article_id]
        if exact_matches:
            candidate.append(article_id)
    
    # keyword match
    title_keywords = title_tok.split()
    question_keywords = que_tok.split() if question else []
    all_keywords = set(title_keywords + question_keywords)

    keyword_matches = []
    for context, article in zip(article_tokens, law_index):
        match_count = sum(1 for keyword in all_keywords if keyword in context)
        if match_count > 0:
            keyword_matches.append((article["id"], match_count))

    # tf-idf Similarity
    vectorizer = TfidfVectorizer().fit([full] + contexts)
    query_vector = vectorizer.transform([full])
    context_vectors = vectorizer.transform(contexts)

    similarities = cosine_similarity(query_vector, context_vectors).flatten()
    similarity_matches = [(law_index[i]["id"], similarities[i]) for i in range(len(law_index)) if similarities[i] > thresh]

    # Stage 4: Combine and Rank Results
    combined_results = []
    for article_id, count in keyword_matches:
        tfidf_score = next((sim for sim_id, sim in similarity_matches if sim_id == article_id), 0)
        final_score = 0.3 * count + 0.7 * tfidf_score
        combined_results.append((article_id, final_score))

    combined_results = sorted(combined_results, key=lambda x: -x[1])

    candidate.extend([combined[0] for combined in combined_results[:top_n]])
    return candidate[:top_n]

with open('results/tfidf.csv', 'w', newline='') as csvfile:
    fieldnames = ['id', 'TARGET']
    writer = csv.writer(csvfile)
    writer.writerow(fieldnames)

    for query in tqdm(queries):
        q_id = query["id"]
        title = query["title"]
        question = query["question"]

        relate = retrieve(title, question, top_n=5)
        ans = ",".join(relate).strip()
        row = [q_id, ans]
        writer.writerow(row)
