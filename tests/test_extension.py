from keybert import KeyBERT
import os
import zipfile
import urllib.request
from tqdm import tqdm
import spacy

import os

# read docs and gold keywords   

docs_dir = os.path.join("500N-KPCrowd-v1.1", "500N-KPCrowd-v1.1/docsutf8")
keys_dir = os.path.join("500N-KPCrowd-v1.1", "500N-KPCrowd-v1.1/keys")
doc_files = sorted(os.listdir(docs_dir))
key_files = sorted(os.listdir(keys_dir))
docs = []
gold_keywords = [] 
for doc_file, key_file in zip(doc_files, key_files):
    with open(os.path.join(docs_dir, doc_file), encoding='utf-8') as f:
        docs.append(f.read())
    with open(os.path.join(keys_dir, key_file), encoding='utf-8') as f:
        gold_keywords.append([line.strip().lower() for line in f if line.strip()])


# Load spaCy and define post-processing functions
nlp = spacy.load("en_core_web_sm")

def pos_filter(keywords):
    filtered = []
    for kw, score in keywords:
        doc = nlp(kw)
        if all(token.pos_ in {"NOUN", "PROPN", "ADJ"} for token in doc):
            filtered.append((kw, score))
    return filtered

def entity_boost(keywords, text):
    doc = nlp(text)
    entities = set(ent.text for ent in doc.ents)
    boosted = []
    for kw, score in keywords:
        if kw in entities:
            boosted.append((kw, score + 0.2))
        else:
            boosted.append((kw, score))
    return boosted


def advanced_postprocess(keywords, doc_text, nlp):
    keywords = entity_boost(keywords, doc_text)
    keywords = pos_filter(keywords)

    return keywords

# Define evaluation metrics
def precision(pred, gold):
    pred_set = set(pred)
    gold_set = set(gold)
    return len(pred_set & gold_set) / len(pred_set) if pred_set else 0

def recall(pred, gold):
    pred_set = set(pred)
    gold_set = set(gold)
    return len(pred_set & gold_set) / len(gold_set) if gold_set else 0

def f1(p, r):
    return 2 * p * r / (p + r) if (p + r) else 0

# Define partial matching metric
def partial_match(pred, gold):
    count = 0
    for p in pred:
        for g in gold:
            if p in g or g in p:
                count += 1
                break
    return count


# Initialize KeyBERT model
kw_model = KeyBERT()



# Extract keywords with and without post-processing for the full dataset
N = 5  # Number of keywords to extract
results_no_post = []
results_post = []

if __name__ == "__main__":
    for doc in tqdm(docs):
        kws_no_post = [kw for kw, _ in kw_model.extract_keywords(doc, top_n=N)]
        kws_post = kw_model.extract_keywords(
            doc, top_n=N,
            postprocess=lambda kws: advanced_postprocess(kws, doc, nlp)
        )
        kws_post = [kw for kw, _ in kws_post]
        results_no_post.append(kws_no_post)
        results_post.append(kws_post)

    # Evaluate and print results
    p_no_post, r_no_post, f1_no_post = [], [], []
    p_post, r_post, f1_post = [], [], []

    pm_no_post, pm_post = [], []


    for pred, gold, pred_post in zip(results_no_post, gold_keywords, results_post):
        # Exact match
        p = precision(pred, gold)
        r = recall(pred, gold)
        f = f1(p, r)
        p_no_post.append(p)
        r_no_post.append(r)
        f1_no_post.append(f)

        p2 = precision(pred_post, gold)
        r2 = recall(pred_post, gold)
        f2 = f1(p2, r2)
        p_post.append(p2)
        r_post.append(r2)
        f1_post.append(f2)
        # Partial match
        pm_no_post.append(partial_match(pred, gold) / len(pred) if pred else 0)
        pm_post.append(partial_match(pred_post, gold) / len(pred_post) if pred_post else 0)

    print("No Post-processing: Precision {:.3f}, Recall {:.3f}, F1 {:.3f}".format(
        sum(p_no_post)/len(p_no_post), sum(r_no_post)/len(r_no_post), sum(f1_no_post)/len(f1_no_post)))
    print("With Post-processing: Precision {:.3f}, Recall {:.3f}, F1 {:.3f}".format(
        sum(p_post)/len(p_post), sum(r_post)/len(r_post), sum(f1_post)/len(f1_post)))
    print("No Post-processing Partial Match: {:.3f}".format(sum(pm_no_post)/len(pm_no_post)))
    print("With Post-processing Partial Match: {:.3f}".format(sum(pm_post)/len(pm_post)))


