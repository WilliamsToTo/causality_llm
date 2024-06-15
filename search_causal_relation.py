import json
import argparse
import random
import spacy
import hf_olmo
import pandas as pd
import torch
import ast
import faiss

from rake_nltk import Rake
from tqdm import tqdm
from wimbd.es import es_init, get_indices, count_documents_containing_phrases, get_documents_containing_phrases
import elasticsearch
from elasticsearch import Elasticsearch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer


tqdm.pandas()


def count_causal_relation_occurrence(cause, effect, es, index_name):
    print(f"{cause} causes {effect}")
    templates = [f"{cause} causes {effect}", f"{effect} is caused by {cause}", f"{cause} leads to {effect}",
                 f"{cause} results in {effect}", f"{cause} triggers {effect}", f"{effect} is triggered by {cause}",
                 f"{cause} induces {effect}", f"{cause} influences {effect}", f"{effect} is influenced by {cause}",
                 f"{cause} affects {effect}", f"{effect} is affected by {cause}", f"{cause} impacts {effect}",
                 f"{cause} is impacted by {effect}", f"{cause} is responsible for {effect}",
                 f"{cause} is the reason for {effect}", f"The effect of {cause} is {effect}",
                 f"The result of {cause} is {effect}", f"The consequence of {cause} is {effect}",
                 f"{effect} is a consequence of {cause}", f"{effect} is a result of {cause}", f"{effect} is an effect of {cause}",]

    should_list = []
    for phrase in templates:
        match_phrase = {
            "match_phrase": {
                "text": {
                    "query": phrase,
                    "slop": int(len(phrase.split())*0.25), #"slop": int(len(phrase.split())*0.25)
                }
            }
        }
        should_list.append(match_phrase)
    query = {
        "bool": {
            "should": should_list,
            "minimum_should_match": 1
        }
    }
    #print(query)
    total_search_number = es.count(index=index_name, query=query)['count']

    print("total_search_number: ", total_search_number)
    return total_search_number

def count_events_co_occurrence(cause, effect, es, index_name, paraphrase_model, paraphrase_tokenizer):
    cause = cause.strip()
    effect = effect.strip()
    print(f"{cause} and {effect}")
    # paraphrase
    paraphrase_templates = []
    if paraphrase_model is not None:
        cause_papraphrases = olmo_generation(paraphrase_model, paraphrase_tokenizer, cause, max_new_tokens=8, num_return_sequences=2)
        effect_papraphrases = olmo_generation(paraphrase_model, paraphrase_tokenizer, effect, max_new_tokens=8, num_return_sequences=2)
        for cas in cause_papraphrases:
            for eft in effect_papraphrases:
                paraphrase_templates.append((cas, eft))

    if len(paraphrase_templates) > 0:
        paraphrase_templates = [(cause, effect)] + paraphrase_templates
        print(paraphrase_templates)
        paraphrase_search_number = 0
        for tuple in paraphrase_templates:
            cause_clauses = []
            for item in tuple[0].strip().split():
                cause_clauses.append({"span_term": {"text": item}})
            effect_clauses = []
            for item in tuple[1].strip().split():
                effect_clauses.append({"span_term": {"text": item}})
            query = {
                "span_near": {
                    "clauses": [
                        {
                            "span_near": {
                                "clauses": cause_clauses,
                                "slop": 0,
                                "in_order": True
                            }
                        },
                        {
                            "span_near": {
                                "clauses": effect_clauses,
                                "slop": 0,
                                "in_order": True
                            }
                        }
                    ],
                    "slop": 32,
                    "in_order": True
                }
            }
            paraphrase_search_number += es.count(index=index_name, query=query)['count']

        print("paraphrase_search_number: ", paraphrase_search_number)
        return  paraphrase_search_number
    else:
        cause_clauses = []
        for item in cause.split():
            cause_clauses.append({"span_term": {"text": item}})
        effect_clauses = []
        for item in effect.split():
            effect_clauses.append({"span_term": {"text": item}})
        query = {
            "span_near": {
                "clauses": [
                    {
                        "span_near": {
                            "clauses": cause_clauses,
                            "slop": 0,
                            "in_order": True
                        }
                    },
                    {
                        "span_near": {
                            "clauses": effect_clauses,
                            "slop": 0,
                            "in_order": True
                        }
                    }
                ],
                "slop": 32,
                "in_order": True
            }
        }
        # print(query)
        total_search_number = es.count(index=index_name, query=query)['count']
        print("total_search_number: ", total_search_number)
        return total_search_number

def count_events_cause_co_occurrence(cause, effect, es, index_name, paraphrase_model, paraphrase_tokenizer):
    cause = cause.strip()
    effect = effect.strip()
    print(f"{cause} causes {effect}")
    causal_mentions = ["causes", "leads to", "results in", "triggers", "induces", "influences", "affects", "impacts",
                       "is responsible for", "is the reason for", "cause", "lead to", "result in", "trigger", "induce",
                       "influence", "affect", "impact", "are responsible for", "are the reason for"]
    # paraphrase
    paraphrase_templates = []
    if paraphrase_model is not None:
        cause_papraphrases = olmo_generation(paraphrase_model, paraphrase_tokenizer, cause, max_new_tokens=8, num_return_sequences=2)
        effect_papraphrases = olmo_generation(paraphrase_model, paraphrase_tokenizer, effect, max_new_tokens=8, num_return_sequences=2)
        for cas in cause_papraphrases:
            for eft in effect_papraphrases:
                paraphrase_templates.append((cas, eft))

    if len(paraphrase_templates) > 0:
        paraphrase_templates = [(cause, effect)] + paraphrase_templates
        print(paraphrase_templates)
        paraphrase_search_number = 0
        for tuple in paraphrase_templates:
            cause_clauses = []
            for item in tuple[0].strip().split():
                cause_clauses.append({"span_term": {"text": item}})
            effect_clauses = []
            for item in tuple[1].strip().split():
                effect_clauses.append({"span_term": {"text": item}})
            query = {
                "span_near": {
                    "clauses": [
                        {
                            "span_near": {
                                "clauses": cause_clauses,
                                "slop": 0,
                                "in_order": True
                            }
                        },
                        {
                            "span_near": {
                                "clauses": effect_clauses,
                                "slop": 0,
                                "in_order": True
                            }
                        }
                    ],
                    "slop": 32,
                    "in_order": True
                }
            }
            paraphrase_search_number += es.count(index=index_name, query=query)['count']

        print("paraphrase_search_number: ", paraphrase_search_number)
        return  paraphrase_search_number
    else:
        cause_clauses = []
        for item in cause.split():
            cause_clauses.append({"span_term": {"text": item}})
        effect_clauses = []
        for item in effect.split():
            effect_clauses.append({"span_term": {"text": item}})

        all_relation_clauses = []
        for rel in causal_mentions:
            relation_clauses = []
            for term in rel.split():
                relation_clauses.append({"span_term": {"text": term}})
            all_relation_clauses.append(relation_clauses)

        total_search_number = 0
        for relation_clauses in all_relation_clauses:
            query = {
                "span_near": {
                    "clauses": [
                        {
                            "span_near": {
                                "clauses": cause_clauses,
                                "slop": 0,
                                "in_order": True
                            }
                        },
                        {
                            "span_near": {
                                "clauses": relation_clauses,
                                "slop": 0,
                                "in_order": True
                            }
                        },
                        {
                            "span_near": {
                                "clauses": effect_clauses,
                                "slop": 0,
                                "in_order": True
                            }
                        }
                    ],
                    "slop": 32,
                    "in_order": True
                }
            }
            # print(query)
            total_search_number += es.count(index=index_name, query=query)['count']
        print("total_search_number: ", total_search_number)
        return total_search_number

def vector_retrieve_threshold(retrieved_texts, query_texts, model, threshold):
    # Embeddings for the retrieved and query texts
    retrieved_embeddings = model.encode(retrieved_texts)
    query_embeddings = model.encode(query_texts)

    # Normalize embeddings
    faiss.normalize_L2(retrieved_embeddings)
    faiss.normalize_L2(query_embeddings)

    d = retrieved_embeddings.shape[1]

    index = faiss.IndexFlatIP(d)  # Use IndexFlatIP for cosine similarity
    index.add(retrieved_embeddings)  # Add embeddings to index

    similarity_threshold = threshold  # Cosine similarity threshold

    # Perform range search for each query embedding
    lims, D, I = index.range_search(query_embeddings, similarity_threshold)

    unique_indices = set(I)
    num_unique_texts = len(unique_indices)

    return num_unique_texts

def count_relation_vector_retrieve(cause, effect, es, index_name, embed_model, threshold, paraphrase_model, paraphrase_tokenizer):
    cause = cause.strip()
    effect = effect.strip()
    print(f"{cause} causes {effect}")
    query_texts = [f"{cause} causes {effect}", f"{cause} leads to {effect}", f"{cause} results in {effect}",
                 f"{cause} triggers {effect}", f"{cause} induces {effect}", f"{cause} influences {effect}",
                 f"{cause} affects {effect}", f"{cause} impacts {effect}", f"{cause} is responsible for {effect}",
                 f"{cause} is the reason for {effect}", f"The effect of {cause} is {effect}",
                 f"The result of {cause} is {effect}", f"The consequence of {cause} is {effect}",
                 f"{effect} is a consequence of {cause}", f"{effect} is a result of {cause}", f"{effect} is an effect of {cause}",]


    cause_clauses = []
    for item in cause.split():
        cause_clauses.append({"span_term": {"text": item}})
    effect_clauses = []
    for item in effect.split():
        effect_clauses.append({"span_term": {"text": item}})
    query = {
        "span_near": {
            "clauses": [
                {
                    "span_near": {
                        "clauses": cause_clauses,
                        "slop": 0,
                        "in_order": True
                    }
                },
                {
                    "span_near": {
                        "clauses": effect_clauses,
                        "slop": 0,
                        "in_order": True
                    }
                }
            ],
            "slop": 64,
            "in_order": False
        }
    }
    # print(query)
    highlight = {
        "fields": {
            "text": {
                "type": "unified",
                "fragment_size": 0,  # characters length
                "number_of_fragments": 1,
                "boundary_chars": ".!? \t\n"
            }
        }
    }
    total_search_number = es.count(index=index_name, query=query)['count']
    print(total_search_number)
    scroll = '2m'
    result = es.search(index=index_name, query=query, scroll=scroll, highlight=highlight, size=10000)
    scroll_id = result['_scroll_id']

    # Extract hits from the first batch
    retrieved_texts = []
    for doc in result['hits']['hits']:
        if "highlight" in doc:
            retrieved_texts.append(doc["highlight"]["text"][0].replace("<em>", "").replace("</em>", ""))

    # Start scrolling
    while True:
        try:
            # Make a scroll request using the Scroll ID
            result = es.scroll(scroll_id=scroll_id, scroll=scroll)
            print(len(result['hits']['hits']))
            # Break the loop if no more results are returned
            if not result['hits']['hits'] or len(retrieved_texts) > 100000:
                break

            for doc in result['hits']['hits']:
                if "highlight" in doc:
                    retrieved_texts.append(doc["highlight"]["text"][0].replace("<em>", "").replace("</em>", ""))

            # Update the scroll ID
            scroll_id = result['_scroll_id']
        except Exception as e:
            print("Failed to scroll:", e)
            break

    # es.clear_scroll(scroll_id=scroll_id)

    if len(retrieved_texts) > 0:
        vector_retrieve_num = vector_retrieve_threshold(retrieved_texts, query_texts, embed_model, threshold=threshold)
        print(len(retrieved_texts), vector_retrieve_num)
        return vector_retrieve_num
    else:
        return 0


def conceptnet_causal_relation_get_occurrence(read_path, save_path, api_key, index_name):
    df = pd.read_csv(read_path)
    es = es_init(config=api_key, timeout=1000)
    # df = df[70:]
    # load model: prithivida/parrot_paraphraser_on_T5, humarin/chatgpt_paraphraser_on_T5_base, #/data/ccu/taof/olmo/olmo_7B_instruct/
    device = "cuda"
    # tokenizer = AutoTokenizer.from_pretrained("/data/ccu/taof/olmo/olmo_7B_instruct/")
    # model = AutoModelForCausalLM.from_pretrained("/data/ccu/taof/olmo/olmo_7B_instruct/").to(device)

    # df['dolma_causal_co_occurrence'] = df.progress_apply(
    #     lambda row: count_events_cause_co_occurrence(row['cause'], row['effect'], es, index_name, None, None), axis=1)
    # df['dolma_events_co_occurrence'] = df.progress_apply(
    #     lambda row: count_events_co_occurrence(row['cause'], row['effect'], es, index_name, None, None), axis=1)
    df['dolma_causal_match'] = df.progress_apply(
        lambda row: count_causal_relation_occurrence(row['cause'], row['effect'], es, index_name, None, None), axis=1)
    # embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    # df['dolma_vector_retrieve_0.9'] = df.progress_apply(
    #     lambda row: count_relation_vector_retrieve(row["cause"], row["effect"], es, index_name, embed_model=embed_model, threshold=0.9, paraphrase_model=None, paraphrase_tokenizer=None), axis=1)

    df.to_csv(save_path, index=False)


parser = argparse.ArgumentParser(description='search.')

# Add arguments
parser.add_argument('--read_path', type=str, help='file path to the dataset e.g., /home/taof/causal-llm-bfs/data/conceptNet/causes_relation_en_0_10_olmo7bInstruct.csv')
parser.add_argument('--save_path', type=str, help='save path to the dataset')

# Parse the arguments
args = parser.parse_args()

conceptnet_causal_relation_get_occurrence(read_path=args.read_path, save_path=args.save_path,
                                            api_key="your_api_key.yml", index_name="docs_v1.5_2023-11-02")



