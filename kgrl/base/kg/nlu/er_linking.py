import numpy as np
import torch
from pykeen.pipeline import PipelineResult
from typing import Dict, List, Tuple
from rdflib import Graph, URIRef
from sentence_transformers import SentenceTransformer, util
from kgrl.dto.embedding import KGEmbeddingModel
from pykeen.models import TransE

def er_linker(kg: Graph, mission: str) -> Dict[List[URIRef], float]:
    """
    Evaluate knowledge graph triples relevance based on the mission.
    """
    preprocess_uri = lambda x: str(x.split("/")[-1].split("_")[0]) # preprocessing function for uri
    sub_obj_list = list(kg.subject_objects()) # list of subj and corresponding objects to create labels for triples
    triples = [list(kg.triples((subj, None, obj))) for subj, obj in sub_obj_list] # all triples from kg
    triples = list(map(lambda x: x[0], triples))
    obj2subj = [f'{preprocess_uri(pairs[1])} {preprocess_uri(pairs[0])}' for pairs in sub_obj_list]
    label2triple_dict = {k:v for k, v in list(zip(triples, obj2subj))} # creating labels for triples
    obj2subj_unique = list(set(obj2subj))
    words = [word for word in mission.split() if word not in ["the", "a", "an", "to"]] # exclude articles and "to" from the mission words
    two_grams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1) if words[i] and words[i+1]] 
    model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
    two_grams_encoded = model.encode(two_grams)
    words_encoded = model.encode(words)
    obj2subj_encoded = model.encode(obj2subj_unique)
    scores_two_grams = []
    scores_words = []
    for two_gram in two_grams_encoded:
        scores_two_grams.append(util.cos_sim(two_gram, obj2subj_encoded)[0]) # similarity for each two_gram with triple labels
    scores_two_grams = torch.max(torch.stack(scores_two_grams),dim=0).values # consider max similarity for all two_grams 
    for word in words_encoded:
        scores_words.append(util.cos_sim(word, obj2subj_encoded)[0]) # similarity for each word with triple labels
    scores_words = torch.max(torch.stack(scores_words),dim=0).values  # consider max similarity for all words
    all_scores = torch.stack((scores_two_grams,scores_words),dim=0)
    max_scores = torch.max(all_scores, dim=0).values # consider max similarity for two_grams and words
    er_link = {}
    for key, value in label2triple_dict.items():
        score = max_scores[obj2subj_unique.index(value)]
        er_link[key] = float(score)
    er_link = {k:v for k, v in sorted(er_link.items(), key=lambda x: x[1], reverse=True)}
    return er_link

def er_linker_emb(kg: Graph, mission: str, n_triples: int, kg_emb: KGEmbeddingModel, padding: str = "max_sim") -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract embeddings for n most relevant triples. 
    :param kg: knowledge graph for the environment
    :param mission: mission string for the environment 
    :param n_triples: number of triples to consider 
    :param kg_emb: knowledge graph embedding from GraphHandler
    :param padding: padding approach to fit the needed shape
    :return: tuple(embedded triples, confidence scores)
    """
    er_linking_dict = er_linker(kg, mission)
    triple_keys = list(er_linking_dict.keys())[:n_triples] 
    confidence = np.array(list(er_linking_dict.values())[:n_triples])
    kg_ent2id = kg_emb.triples_factory.entity_to_id  # entity to id dict from PipelineResult
    kg_rel2id = kg_emb.triples_factory.relation_to_id  # relation to id dict from PipelineResult
    kg_ent_emb = kg_emb.pipeline_results.model.entity_representations[0]()  # entity_embeddings from PipelineResult
    kg_rel_emb = kg_emb.pipeline_results.model.relation_representations[0]()  # relation_embeddings from PipelineResult
    triples_ids = [[(kg_ent2id[str(key[0])]), (kg_rel2id[str(key[1])]), (kg_ent2id[str(key[2])])] for key in triple_keys]  # ids for entities and relation for each triple
    triples_embs = [[kg_ent_emb[ids[0]].detach().cpu().numpy(), kg_rel_emb[ids[1]].detach().cpu().numpy(), kg_ent_emb[ids[2]].detach().cpu().numpy()] for ids in triples_ids]  # embs for entities and relation for each triple
    emb_dim = triples_embs[0][0].size
    while len(triples_embs) < n_triples:
        if padding == "max_sim":
            triple_keys.append(triple_keys[0])
            triples_embs.append(triples_embs[0])
    er_linker_emb = np.concatenate(triples_embs).reshape((n_triples, 3, emb_dim))
    return (er_linker_emb, confidence)