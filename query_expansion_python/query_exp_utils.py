import nltk
import torch
import itertools

from nltk.corpus import wordnet as wn
from transformers import AutoTokenizer, AutoModelForMaskedLM, BertTokenizer, BertModel
from scipy.spatial.distance import cosine


def _convert_nltk_to_wordnet_tag(pos_tag):
    if pos_tag.startswith("N"):
        return wn.NOUN
    elif pos_tag.startswith("V"):
        return wn.VERB
    # elif pos_tag.startswith("R"):
    #     return wn.ADV
    # elif pos_tag.startswith("J"):
    #     return wn.ADJ
    else:
        return None


def _get_bert_sentence_embedding(tokenizer, model, sentence):

    marked_sentence = "[CLS] " + sentence + " [SEP]"
    tokenized_sentence = tokenizer.tokenize(marked_sentence)                    # Tokenize our sentence with the BERT tokenizer
    segments_ids = [1] * len(tokenized_sentence)                                # Map the token strings to their vocabulary indexes.
    indexed_sentence = tokenizer.convert_tokens_to_ids(tokenized_sentence)      # Mark each query token as belonging to sentence "1"
    sentence_tensor = torch.tensor([indexed_sentence])                          # Convert inputs to PyTorch tensors
    segments_tensor = torch.tensor([segments_ids])

    with torch.no_grad():
        bert_outputs = model(sentence_tensor, segments_tensor)                  # Run the text through BERT
        hidden_states = bert_outputs[2]                                         # Collect all of the hidden states produced from all 12 layers

        # To get a single vector for the entire query (basically a query embedding) we can average the second
        # to last hidden layer of each token producing a single 768 length vector.
        token_vectors = hidden_states[-2][0]
        sentence_embedding = torch.mean(token_vectors, dim=0)

    return sentence_embedding


def generate_similar_queries(input_query, verbose=False):

    # Use first a BERT model to get a list of proposed words in place of masked ones
    mask_tokenizer = AutoTokenizer.from_pretrained("../bert-base-uncased")
    mask_model = AutoModelForMaskedLM.from_pretrained("../bert-base-uncased")

    tokenized_query = mask_tokenizer.tokenize(input_query)
    pos_tags_wn = nltk.pos_tag(tokenized_query)
    if verbose:
        print("Recognized POS tags: ", pos_tags_wn)

    # Use WordNet to find synonyms for each noun in the query
    masked_words_indexes = list()
    for i in range(len(pos_tags_wn)):
        pos_tag_wn = pos_tags_wn[i]
        # We want to replace only nouns with synonyms (all other POS return None)
        converted_pos_tag = _convert_nltk_to_wordnet_tag(pos_tag_wn[1])
        if converted_pos_tag is not None:
            masked_words_indexes.append(i)

    # Generate all the queries with [MASK] special token in place of the nouns
    masked_query_strings_list = list()
    for word_index_to_mask in masked_words_indexes:
        tokenized_query_to_mask = list(tokenized_query)  # Just take a copy not to touch
        tokenized_query_to_mask[word_index_to_mask] = mask_tokenizer.mask_token
        # print("tokenized_masked_query: ", tokenized_query_to_mask)

        masked_query_string = " ".join(tokenized_query_to_mask)
        if verbose:
            print("Masked query: ", masked_query_string)
        masked_query_strings_list.append(masked_query_string)

    # For each masked_query_string generate the top 10 words that fit inside the [MASK] position
    best_tokens_list = list()
    for i in range(len(masked_query_strings_list)):
        masked_query_string = masked_query_strings_list[i]

        encoded_input = mask_tokenizer.encode(masked_query_string, return_tensors="pt")
        mask_token_index = torch.where(encoded_input == mask_tokenizer.mask_token_id)[1]
        token_logits = mask_model(encoded_input).logits
        mask_token_logits = token_logits[0, mask_token_index, :]

        # Get the best 10 predictions from BERT
        top_10_encoded_tokens = torch.topk(mask_token_logits, 10, dim=1).indices[0].tolist()
        top_10_tokens = list((mask_tokenizer.decode(encoded_token)) for encoded_token in top_10_encoded_tokens)

        # Add the original word of the query if not already inside best_tokens
        masked_word_index = masked_words_indexes[i]
        if pos_tags_wn[masked_word_index][0] not in top_10_tokens:
            top_10_tokens.append(pos_tags_wn[masked_word_index][0])

        # Remove all the partial tokens retrieved by BERT (with format "##abcdefgh")
        top_10_tokens_filtered = [token for token in top_10_tokens if not token.startswith("##")]
        if verbose:
            print("Top 10 tokens: ", top_10_tokens_filtered)
        best_tokens_list.append(top_10_tokens_filtered)


    # Build query_synonyms_list with the list of feasible tokens for each position inside the query
    query_synonyms_list = list([None] * len(pos_tags_wn))
    for i in range(len(pos_tags_wn)):
        if i not in masked_words_indexes:                                       # Add a list with just the original word
            query_synonyms_list[i] = list()
            query_synonyms_list[i].append(pos_tags_wn[i][0])
    for j in range(len(masked_words_indexes)):                                  # Add the list of best tokens we just found
        query_synonyms_list[masked_words_indexes[j]] = best_tokens_list[j]

    # Compose all new queries obtained combining the best tokens
    new_queries_list = list(itertools.product(*query_synonyms_list))
    new_queries_strings = list(" ".join(new_query) for new_query in new_queries_list)

    # Use another BERT model and tokenizer to get the query embeddings
    emb_tokenizer = BertTokenizer.from_pretrained('../bert-base-uncased')
    emb_model = BertModel.from_pretrained("../bert-base-uncased", output_hidden_states=True)

    # Compute the embeddings for each new query and compute the cosine similarity to the original one,
    # which is the first query in new_queries_strings, so the first cos_sim should be 1.00
    original_query_embedding = _get_bert_sentence_embedding(emb_tokenizer, emb_model, new_queries_strings[0])

    cos_sim_list = list()
    for new_query in new_queries_strings:
        new_query_embedding = _get_bert_sentence_embedding(emb_tokenizer, emb_model, new_query)
        cos_sim = 1 - cosine(original_query_embedding, new_query_embedding)
        cos_sim_list.append(cos_sim)

        if verbose:
            print(f"{new_query} -> cos similarity score: {cos_sim}")

    # Collect all (new_query, cosine_similarity_score) pairs inside a dictionary
    query_sim_dict = dict(zip(new_queries_strings, cos_sim_list))

    # Prune the previous dictionary keeping only the "best queries", which are the ones with higher scores (80+ %)
    max_cos_sim = 1.0
    min_cos_sim = min(cos_sim_list)
    sim_thresh = min_cos_sim + 0.8 * (max_cos_sim - min_cos_sim)
    best_query_sim_dict = {k: v for k, v in query_sim_dict.items() if v >= sim_thresh}
    best_queries_list = [k for k in best_query_sim_dict.keys()]

    return best_queries_list
