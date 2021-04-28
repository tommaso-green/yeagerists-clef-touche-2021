import nltk
import torch
import itertools
import random
import math

from torch.nn import CosineSimilarity
from nltk.corpus import wordnet as wn
from transformers import AutoTokenizer, AutoModelForMaskedLM, BertTokenizer, BertModel, BertTokenizerFast
from scipy.spatial.distance import cosine
from fastdist import fastdist


def _convert_nltk_to_wordnet_tag(pos_tag):
    if pos_tag.startswith("N"):
        return wn.NOUN
    # elif pos_tag.startswith("V"):
    #     return wn.VERB
    # elif pos_tag.startswith("R"):
    #     return wn.ADV
    elif pos_tag.startswith("J"):
        return wn.ADJ
    else:
        return None


def _is_nltk_pos_tag_to_mask(pos_tag):
    if pos_tag.startswith("N"):             # Corresponds to a wn.NOUN
        return True
    elif pos_tag.startswith("VBN"):
        return True
    elif pos_tag.startswith("J"):           # Corresponds to a wn.ADJ
        return True
    else:
        return False


def _get_bert_sentence_embedding(tokenizer, model, sentence):
    marked_sentence = "[CLS] " + sentence + " [SEP]"
    tokenized_sentence = tokenizer.tokenize(marked_sentence)                    # Tokenize our sentence with the BERT tokenizer
    indexed_sentence = tokenizer.convert_tokens_to_ids(tokenized_sentence)      # Map the token strings to their vocabulary indexes.
    segments_ids = [1] * len(indexed_sentence)                                  # Mark each query token as belonging to sentence "1"
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


def _get_bert_token_embedding(tokenizer, model, sentence, token_index):
    marked_sentence = "[CLS] " + sentence + " [SEP]"
    tokenized_sentence = tokenizer.tokenize(marked_sentence)                    # Tokenize our sentence with the BERT tokenizer
    indexed_sentence = tokenizer.convert_tokens_to_ids(tokenized_sentence)      # Map the token strings to their vocabulary indexes.
    segments_ids = [1] * len(indexed_sentence)                                  # Mark each query token as belonging to sentence "1"
    sentence_tensor = torch.tensor([indexed_sentence])                          # Convert inputs to PyTorch tensors
    segments_tensor = torch.tensor([segments_ids])

    with torch.no_grad():
        bert_outputs = model(sentence_tensor, segments_tensor)                  # Run the text through BERT
        hidden_states = bert_outputs[2]                                         # Collect all of the hidden states produced from all 12 layers

        # Get the embeddings list organized token-wise
        tokens_layers_embeddings = torch.stack(hidden_states, dim=0)
        tokens_layers_embeddings = torch.squeeze(tokens_layers_embeddings, dim=1)
        tokens_layers_embeddings = tokens_layers_embeddings.permute(1, 0, 2)

    # We are interested in the embedding of the token with index (token_index + 1) because we take into account the embedding of [CLS] special token
    token_layers_emb = tokens_layers_embeddings[token_index + 1]
    # To get a manageable token embedding, we concatenate the last four layers values, giving us a single word vector per token
    token_embedding = torch.cat((token_layers_emb[-1], token_layers_emb[-2], token_layers_emb[-3], token_layers_emb[-4]), dim=0)
    return token_embedding


def _get_sentence_words_to_mask_indexes(pos_tags):
    # Get the indexes of just some specific POS of the query (es. just nouns), so that we know which tokens to mask
    words_to_mask_indexes = [i for i in range(len(pos_tags)) if _convert_nltk_to_wordnet_tag(pos_tags[i][1]) is not None]
    return words_to_mask_indexes


def _get_bert_topk_predictions(mask_tokenizer, mask_model, masked_query_string, original_masked_word, k):
    encoded_input = mask_tokenizer.encode(masked_query_string, return_tensors="pt")
    mask_token_index = torch.where(encoded_input == mask_tokenizer.mask_token_id)[1]
    token_logits = mask_model(encoded_input).logits
    mask_token_logits = token_logits[0, mask_token_index, :]
    # softmax_mask_token_logits = torch.nn.functional.softmax(mask_token_logits, dim=None, _stacklevel=3, dtype=None)

    # Get the best 10 predictions from BERT
    topk_encoded_tokens = torch.topk(mask_token_logits, k, dim=1).indices[0].tolist()
    topk_tokens = list((mask_tokenizer.decode(encoded_token)) for encoded_token in topk_encoded_tokens)

    # Add the original word of the query if not already inside best_tokens
    if original_masked_word not in topk_tokens:
        topk_tokens.append(original_masked_word)

    # Remove all the partial tokens retrieved by BERT (with format "##abcdefgh")
    topk_tokens_filtered = [token for token in topk_tokens if not token.startswith("##")]
    return topk_tokens_filtered


def _replace_partial_tokens(tokenized_query):
    i = 0
    while i < len(tokenized_query):
        if tokenized_query[i].startswith("##"):
            tokenized_query[i] = tokenized_query[i].replace('##', '')
            tokenized_query[i - 1] = tokenized_query[i - 1] + tokenized_query[i]
            tokenized_query.pop(i)
        else:
            i += 1
    return tokenized_query


def generate_similar_queries(input_query: str, verbose=False):

    # Use first a BERT model to get a list of proposed words in place of masked ones
    mask_tokenizer = AutoTokenizer.from_pretrained("../bert-base-uncased")
    mask_model = AutoModelForMaskedLM.from_pretrained("../bert-base-uncased")

    # Get the indexes of just some specific POS of the query (es. just nouns), so that we know which tokens to mask
    tokenized_query = mask_tokenizer.tokenize(input_query)
    pos_tags_wn = nltk.pos_tag(tokenized_query)
    masked_words_indexes = [i for i in range(len(pos_tags_wn)) if _convert_nltk_to_wordnet_tag(pos_tags_wn[i][1] is not None)]

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
    print("query_sim_dict: ", query_sim_dict)

    # Prune the previous dictionary keeping only the "best queries", which are the ones with higher scores (80+ %)
    max_cos_sim = 1.0
    min_cos_sim = min(cos_sim_list)
    sim_thresh = min_cos_sim + 0.8 * (max_cos_sim - min_cos_sim)
    best_query_sim_dict = {k: v for k, v in query_sim_dict.items() if v >= sim_thresh}
    best_queries_list = [k for k in best_query_sim_dict.keys()]

    return best_queries_list


def impr_generate_similar_queries(input_query: str, max_n_query=20, verbose=False):

    if max_n_query <= 0:
        print("Can't generate a negative number of new queries or no queries at all!")
        return 1

    # Use just one tokenizer for the whole task
    bert_tokenizer = BertTokenizerFast.from_pretrained('../bert-base-uncased')

    # Use first a BERT model to get a list of proposed words in place of masked ones
    mask_model = AutoModelForMaskedLM.from_pretrained("../bert-base-uncased")
    mask_model.eval()

    tokenized_query = bert_tokenizer.tokenize(input_query)
    if verbose:
        print(f"Original tokenized_query: {tokenized_query}")

    # Refine the tokenized query to concatenate partial tokens "##abcdefgh" with their own full tokens
    tokenized_query = _replace_partial_tokens(tokenized_query)
    if verbose:
        print(f"Refined tokenized_query: {tokenized_query}")

    # Get the indexes of just some specific POS of the query (es. just nouns and adjectives), which will be the tokens we actually mask
    pos_tags_wn = nltk.pos_tag(tokenized_query)
    if verbose:
        print(f"POS tagging result: {pos_tags_wn}\n")
    masked_words_indexes = [i for i in range(len(pos_tags_wn)) if _is_nltk_pos_tag_to_mask(pos_tags_wn[i][1])]
    original_masked_words = list(tokenized_query[i] for i in masked_words_indexes)
    pos_tags_masked_words = list(pos_tags_wn[i][1] for i in masked_words_indexes)
    n_masked_words = len(masked_words_indexes)              # Depending on this value we decide how many substitutions we can accept per masked token

    # Generate all the queries with [MASK] special token in place of the nouns
    masked_query_strings_list = list()
    for word_index_to_mask in masked_words_indexes:
        tokenized_query_to_mask = list(tokenized_query)  # Just take a copy not to touch
        tokenized_query_to_mask[word_index_to_mask] = bert_tokenizer.mask_token
        # print("tokenized_masked_query: ", tokenized_query_to_mask)

        masked_query_string = " ".join(tokenized_query_to_mask)
        if verbose:
            print(f"Query with {word_index_to_mask + 1}° word masked: {masked_query_string}")
        masked_query_strings_list.append(masked_query_string)

    # For each masked_query_string generate the top 10 words that fit inside the [MASK] position
    best_tokens_list = list()
    for i in range(len(masked_query_strings_list)):
        best_tokens = _get_bert_topk_predictions(bert_tokenizer, mask_model, masked_query_strings_list[i], original_masked_words[i], 10)
        best_tokens_list.append(best_tokens)
    if verbose:
        print(f"best_tokens_list: {best_tokens_list}")
        print(f"original_masked_words: {original_masked_words}")
        print(f"pos_tags_masked_words: {pos_tags_masked_words}\n")

    # Use another BERT model to get the query embeddings
    emb_model = BertModel.from_pretrained("../bert-base-uncased", output_hidden_states=True)
    emb_model.eval()

    # Fill best_new_tokens_list with the n (which is not fixed) best "substitute tokens" with respect to the K masked ones.
    # Example: [['best_tok_mask_1_#1', ... , 'best_tok_mask_1_#n'], ... , ['best_tok_mask_K_#0', ... , 'best_tok_mask_K_#n']]
    best_new_tokens_list = list()
    for i in range(len(best_tokens_list)):
        original_token_embedding = _get_bert_token_embedding(bert_tokenizer, emb_model, input_query, masked_words_indexes[i])
        best_tokens = best_tokens_list[i]

        # Collect inside the new_token_sim_dict dictionary all (new_token_for_masked_word_index, cosine_similarity_score) pairs
        new_token_sim_dict = dict()
        for best_token in best_tokens:
            new_tokenized_query = tokenized_query.copy()
            new_tokenized_query[masked_words_indexes[i]] = best_token
            new_query_string = " ".join(new_tokenized_query)
            # print(new_query_string)

            new_token_embedding = _get_bert_token_embedding(bert_tokenizer, emb_model, new_query_string, masked_words_indexes[i])
            # print(new_token_embedding.size())

            cos_sim = 1 - cosine(original_token_embedding, new_token_embedding)
            new_token_sim_dict[best_token] = cos_sim

            if verbose:
                print(f"{best_token} -> cos similarity score: {cos_sim}")
        if verbose:
            print(f"new_token_sim_dict: {new_token_sim_dict}\n")

        # N.B. These variables values are chosen just according to a heuristics!
        base_sim_thresh = 0.85                                              # Min threshold to consider two embeddings similar at first step
        low_sim_thresh = 0.75                                               # Min threshold to consider two embeddings similar at second step
        max_combinations = 5000                                             # Max number of query combinations, computed as: top_k^(n_masked_words)
        top_k = math.floor(max_combinations ** (1 / n_masked_words))        # Min top_k value is 1 by definition
        top_k = top_k if top_k <= 10 else 10                                # Max top_k value have to be 10, otherwise errors arise!

        # Perform an initial screening => accept only tokens with at least base_sim_thresh similarity with the original "masked" ones
        best_new_token_sim_dict = {k: v for k, v in new_token_sim_dict.items() if v >= base_sim_thresh}
        if verbose:
            print("Sub-dict of tokens with scores above sim_thresh: ", best_new_token_sim_dict)

        if len(best_new_token_sim_dict) > 1:
            # Default case: at least 1 out of 10 tokens initially proposed by BERT is good enough
            # Guarantee that for each token there are at most top_k new tokens (to limit execution time while keeping results quality high)
            if len(new_token_sim_dict) >= top_k:  # Should always be true, since Bert retrieves top 10 tokens
                # Sort the dictionary of candidates by score, keep only tokens with score above low_sim_thresh and of these take top_k
                sorted_new_token_sim_dict = {k: v for k, v in sorted(new_token_sim_dict.items(), reverse=True, key=lambda item: item[1])}
                sorted_new_token_sim_dict_over_thresh = {k: v for k, v in sorted_new_token_sim_dict.items() if v >= low_sim_thresh}
                best_new_token_sim_dict = {k: sorted_new_token_sim_dict_over_thresh[k] for k in list(sorted_new_token_sim_dict_over_thresh)[:top_k]}

        elif len(best_new_token_sim_dict) == 1:
            # Special case: none of the 10 tokens initially proposed by BERT is good enough
            # => we need to extract new candidates, 20 per iteration (max 100), until at least one good token in the batch is found
            n_words_to_extract = 20                                 # We generate again the first 10, because now we consider low_sim_thresh
            while len(best_new_token_sim_dict) <= 1 and n_words_to_extract < 100:
                best_tokens = _get_bert_topk_predictions(bert_tokenizer, mask_model, masked_query_strings_list[i], original_masked_words[i],
                                                         n_words_to_extract)
                if verbose:
                    print(f"\nLet's analyze the next set of 20 BERT candidate tokens:")

                new_token_sim_dict = dict()
                for best_token in best_tokens[n_words_to_extract-20:n_words_to_extract]:        # Compute cos_sim only on new 20 tokens
                    new_tokenized_query = tokenized_query.copy()
                    new_tokenized_query[masked_words_indexes[i]] = best_token
                    new_query_string = " ".join(new_tokenized_query)
                    # print("new_query_string: ", new_query_string)

                    new_token_embedding = _get_bert_token_embedding(bert_tokenizer, emb_model, new_query_string, masked_words_indexes[i])
                    # print(new_token_embedding.size())

                    cos_sim = 1 - cosine(original_token_embedding, new_token_embedding)
                    new_token_sim_dict[best_token] = cos_sim

                    if verbose:
                        print(f"{best_token} -> cos similarity score: {cos_sim}")

                # Add to best_new_token_sim_dict the words that, in the current batch of 20 candidates, have score above 0.7
                for k in new_token_sim_dict.keys():
                    if new_token_sim_dict[k] >= low_sim_thresh:
                        best_new_token_sim_dict[k] = new_token_sim_dict[k]

                n_words_to_extract += 20

            # If now, thanks to the lower similarity threshold, too many tokens are selected, keep only the best top_k
            if len(best_new_token_sim_dict) >= top_k:
                # Sort the dictionary of candidates by score, keep only tokens with score above 0.7 and of these take top_k
                sorted_best_new_token_sim_dict = {k: v for k, v in sorted(best_new_token_sim_dict.items(), reverse=True, key=lambda item: item[1])}
                best_new_token_sim_dict = {k: sorted_best_new_token_sim_dict[k] for k in list(sorted_best_new_token_sim_dict)[:top_k]}

        else:
            # This should never happen, since at least the original word should be 100% similar to itself
            print("ERROR: best_new_token_sim_dict should not be empty!")
            return 1

        if verbose:
            print(f"best_new_token_sim_dict: {best_new_token_sim_dict}")


        best_new_tokens = [k for k in best_new_token_sim_dict.keys()]
        if verbose:
            print(f"best_new_tokens list for query with {masked_words_indexes[i] + 1}° word masked: {best_new_tokens}\n")

        best_new_tokens_list.append(best_new_tokens)

    if verbose:
        print("best_new_tokens_list: ", best_new_tokens_list)

    # Build query_synonyms_list with the list of feasible tokens for each position inside the query
    query_synonyms_list = list([None] * len(pos_tags_wn))
    for i in range(len(pos_tags_wn)):
        if i not in masked_words_indexes:  # Add a list with just the original word
            query_synonyms_list[i] = list()
            query_synonyms_list[i].append(tokenized_query[i])
    for j in range(len(masked_words_indexes)):  # Add the list of best tokens we just found
        query_synonyms_list[masked_words_indexes[j]] = best_new_tokens_list[j]

    # Compose all new queries obtained combining the best tokens
    new_queries_list = list(itertools.product(*query_synonyms_list))
    new_queries_strings = list(" ".join(new_query) for new_query in new_queries_list)

    if verbose:
        print(f"\nTotal number of queries generated: {len(new_queries_strings)}")

    # If the number of queries generated is too high (> max_n_query) then select a random (deterministic) subset
    if len(new_queries_strings) > max_n_query:
        random.seed(0)
        original_query = new_queries_strings[0]
        new_queries_strings = random.sample(new_queries_strings, max_n_query - 1)
        new_queries_strings.insert(0, original_query)                                       # Don't forget to keep the original query

    if verbose:
        print(f"Final list of {len(new_queries_strings)} new queries: {new_queries_strings}\n")

    return new_queries_strings


def generate_similar_queries_all_topics(input_query_list, max_n_query=20, verbose=False):

    if max_n_query <= 0:
        print("Can't generate a negative number of new queries or no queries at all!")
        return 1

    # Load just one tokenizer for the whole task
    bert_tokenizer = BertTokenizerFast.from_pretrained('../bert-base-uncased')

    # Load a BERT model to get the list of candidate words to replace the masked ones
    mask_model = AutoModelForMaskedLM.from_pretrained("../bert-base-uncased")
    mask_model.eval()

    # Load another BERT model to compute the query embeddings
    emb_model = BertModel.from_pretrained("../bert-base-uncased", output_hidden_states=True)
    emb_model.eval()

    new_queries_strings_list = list()
    for t in range(len(input_query_list)):

        input_query = input_query_list[t]
        tokenized_query = bert_tokenizer.tokenize(input_query)
        if verbose:
            print(f"Original tokenized_query: {tokenized_query}")

        # Refine the tokenized query to concatenate partial tokens "##abcdefgh" with their own full tokens
        tokenized_query = _replace_partial_tokens(tokenized_query)
        if verbose:
            print(f"Refined tokenized_query: {tokenized_query}")

        # Get the indexes of just some specific POS of the query (es. just nouns and adjectives), which will be the tokens we actually mask
        pos_tags_wn = nltk.pos_tag(tokenized_query)
        if verbose:
            print(f"POS tagging result: {pos_tags_wn}\n")
        masked_words_indexes = [i for i in range(len(pos_tags_wn)) if _is_nltk_pos_tag_to_mask(pos_tags_wn[i][1])]
        original_masked_words = list(tokenized_query[i] for i in masked_words_indexes)
        pos_tags_masked_words = list(pos_tags_wn[i][1] for i in masked_words_indexes)
        n_masked_words = len(masked_words_indexes)  # Depending on this value we decide how many substitutions we can accept per masked token

        # Generate all the queries with [MASK] special token in place of the nouns
        masked_query_strings_list = list()
        for word_index_to_mask in masked_words_indexes:
            tokenized_query_to_mask = list(tokenized_query)  # Just take a copy not to touch
            tokenized_query_to_mask[word_index_to_mask] = bert_tokenizer.mask_token
            # print("tokenized_masked_query: ", tokenized_query_to_mask)

            masked_query_string = " ".join(tokenized_query_to_mask)
            if verbose:
                print(f"Query with {word_index_to_mask + 1}° word masked: {masked_query_string}")
            masked_query_strings_list.append(masked_query_string)

        # For each masked_query_string generate the top 10 words that fit inside the [MASK] position
        best_tokens_list = list()
        for i in range(len(masked_query_strings_list)):
            best_tokens = _get_bert_topk_predictions(bert_tokenizer, mask_model, masked_query_strings_list[i], original_masked_words[i], 10)
            best_tokens_list.append(best_tokens)
        if verbose:
            print(f"best_tokens_list: {best_tokens_list}")
            print(f"original_masked_words: {original_masked_words}")
            print(f"pos_tags_masked_words: {pos_tags_masked_words}\n")


        # Fill best_new_tokens_list with the n (which is not fixed) best "substitute tokens" with respect to the K masked ones.
        # Example: [['best_tok_mask_1_#1', ... , 'best_tok_mask_1_#n'], ... , ['best_tok_mask_K_#0', ... , 'best_tok_mask_K_#n']]
        best_new_tokens_list = list()
        for i in range(len(best_tokens_list)):
            original_token_embedding = _get_bert_token_embedding(bert_tokenizer, emb_model, input_query, masked_words_indexes[i])
            best_tokens = best_tokens_list[i]

            # Collect inside the new_token_sim_dict dictionary all (new_token_for_masked_word_index, cosine_similarity_score) pairs
            new_token_sim_dict = dict()
            for best_token in best_tokens:
                new_tokenized_query = tokenized_query.copy()
                new_tokenized_query[masked_words_indexes[i]] = best_token
                new_query_string = " ".join(new_tokenized_query)
                # print(new_query_string)

                new_token_embedding = _get_bert_token_embedding(bert_tokenizer, emb_model, new_query_string, masked_words_indexes[i])
                # print(new_token_embedding.size())

                cos_sim = 1 - cosine(original_token_embedding, new_token_embedding)
                new_token_sim_dict[best_token] = cos_sim

                if verbose:
                    print(f"{best_token} -> cos similarity score: {cos_sim}")
            if verbose:
                print(f"new_token_sim_dict: {new_token_sim_dict}\n")

            # N.B. These variables values are chosen just according to a heuristics!
            base_sim_thresh = 0.85  # Min threshold to consider two embeddings similar at first step
            low_sim_thresh = 0.75  # Min threshold to consider two embeddings similar at second step
            max_combinations = 5000  # Max number of query combinations, computed as: top_k^(n_masked_words)
            top_k = math.floor(max_combinations ** (1 / n_masked_words))  # Min top_k value is 1 by definition
            top_k = top_k if top_k <= 10 else 10  # Max top_k value have to be 10, otherwise errors arise!

            # Perform an initial screening => accept only tokens with at least base_sim_thresh similarity with the original "masked" ones
            best_new_token_sim_dict = {k: v for k, v in new_token_sim_dict.items() if v >= base_sim_thresh}
            if verbose:
                print("Sub-dict of tokens with scores above sim_thresh: ", best_new_token_sim_dict)

            if len(best_new_token_sim_dict) > 1:
                # Default case: at least 1 out of 10 tokens initially proposed by BERT is good enough
                # Guarantee that for each token there are at most top_k new tokens (to limit execution time while keeping results quality high)
                if len(new_token_sim_dict) >= top_k:  # Should always be true, since Bert retrieves top 10 tokens
                    # Sort the dictionary of candidates by score, keep only tokens with score above low_sim_thresh and of these take top_k
                    sorted_new_token_sim_dict = {k: v for k, v in sorted(new_token_sim_dict.items(), reverse=True, key=lambda item: item[1])}
                    sorted_new_token_sim_dict_over_thresh = {k: v for k, v in sorted_new_token_sim_dict.items() if v >= low_sim_thresh}
                    best_new_token_sim_dict = {k: sorted_new_token_sim_dict_over_thresh[k] for k in
                                               list(sorted_new_token_sim_dict_over_thresh)[:top_k]}

            elif len(best_new_token_sim_dict) == 1:
                # Special case: none of the 10 tokens initially proposed by BERT is good enough
                # => we need to extract new candidates, 20 per iteration (max 100), until at least one good token in the batch is found
                n_words_to_extract = 20  # We generate again the first 10, because now we consider low_sim_thresh
                while len(best_new_token_sim_dict) <= 1 and n_words_to_extract < 100:
                    best_tokens = _get_bert_topk_predictions(bert_tokenizer, mask_model, masked_query_strings_list[i], original_masked_words[i],
                                                             n_words_to_extract)
                    if verbose:
                        print(f"\nLet's analyze the next set of 20 BERT candidate tokens:")

                    new_token_sim_dict = dict()
                    for best_token in best_tokens[n_words_to_extract - 20:n_words_to_extract]:  # Compute cos_sim only on new 20 tokens
                        new_tokenized_query = tokenized_query.copy()
                        new_tokenized_query[masked_words_indexes[i]] = best_token
                        new_query_string = " ".join(new_tokenized_query)
                        # print("new_query_string: ", new_query_string)

                        new_token_embedding = _get_bert_token_embedding(bert_tokenizer, emb_model, new_query_string, masked_words_indexes[i])
                        # print(new_token_embedding.size())

                        cos_sim = 1 - cosine(original_token_embedding, new_token_embedding)
                        new_token_sim_dict[best_token] = cos_sim

                        if verbose:
                            print(f"{best_token} -> cos similarity score: {cos_sim}")

                    # Add to best_new_token_sim_dict the words that, in the current batch of 20 candidates, have score above 0.7
                    for k in new_token_sim_dict.keys():
                        if new_token_sim_dict[k] >= low_sim_thresh:
                            best_new_token_sim_dict[k] = new_token_sim_dict[k]

                    n_words_to_extract += 20

                # If now, thanks to the lower similarity threshold, too many tokens are selected, keep only the best top_k
                if len(best_new_token_sim_dict) >= top_k:
                    # Sort the dictionary of candidates by score, keep only tokens with score above 0.7 and of these take top_k
                    sorted_best_new_token_sim_dict = {k: v for k, v in
                                                      sorted(best_new_token_sim_dict.items(), reverse=True, key=lambda item: item[1])}
                    best_new_token_sim_dict = {k: sorted_best_new_token_sim_dict[k] for k in list(sorted_best_new_token_sim_dict)[:top_k]}

            else:
                # This should never happen, since at least the original word should be 100% similar to itself
                print("ERROR: best_new_token_sim_dict should not be empty!")
                return 1

            if verbose:
                print(f"best_new_token_sim_dict: {best_new_token_sim_dict}")

            best_new_tokens = [k for k in best_new_token_sim_dict.keys()]
            if verbose:
                print(f"best_new_tokens list for query with {masked_words_indexes[i] + 1}° word masked: {best_new_tokens}\n")

            best_new_tokens_list.append(best_new_tokens)

        if verbose:
            print("best_new_tokens_list: ", best_new_tokens_list)

        # Build query_synonyms_list with the list of feasible tokens for each position inside the query
        query_synonyms_list = list([None] * len(pos_tags_wn))
        for i in range(len(pos_tags_wn)):
            if i not in masked_words_indexes:  # Add a list with just the original word
                query_synonyms_list[i] = list()
                query_synonyms_list[i].append(tokenized_query[i])
        for j in range(len(masked_words_indexes)):  # Add the list of best tokens we just found
            query_synonyms_list[masked_words_indexes[j]] = best_new_tokens_list[j]

        # Compose all new queries obtained combining the best tokens
        new_queries_list = list(itertools.product(*query_synonyms_list))
        new_queries_strings = list(" ".join(new_query) for new_query in new_queries_list)

        if verbose:
            print(f"\nTotal number of queries generated: {len(new_queries_strings)}")

        # If the number of queries generated is too high (> max_n_query) then select a random (deterministic) subset
        if len(new_queries_strings) > max_n_query:
            random.seed(0)
            original_query = new_queries_strings[0]
            new_queries_strings = random.sample(new_queries_strings, max_n_query - 1)
            new_queries_strings.insert(0, original_query)  # Don't forget to keep the original query

        if verbose:
            print(f"Final list of {len(new_queries_strings)} new queries: {new_queries_strings}\n")

        new_queries_strings_list.append(new_queries_strings)
        print(f"Ended processing topic n°{t}")

    return new_queries_strings_list
