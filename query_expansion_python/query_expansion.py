import itertools
import os
import urllib
import nltk
import torch
import sys
import xmltodict

from urllib.error import HTTPError
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline, BeamSearchScorer, AutoModelForMaskedLM, AutoTokenizer
from textblob import TextBlob
from string import punctuation
from pprint import pprint
from nltk.corpus import wordnet as wn


def get_topic_list(filename):
    f = open(filename, "r")
    xml_data = f.read()
    # converting xml to dictionary
    my_dict = xmltodict.parse(xml_data)
    # determining the type of converted object
    topic_list_full = my_dict['topics']['topic']
    topic_list = [element['title'] for element in topic_list_full]
    return topic_list


def _clean_word(word):
    return word.lower().strip(punctuation)


def _spin_text(text, foreign_language):
    try:
        spun_text = _clean_word(TextBlob(TextBlob(text).translate(from_lang="en",to=foreign_language).raw).translate(from_lang=foreign_language,to="en").raw)
        return spun_text if spun_text != _clean_word(text) else None
    except:
        return None


def synonyms(word, pos_tag):
    return list({lemma.replace("_", " ").replace("-", " ") for synset in wn.synsets(
        _clean_word(word), pos_tag, ) for lemma in synset.lemma_names()})


def _infer_pos_tags(tokens):
    return [(token, _convert_nltk_to_wordnet_tag(nltk_tag)) for token, nltk_tag in nltk.pos_tag(tokens)]


def _convert_nltk_to_wordnet_tag(pos_tag):
    if pos_tag.startswith("N"):
        return wn.NOUN
    # elif pos_tag.startswith("V"):
    #     return wn.VERB
    elif pos_tag.startswith("R"):
        return wn.ADV
    elif pos_tag.startswith("J"):
        return wn.ADJ
    else:
        return None


def intersection(list1, list2):
    return list(set(list1) & set(list2))


def main():

    filename = str(sys.argv[1])
    task = int(sys.argv[2])
    num_query_submitted = int(sys.argv[3])

    topic_list = get_topic_list(filename)
    print(f"Topics parsing completed. Starting with task #{task}.\n")

    # Text generation with GPT-2 and both positive and negative prompts
    if task == 0:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2')

        num_beams = 2
        beam_scorer = BeamSearchScorer(
            batch_size=1,
            max_length=model.config.max_length,
            num_beams=num_beams,
            device=model.device,
        )

        for query in topic_list[:num_query_submitted]:
            # Attach two different prompts to get some positive and negative related words
            positive_prompt = " Yes, I think that"
            negative_prompt = " No, I think that"

            pos_prompted_query = str(query + positive_prompt)
            neg_prompted_query = str(query + negative_prompt)

            pos_inputs = tokenizer(pos_prompted_query, return_tensors="pt")
            neg_inputs = tokenizer(neg_prompted_query, return_tensors="pt")
            #
            # # greedy_output = model.generate(input_ids=inputs.input_ids, min_length=20, max_length=40,
            # #                                return_dict_in_generate=True, output_scores=True)
            #
            pos_beam_output = model.generate(input_ids=pos_inputs.input_ids, num_beams=3, min_length=20, max_length=40,
                                             num_return_sequences=1, top_p=0.95, top_k=60)
            neg_beam_output = model.generate(input_ids=neg_inputs.input_ids, num_beams=3, min_length=20, max_length=40,
                                             num_return_sequences=1, top_p=0.95, top_k=60)
            #
            # #greedy_output = tokenizer.decode(greedy_output.sequences[0], skip_special_tokens=True)
            pos_beam_output = tokenizer.decode(pos_beam_output[0], skip_special_tokens=True, output_scores=True)
            neg_beam_output = tokenizer.decode(neg_beam_output[0], skip_special_tokens=True, output_scores=True)
            # beam_output = model.beam_search(inputs.input_ids, beam_scorer=beam_scorer, return_dict_in_generate=True, output_scores=True)
            #
            print(f"1) Original: {query}")
            # print(f"2) Greedy: {greedy_output}")
            print(f"3) Beam Search (+): {pos_beam_output}")
            print(f"4) Beam Search (-): {neg_beam_output}")

        # inputs.input_ids {batch_size} --> {batch_size*num_beams}

    elif task == 1:
        classifier = pipeline('sentiment-analysis')
        results = list()  # List for sentiment classification results
        for query in topic_list[:num_query_submitted]:
            result = classifier(query)
            print(result)
            results.append(result)
            # print(f"label: {results['label']}, with score: {round(results['score'], 4)}")

    # Back-Translation (aka Spinning):
    # uses a translation model to translate a sentence into a foreign language and back again into the original one.
    # N.B. Not to be used in final implementation because it sends http requests to Google Translate servers...
    elif task == 2:
        for query in topic_list[:num_query_submitted]:
            print(query)
            try:
                a = TextBlob(query)
                b = a.translate(from_lang="en", to="ar")
                c = _clean_word(b.raw)
            except HTTPError as http_exception:
                print("An HTTP error occurred: ", http_exception.reason)
                print("Full HTTP error header:\n", http_exception.headers)
            except:
                print("Another error occurred!")

            # for lang in ["ar", "de", "es", "it", "eo", "et", "fr", "la", "no", "sv", "ru", "ja"]:
            #
            #     spinned_query = _spin_text(query, lang)
            #     print(spinned_query)
            print()

    # Substituting Synonyms with Part-Of-Speech (POS) filtering:
    # pick a word in the sentence and replace with one of its synonyms.
    elif task == 3:
        # Download required datasets
        nltk.download('wordnet')
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')

        all_new_queries_list = list()
        for query in topic_list[:num_query_submitted]:
            tokenized_query = nltk.word_tokenize(query)
            print("tokenized_query: ", tokenized_query)

            pos_tags_wn = nltk.pos_tag(tokenized_query)
            print("pos_tags for WordNet: ", pos_tags_wn)

            query_synonyms_list = list()
            for pos_tag_wn in pos_tags_wn:
                # synonyms_list = synonyms(pos_tag_wn[0], _convert_nltk_to_wordnet_tag(pos_tag_wn[1]))

                # We want to replace only nouns with synonyms (all other POS return None)
                converted_pos_tag = _convert_nltk_to_wordnet_tag(pos_tag_wn[1])
                if converted_pos_tag is not None:
                    synonyms_list = synonyms(pos_tag_wn[0], converted_pos_tag)
                    # Preprocess each synonym of the list so that there are no duplicates with capital letters
                    synonyms_list = list(_clean_word(synonym) for synonym in synonyms_list)
                    query_synonyms_list.append(synonyms_list)
                else:
                    query_synonyms_list.append([])

            # Add to the synonyms list also the original words
            for i in range(len(pos_tags_wn)):
                query_synonyms_list[i].insert(0, pos_tags_wn[i][0])
                query_synonyms_list[i] = list(dict.fromkeys(query_synonyms_list[i]))        # Remove duplicate tokens

            print("Current query synonyms list: ", query_synonyms_list)
            print()

            # Compose all queries obtained combining the new synonyms
            new_queries_list = list(itertools.product(*query_synonyms_list))
            new_queries_strings = list(" ".join(new_query) for new_query in new_queries_list)

            # Add to the list of new queries for all original queries
            all_new_queries_list.append(new_queries_strings)

        # Just print all the resulting new queries (the first one should be the original)
        for new_query_list in all_new_queries_list:
            for new_query in new_query_list:
                print(new_query)
            print()

    # Back-translation using Transformers
    elif task == 4:

        for query in topic_list[:num_query_submitted]:
            translator = pipeline("translation_en_to_de")
            de_queries = list(de_query["translation_text"] for de_query in translator(query, max_length=40))
            print(de_queries)

    # Substituting Synonyms using BERT with masked query (just for nouns?) and checking if the proposed word,
    # which should respect the context due to BERT capabilities, is included in the list of synonyms from WordNet.
    # If so, the word is added to the query_synonyms_list, otherwise it means that the meaning of the resulting
    # sentence would have changed too much from the original.
    # N.B. Can be used in final implementation but requires to download distilbert-base-cased repo in the
    #      root folder of the project (es. in our case inside /seupd2021-yeager)
    elif task == 5:
        # Download required datasets
        nltk.download('wordnet')
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')

        all_new_queries_list = list()
        for query in topic_list[:num_query_submitted]:

            print("Original query: ", query)
            tokenized_query = nltk.word_tokenize(query)
            pos_tags_wn = nltk.pos_tag(tokenized_query)
            print("pos_tags for WordNet: ", pos_tags_wn)

            # Use WordNet to find synonyms for each noun in the query
            masked_words_synonyms_list = list()
            masked_words_indexes = list()
            for i in range(len(pos_tags_wn)):
                pos_tag_wn = pos_tags_wn[i]
                # We want to replace only nouns with synonyms (all other POS return None)
                converted_pos_tag = _convert_nltk_to_wordnet_tag(pos_tag_wn[1])
                if converted_pos_tag is not None:
                    synonyms_list = synonyms(pos_tag_wn[0], converted_pos_tag)
                    masked_words_synonyms_list.append(synonyms_list)
                    masked_words_indexes.append(i)

            # print("masked_words_synonyms_list: ", masked_words_synonyms_list)
            # print("masked_words_indexes: ", masked_words_indexes)

            # Use BERT model to get a list of proposed words in place of masked ones
            tokenizer = AutoTokenizer.from_pretrained("../distilbert-base-cased")
            model = AutoModelForMaskedLM.from_pretrained("../distilbert-base-cased")

            masked_query_strings_list = list()
            for word_index_to_mask in masked_words_indexes:
                tokenized_query_to_mask = list(tokenized_query)                     # Just take a copy not to touch
                tokenized_query_to_mask[word_index_to_mask] = tokenizer.mask_token
                # print("tokenized_masked_query: ", tokenized_query_to_mask)

                masked_query_string = " ".join(tokenized_query_to_mask)
                print("masked_query_string: ", masked_query_string)
                masked_query_strings_list.append(masked_query_string)

            print()

            # For each masked_query_string generate the top 50 words that fit inside the [MASK] position
            # Then perform the intersection between this list of candidate words and that of synonyms
            best_tokens_list = list()
            for i in range(len(masked_query_strings_list)):
                masked_query_string = masked_query_strings_list[i]

                encoded_input = tokenizer.encode(masked_query_string, return_tensors="pt")
                mask_token_index = torch.where(encoded_input == tokenizer.mask_token_id)[1]
                # mask_token_index will be one unit higher than expected because of Bert [CLS] special token
                # at the beginning of the embedding
                # print("mask_token_index: ", mask_token_index)

                token_logits = model(encoded_input).logits
                mask_token_logits = token_logits[0, mask_token_index, :]

                top_50_encoded_tokens = torch.topk(mask_token_logits, 50, dim=1).indices[0].tolist()
                top_50_tokens = list(tokenizer.decode(encoded_token) for encoded_token in top_50_encoded_tokens)
                print("Top 50 tokens according to BERT: ", top_50_tokens)
                print("Top synonyms according to WordNet: ", masked_words_synonyms_list[i])

                best_tokens = intersection(masked_words_synonyms_list[i], top_50_tokens)

                # Add the original word of the query if not already inside best_tokens
                masked_word_index = masked_words_indexes[i]
                if pos_tags_wn[masked_word_index][0] not in best_tokens:
                    best_tokens.append(pos_tags_wn[masked_word_index][0])
                best_tokens_list.append(best_tokens)
                print("-> Resulting best_tokens: ", best_tokens)

            print()

            # Build query_synonyms_list with the list of feasible tokens for each position inside the query
            query_synonyms_list = list([None] * len(pos_tags_wn))
            for i in range(len(pos_tags_wn)):
                if i not in masked_words_indexes:               # Add a list with just the original word
                    query_synonyms_list[i] = list()
                    query_synonyms_list[i].append(pos_tags_wn[i][0])
            for j in range(len(masked_words_indexes)):          # Add the list of best tokens we just found
                query_synonyms_list[masked_words_indexes[j]] = best_tokens_list[j]

            # Compose all new queries obtained combining the best tokens
            new_queries_list = list(itertools.product(*query_synonyms_list))
            new_queries_strings = list(" ".join(new_query) for new_query in new_queries_list)

            # Add to the list of new queries for all original queries
            all_new_queries_list.append(new_queries_strings)

            print()

        # Just print all the resulting new queries (the first one should be the original)
        for new_query_list in all_new_queries_list:
            for new_query in new_query_list:
                print(new_query)
            print()

    # Idea taken from "BERT-based Lexical Substitution" paper: use BERT with masked queries to propose words replacing
    # the original ones, the evaluate the candidate’s fitness by comparing (using cosine similarity between word
    # embeddings inside the respective query, original and modified) the sentence’s contextualized representation
    # before and after the substitution. Based on this, compute a "changed fitness" score and decide whether
    # to keep the proposed word or not.
    elif task == 6:
        print("TO DO")



if __name__ == "__main__":
    main()
