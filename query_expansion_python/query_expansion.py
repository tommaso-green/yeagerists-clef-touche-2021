import itertools
import sys
from string import punctuation
from urllib.error import HTTPError

from scipy.spatial.distance import cosine

import nltk
import torch
import xmltodict
from nltk.corpus import wordnet as wn
from textblob import TextBlob
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline, BeamSearchScorer, AutoModelForMaskedLM, AutoTokenizer, BertTokenizer, BertModel

from query_exp_utils import *


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
    elif pos_tag.startswith("V"):
        return wn.VERB
    # elif pos_tag.startswith("R"):
    #     return wn.ADV
    # elif pos_tag.startswith("J"):
    #     return wn.ADJ
    else:
        return None


def intersection(list1, list2):
    return list(set(list1) & set(list2))


def get_bert_sentence_embedding(tokenizer, model, sentence):

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


def get_bert_token_embedding(indexed_sentence, token_index):
    model = BertModel.from_pretrained("../bert-base-uncased", output_hidden_states=True)

    segments_ids = [1] * len(indexed_sentence)
    sentence_tensor = torch.tensor([indexed_sentence])
    segments_tensor = torch.tensor([segments_ids])

    with torch.no_grad():
        bert_outputs = model(sentence_tensor, segments_tensor)
        hidden_states = bert_outputs[2]
        token_layers_embeddings = torch.stack(hidden_states, dim=0)
        token_layers_embeddings = torch.squeeze(token_layers_embeddings, dim=1)
        token_layers_embeddings = token_layers_embeddings.permute(1, 0, 2)

    token = token_layers_embeddings[token_index]
    token_embedding = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)

    print("Shape is: ", token_embedding.size())

    return token_embedding


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

                for lang in ["ar", "de", "es", "it", "eo", "et", "fr", "la", "no", "sv", "ru", "ja"]:
                    spinned_query = _spin_text(c, lang)
                    print(spinned_query)
            except HTTPError as http_exception:
                print("An HTTP error occurred: ", http_exception.reason)
                print("Full HTTP error header:\n", http_exception.headers)
            except:
                print("Another error occurred!")

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

    # Back-translation using HuggingFace's Transformers models
    # N.B. Can be used in final implementation but requires to download t5-base repo in the
    # root folder of the project (es. in our case inside /seupd2021-yeager)
    # N.B. Useless with "mt5-small" since it can't translate to more than 3 languages + can't perform back-translation
    elif task == 4:

        # model = AutoModelForSeq2SeqLM.from_pretrained("../mt5-small", from_tf=True)
        # tokenizer = AutoTokenizer.from_pretrained("../mt5-small", from_tf=True)

        # Take all the languages available in mbart-large-cc25:
        # languages = ["ar_AR", "cs_CZ", "de_DE", "en_XX", "es_XX", "et_EE", "fi_FI", "fr_XX", "gu_IN", "hi_IN", "it_IT", "ja_XX", "kk_KZ", "ko_KR",
        #              "lt_LT", "lv_LV", "my_MM", "ne_NP", "nl_XX", "ro_RO", "ru_RU", "si_LK", "tr_TR", "vi_VN", "zh_CN"]
        # tokenizer = MBartTokenizer.from_pretrained("../mbart-large-cc25", src_lang="en_XX")
        # model = AutoModelForSeq2SeqLM.from_pretrained("../mbart-large-cc25")

        for query in topic_list[:num_query_submitted]:
            from transformers import MBartTokenizer, BartForConditionalGeneration

            tokenizer = MBartTokenizer.from_pretrained('../mbart-large-cc25')
            model = BartForConditionalGeneration.from_pretrained('../mbart-large-cc25')

            src_sent = "UN Chief Says There Is No Military Solution in Syria"

            src_ids = tokenizer.prepare_translation_batch([src_sent])

            output_ids = model.generate(src_ids["input_ids"], decoder_start_token_id=tokenizer.lang_code_to_id["ro_RO"])

            output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

            print('src_sent: ', src_sent)
            print('src_ids: ', src_ids)
            print('output_ids: ', output_ids)
            print('output: ', output)
            # inputs = tokenizer(query, return_tensors="pt")
            # # translated_tokens = model.generate(**inputs, decoder_start_token_id=tokenizer.lang_code_to_id["ro_RO"])
            # translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.bos_token["ro_RO"])
            # print(tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0])

        # translate_prompts = ["Translate English to German: ",
        #                      "Translate English to French: ",
        #                      "Translate English to Romanian: "]
        # inverse_translate_prompts = ["Translate German to English: ",
        #                              "Translate French to English: ",
        #                              "Translate Romanian to English: "]
        #
        # for query in topic_list[:num_query_submitted]:
        #     print("Query to translate: ", query)
        #
        #     for i in range(len(translate_prompts)):                        # Translate each query in multiple languages
        #
        #         # Move from English to the other language
        #         prompted_query = translate_prompts[i] + query
        #         encoded_query = tokenizer.encode(prompted_query, return_tensors="pt")
        #         translation_outputs = model.generate(encoded_query, max_length=40, num_beams=4, early_stopping=True)
        #         translated_query = tokenizer.decode(translation_outputs[0])
        #         # Immediately get back to English
        #         inverted_prompted_query = inverse_translate_prompts[i] + translated_query
        #         inverted_encoded_query = tokenizer.encode(inverted_prompted_query, return_tensors="pt")
        #         back_translation_outputs = model.generate(inverted_encoded_query, max_length=40, num_beams=4, early_stopping=True)
        #         new_query = tokenizer.decode(back_translation_outputs[0])
        #
        #         polished_translated_query = translated_query.strip("<pad>").strip("</s>")
        #         polished_new_query = new_query.strip("<pad>").strip("</s>")
        #         print("Translated query: ", polished_translated_query)
        #         print("New query: ", polished_new_query)

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
    # the original ones, then evaluate the candidate’s fitness by comparing (using cosine similarity between word
    # embeddings inside the respective query, original and modified) the sentence’s contextualized representation
    # before and after the substitution. Based on this, compute a "changed fitness" score and decide whether
    # to keep the proposed word or not.
    elif task == 6:

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # tokenizer = AutoTokenizer.from_pretrained("../distilbert-base-cased")
        model = BertModel.from_pretrained("../bert-base-uncased", output_hidden_states=True)
        # model = AutoModelForMaskedLM.from_pretrained("../distilbert-base-cased")


    # Get the 10 best tokens which substitutes the masked tokens of the queries (only the nouns according to WordNet),
    # compose the new queries computing all the possible combinations and compare their embeddings to the original query embedding
    # to understand if the query meaning changed a lot or not.
    # Keep only the queries which lead to good cosine similarity with the original one.
    elif task == 7:

        # Use first a BERT model to get a list of proposed words in place of masked ones
        tokenizer = AutoTokenizer.from_pretrained("../bert-base-uncased")
        model = AutoModelForMaskedLM.from_pretrained("../bert-base-uncased")

        all_new_queries_list = list()
        for query in topic_list[:num_query_submitted]:

            print("Original query: ", query)
            tokenized_query = tokenizer.tokenize(query)
            print("tokenized_query: ", tokenized_query)
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
                    masked_words_indexes.append(i)

            print("masked_words_indexes: ", masked_words_indexes)

            # Generate all the queries with [MASK] special token in place of the nouns
            masked_query_strings_list = list()
            for word_index_to_mask in masked_words_indexes:
                tokenized_query_to_mask = list(tokenized_query)  # Just take a copy not to touch
                tokenized_query_to_mask[word_index_to_mask] = tokenizer.mask_token
                # print("tokenized_masked_query: ", tokenized_query_to_mask)

                masked_query_string = " ".join(tokenized_query_to_mask)
                print("masked_query_string: ", masked_query_string)
                masked_query_strings_list.append(masked_query_string)

            print()

            # For each masked_query_string generate the top 10 words that fit inside the [MASK] position
            best_tokens_list = list()
            for i in range(len(masked_query_strings_list)):
                masked_query_string = masked_query_strings_list[i]

                encoded_input = tokenizer.encode(masked_query_string, return_tensors="pt")
                mask_token_index = torch.where(encoded_input == tokenizer.mask_token_id)[1]
                # mask_token_index will be one unit higher than expected because of Bert [CLS] special token
                # at the beginning of the embedding
                # print("mask_token_index: ", mask_token_index)

                token_logits = model(encoded_input).logits
                # print("token_logits: ", token_logits)
                mask_token_logits = token_logits[0, mask_token_index, :]
                # print("mask_token_logits: ", mask_token_logits)

                top_10_encoded_tokens = torch.topk(mask_token_logits, 10, dim=1).indices[0].tolist()
                top_10_tokens = list((tokenizer.decode(encoded_token)) for encoded_token in top_10_encoded_tokens)

                # Add the original word of the query if not already inside best_tokens
                masked_word_index = masked_words_indexes[i]
                if pos_tags_wn[masked_word_index][0] not in top_10_tokens:
                    top_10_tokens.append(pos_tags_wn[masked_word_index][0])
                best_tokens_list.append(top_10_tokens)
                print("-> Resulting 10 best_tokens: ", top_10_tokens)

            print()

            # Build query_synonyms_list with the list of feasible tokens for each position inside the query
            query_synonyms_list = list([None] * len(pos_tags_wn))
            for i in range(len(pos_tags_wn)):
                if i not in masked_words_indexes:  # Add a list with just the original word
                    query_synonyms_list[i] = list()
                    query_synonyms_list[i].append(pos_tags_wn[i][0])
            for j in range(len(masked_words_indexes)):  # Add the list of best tokens we just found
                query_synonyms_list[masked_words_indexes[j]] = best_tokens_list[j]

            # Compose all new queries obtained combining the best tokens
            new_queries_list = list(itertools.product(*query_synonyms_list))
            new_queries_strings = list(" ".join(new_query) for new_query in new_queries_list)
            print("new_queries_strings: ", new_queries_strings)

            # Add to the list of new queries for all original queries
            all_new_queries_list.append(new_queries_strings)

            print()

            # Use another BERT model and tokenizer to get the query embeddings
            tokenizer_2 = BertTokenizer.from_pretrained('../bert-base-uncased')
            model_2 = BertModel.from_pretrained("../bert-base-uncased", output_hidden_states=True)

            # Compute the embeddings for each new query and compute the cosine similarity to the original one,
            # which is the first query in new_queries_strings, so the first cos_sim should be 1.00
            original_query_embedding = get_bert_sentence_embedding(tokenizer_2, model_2,new_queries_strings[0])

            cos_sim_list = list()
            for new_query in new_queries_strings:
                new_query_embedding = get_bert_sentence_embedding(tokenizer_2, model_2, new_query)
                # print(new_query_embedding.size())
                # new_queries_embeddings.append(new_query_embedding)
                cos_sim = 1 - cosine(original_query_embedding, new_query_embedding)
                cos_sim_list.append(cos_sim)
                print(f"{new_query} -> cos similarity score: {cos_sim}")

            query_sim_dict = dict(zip(new_queries_strings, cos_sim_list))
            print(query_sim_dict)

            # Keep only the "best queries", which are the ones with higher scores (80+ %)
            max_cos_sim = 1.0
            min_cos_sim = min(cos_sim_list)
            sim_thresh = min_cos_sim + 0.8 * (max_cos_sim - min_cos_sim)
            print("sim_thresh: ", sim_thresh)
            best_query_sim_dict = {k: v for k, v in query_sim_dict.items() if v >= sim_thresh}
            print(best_query_sim_dict)

            print()

    # Demo task to test query_exp_utils
    elif task == 8:
        all_new_queries = generate_similar_queries_2(topic_list[4], verbose=True)
        print("Total number of queries generated: ", len(all_new_queries))
        for query in all_new_queries:
            print(query)
        print("\nDemo ended successfully!")


if __name__ == "__main__":
    main()
