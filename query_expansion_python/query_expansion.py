import itertools
import os
import urllib

from urllib.error import HTTPError
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline, set_seed, BeamSearchScorer
from textblob import TextBlob
from string import punctuation
import nltk
from nltk.corpus import wordnet as wn
import sys
import xmltodict


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
    # elif pos_tag.startswith("R"):
    #     return wn.ADV
    # elif pos_tag.startswith("J"):
    #     return wn.ADJ
    else:
        return None


def main():

    filename = str(sys.argv[1])
    task = int(sys.argv[2])
    topic_list = get_topic_list(filename)
    print(f"Topics parsing completed. Starting with task #{task}.\n")

    num_query_submitted = 1

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

        for query in topic_list[:num_query_submitted]:
            tokenized_query = nltk.word_tokenize(query)
            print("tokenized_query: ", tokenized_query)

            pos_tags_wn = nltk.pos_tag(tokenized_query)
            print("pos_tages for WordNet: ", pos_tags_wn)

            new_queries = list()
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
            # has_tokens_ended = False
            # while not has_tokens_ended:
            #     new_query = ""
            #     has_tokens_ended = True
            #     for i in range(len(query_synonyms_list)):
            #         if len(query_synonyms_list[i]) == 1:                                    # No synonyms were found
            #             new_query += " "
            #             new_query += query_synonyms_list[i][0]
            #         elif len(query_synonyms_list[i]) == 0:                                  # This should not happen
            #             print("Error: no tokens found to build a query!")
            #         else:                                                                   # Take a synonym from list
            #             new_query += " "
            #             new_query += query_synonyms_list[i].pop()
            #             if len(query_synonyms_list[i]) == 1:
            #                 has_tokens_ended = True
            #             else:
            #                 has_tokens_ended = False
            #
            #     print(new_query)

            # print(list(itertools.combinations(query_synonyms_list, len(query_synonyms_list))))

            # Compose all queries obtained combining the new synonyms
            all_new_queries_list = list(itertools.product(*query_synonyms_list))
            all_new_queries = list(" ".join(new_query) for new_query in all_new_queries_list)
            # Just print all the resulting new queries (the first one should be the original)
            for new_query in all_new_queries:
                print(new_query)


if __name__ == "__main__":
    main()
