import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline, set_seed, BeamSearchScorer
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


def main():
    filename = str(sys.argv[1])
    topic_list = get_topic_list(filename)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # num_beams = 2
    # beam_scorer = BeamSearchScorer(
    #    batch_size=1,
    #   max_length=model.config.max_length,
    #  num_beams=num_beams,
    #   device=model.device,
    # )

    num_query = 3
    for query in topic_list[:num_query]:
        inputs = tokenizer(query, return_tensors="pt")
        greedy_output = model.generate(input_ids=inputs.input_ids, min_length=20, max_length=40,
                                       return_dict_in_generate=True, output_scores=True)
        beam_output = model.generate(input_ids=inputs.input_ids, num_beams=2, min_length=20, max_length=40,
                                     num_return_sequences=1)

        greedy_output = tokenizer.decode(greedy_output.sequences[0], skip_special_tokens=True)
        beam_output = tokenizer.decode(beam_output[0], skip_special_tokens=True)
        # beam_output = model.beam_search(inputs.input_ids, beam_scorer=beam_scorer, return_dict_in_generate=True, output_scores=True)
        print(f"Original: {query} \n Greedy: {greedy_output} \n Beam Search: {beam_output}")

    # inputs.input_ids {batch_size} --> {batch_size*num_beams}


main()
