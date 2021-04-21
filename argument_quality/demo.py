from argquality_dataset import *
from model import *
import os


def main():
    # Setting everything to deterministic behaviour
    SEED = 4223123
    pl.seed_everything(SEED)

    # Load dataset and extract sample_size arguments
    full_dataset = load_dataset('csv', data_files="../webis-argquality20-full.csv").shuffle(SEED)
    sample_size = 15
    arg_batch = full_dataset['train'][:sample_size - 1]  # List of strings of arguments
    for x in arg_batch['Premise']:
        print(x)

    # Load model from training checkpoint
    ckpt_file = "model_checkpoints/" + os.listdir('model_checkpoints')[0]
    arg_quality_model = ArgQualityModel.load_from_checkpoint(ckpt_file)

    # Run model on the arguments: it
    # Returns a list of tuples [ .. (a,s(a)) ..] where a is the argument and s(a) is the score
    output_list = arg_quality_model(arg_batch['Premise'])
    for idx, x in enumerate(output_list):
        x.append(arg_batch['Combined Quality'][idx])

    output_list.sort(key=lambda x: x[1], reverse=True)  # Sort by decreasing scores
    for idx, element in enumerate(output_list):  # Print results
        print(f"\nArgument {idx + 1}:\n{element[0]}")
        print(f"Score: Predicted {element[1]} - Actual: {element[2]}")


if __name__ == "__main__":
    main()
