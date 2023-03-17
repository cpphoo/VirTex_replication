# Run this script to generate vocabulary from the coco captions
# use sentencepiece's BPE algorithm to generate vocabulary
# basically running code here
# https://github.com/google/sentencepiece/blob/master/python/README.md

import argparse
import json

import unicodedata
import tempfile

import os

import sentencepiece as spm


def main(args):
    with open(args.data) as f:
        captions = json.load(f)

    # get all the captions
    all_captions = []
    for ann in captions['annotations']:
        caption = ann['caption'].lower()
        caption = unicodedata.normalize("NFKD", caption)
        caption = "".join([chr for chr in caption if not unicodedata.combining(chr)])
        all_captions.append(caption)

    # use sentencepeice to generate vocabulary
    # Need to first generate a text file with all captions
    # Create a temporary directory and dump the captions corpus as a text file
    # with one caption per line. That's how sentencepiece wants its input.

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "captions.txt"), "w") as captions_file:
        for caption in all_captions:
            captions_file.write(caption + "\n")


    # Run BPE to generate tokenizer 
    # unknown token has idx 0
    # sos has idx 1
    # eos has idx 2
    
    spm.SentencePieceTrainer.train(input=os.path.join(args.output_dir, "captions.txt"), 
                                   model_prefix=os.path.join(args.output_dir, args.output_name),
                                   vocab_size=args.vocab_size, 
                                   character_coverage=1.0,
                                   model_type='bpe',
                                   bos_id=-1,
                                   eos_id=-1,
                                   control_symbols=["[SOS]", "[EOS]", "[MASK]"])











if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, 
                        default='/scratch/datasets/cp598/coco/annotations/captions_train2017.json', 
                        help='data directory')
    parser.add_argument('--output_dir', type=str, default='tokenizer',help='directory to save the tokenizer')
    parser.add_argument('--output_name', type=str, default='coco_vocab_2017', 
                        help='directory to save the generated output')
    parser.add_argument('--vocab_size', type=int, default=10000, help='size of the vocabulary')
    
    args = parser.parse_args()
    main(args)
