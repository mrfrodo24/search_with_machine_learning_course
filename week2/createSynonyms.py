import argparse
import fasttext

parser = argparse.ArgumentParser()
general_args = parser.add_argument_group("general")
general_args.add_argument("--similarity_thresh", default=0.75, type=int)
args = parser.parse_args()

similarity_thresh = args.similarity_thresh

if __name__ == '__main__':
    model = fasttext.load_model('/workspace/datasets/fasttext/title_model.bin')
    words_file = open(r'/workspace/datasets/fasttext/top_words.txt')
    words = words_file.readlines()
    with open('/workspace/datasets/fasttext/synonyms.csv', 'w') as output:
        for word in words:
            neighbors = model.get_nearest_neighbors(word)
            synonyms = [nn for (similarity, nn) in neighbors if similarity >= similarity_thresh]
            if len(synonyms) > 1:
                line = ','.join(synonyms)
                output.write(f'{line}\n')