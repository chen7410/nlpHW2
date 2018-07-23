from __future__ import division
import nltk
from nltk import word_tokenize
import os


# train a language from a give file name and return a 
# tag dictionary and tagged token dictionary in the flowing from.
# tag dictionary : {TAG, count}
# tagged toekn dictionary : {word/TAG, count}
# returns tag dictionary, tagged token dictionary.

def process_file(filename):
    dict_tagged_token = {}
    dict_tag = {}

    with open(filename, 'r') as file:
        for line in file:
            for token in line.split():
                # (word, TAG)
                tagged_token = nltk.tag.str2tuple(token)

                if tagged_token in dict_tagged_token:
                    dict_tagged_token[tagged_token] += 1
                else:
                    dict_tagged_token[tagged_token] = 1
                # TAG
                if tagged_token[1] in dict_tag:
                    dict_tag[tagged_token[1]] += 1
                else:
                    dict_tag[tagged_token[1]] = 1

    file.close()
    return dict_tag, dict_tagged_token
    
# Calculate he probability of the the given word and tag in the training model.
# dict_tagged_token: tagged token dictionary from training data.
# dict_tag: tag dictionary from training data.
# smooth_factor: the smooth factor.
# return: the probability will at least be the smooth factor.
def calculate_prob(dict_tag, dict_tagged_token, word, tag, smooth_factor):
    key = (word, tag)
    # print(key)
    if key in dict_tagged_token:
        return (dict_tagged_token[key] / dict_tag[tag]) + smooth_factor
    else:
        return smooth_factor

# Outpput a table to a specified file name based on a specified training data.
# dict_tagged_token: tagged token dictionary from training data.
# dict_tag: tag dictionary from training data.
def write_table(filename, dict_tag, dict_tagged_token):
    s1 = "NOUN"
    s2 = "VERB"
    s3 = "CONJ"
    s4 = "PRO"
    s0 = " "
    with open(filename, "w") as file:
        file.write("%-20s|%20s|%20s|%20s|%20s\n" %(s0, s1, s2, s3, s4))
        for key in dict_tagged_token:
            c0 = key[0]
            c1 = calculate_prob(dict_tag, dict_tagged_token, key[0], 'N', 0.1)
            c2 = calculate_prob(dict_tag, dict_tagged_token, key[0], 'V', 0.1)
            c3 = calculate_prob(dict_tag, dict_tagged_token, key[0], 'CONJ', 0.1)
            c4 = calculate_prob(dict_tag, dict_tagged_token, key[0], 'PRO', 0.1)
            
            file.write("%-20s|%20.3f|%20.3f|%20.3f|%20.3f\n" %(c0, c1, c2, c3, c4))

    file.close()


def main():
    current_dir = os.path.dirname(os.path.realpath(__file__))

    dict_tag, dict_tagged_token = process_file(current_dir + "/Klingon_Train.txt")
    write_table(current_dir + "/Emission.txt", dict_tag, dict_tagged_token)


main()

