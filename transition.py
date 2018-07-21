from __future__ import division
import nltk
from nltk import bigrams
from nltk import word_tokenize

# train a language from a give file name and return a 
# tag dictionary and bigram tag dictionary in the flowing from.
# bigram tag dictionary : {(TAG1, TAG2), count}
# tagged toekn dictionary : {word/TAG, count}
# returns : tag dictionary, bigram tag dictionary.
def process_file(filename):
    dict_tag = {}
    dict_bigram_tag = {}
    list_tag = []
    list_tag.append("START")
    dict_tag['START'] = 1
    with open(filename, 'r') as file:
        for line in file:
            for token in line.split():
                tagged_token = nltk.tag.str2tuple(token)
                list_tag.append(tagged_token[1])
                if tagged_token[1] in dict_tag:
                    dict_tag[tagged_token[1]] += 1
                else:
                    dict_tag[tagged_token[1]] = 1

        list_bigram_tag = list(nltk.bigrams(list_tag))

        for bigram in list_bigram_tag:
            if bigram in dict_bigram_tag:
                dict_bigram_tag[bigram] += 1
            else:
                dict_bigram_tag[bigram] = 1
            
        # print(list_bigram_tag)
        # print(str(dict_tag) + "\n" + str(dict_bigram_tag))
    file.close()
    return dict_tag, dict_bigram_tag

# Calculate he probability of the the given word and tag in the training model.
# dict_tagged_token: tagged token dictionary from training data.
# dict_tag: tag dictionary from training data.
# smooth_factor: the smooth factor.
# return: the probability will at least be the smooth factor.
def calculate_prob(dict_tag, dict_tagged_token, tag1, tag2, smooth_factor):
    key = (tag1, tag2)
    print(key)
    if key in dict_tagged_token:
        return (dict_tagged_token[key] / dict_tag[tag1]) + smooth_factor
    else:
        return smooth_factor

# Outpput a table to a specified file name based on a specified training data.
# dict_tagged_token: tagged token dictionary from training data.
# dict_tag: tag dictionary from training data.
def write_table(filename, dict_tag, dict_bigram_tag):
    s1 = "NOUN"
    s2 = "VERB"
    s3 = "CONJ"
    s4 = "PRO"
    s5 = " "
    with open(filename, "w") as file:
        file.write("%-20s|%20s|%20s|%20s|%20s\n" %(s5, s1, s2, s3, s4))
        for tag in dict_tag:
            c0 = tag
            c1 = calculate_prob(dict_tag, dict_bigram_tag, tag, 'N', 0.1)
            c2 = calculate_prob(dict_tag, dict_bigram_tag, tag, 'V', 0.1)
            c3 = calculate_prob(dict_tag, dict_bigram_tag, tag, 'CONJ', 0.1)
            c4 = calculate_prob(dict_tag, dict_bigram_tag, tag, 'PRO', 0.1)
            file.write("%-20s|%20.3f|%20.3f|%20.3f|%20.3f\n" %(c0, c1, c2, c3, c4))
        file.close()


def main():
    dict_tag, dict_bigram_tag = process_file('./hw2_chen7410/Klingon_Train.txt')
    write_table("./hw2_chen7410/Transition.txt", dict_tag, dict_bigram_tag)

main()