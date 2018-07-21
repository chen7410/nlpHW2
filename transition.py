from __future__ import division
import nltk
from nltk import bigrams
from nltk import word_tokenize

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

def calculate_prob(dict_tag, dict_tagged_token, tag1, tag2, smooth_factor):
    key = (tag1, tag2)
    print(key)
    if key in dict_tagged_token:
        return (dict_tagged_token[key] / dict_tag[tag1]) + smooth_factor
    else:
        return smooth_factor


def write_table(filename, dict_tag, dict_bigram_tag, smooth_factor):
    s1 = "NOUN"
    s2 = "VERB"
    s3 = "CONJ"
    s4 = "PRO"
    s5 = " "
    with open(filename, "w") as file:
        file.write("%-20s|%20s|%20s|%20s|%20s\n" %(s5, s1, s2, s3, s4))
        for tag in dict_tag:
            c0 = tag
            c1 = calculate_prob(dict_tag, dict_bigram_tag, tag, 'N', smooth_factor)
            c2 = calculate_prob(dict_tag, dict_bigram_tag, tag, 'V', smooth_factor)
            c3 = calculate_prob(dict_tag, dict_bigram_tag, tag, 'CONJ', smooth_factor)
            c4 = calculate_prob(dict_tag, dict_bigram_tag, tag, 'PRO', smooth_factor)
            file.write("%-20s|%20.3f|%20.3f|%20.3f|%20.3f\n" %(c0, c1, c2, c3, c4))
        file.close()


def main():
    dict_tag, dict_bigram_tag = process_file('./hw2_chen7410/Klingon_Train.txt')
    write_table("./hw2_chen7410/Transition.txt", dict_tag, dict_bigram_tag, 0.1)

main()