from __future__ import division
import nltk
from nltk import word_tokenize


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
                # print(str(tagged_token) + " : " + str(dict_tagged_token.get(tagged_token)))
                # TAG
                if tagged_token[1] in dict_tag:
                    dict_tag[tagged_token[1]] += 1
                else:
                    dict_tag[tagged_token[1]] = 1
                # print(str(tagged_token[1]) + " : " +str(dict_tag[tagged_token[1]])) 

        print(dict_tagged_token)
        # print(dict_tag) 
    file.close()
    return dict_tag, dict_tagged_token
    
# Calculate he probability of the the given word and tag in the training model.
# dict_tagged_token: tagged token dictionary from training data.
# dict_tag: tag dictionary from training data.
# return: the probability will be at least 0.1.
def calculate_prob(dict_tag, dict_tagged_token, word, tag):
    key = (word, tag)
    # print(key)
    if key in dict_tagged_token:
        return (dict_tagged_token[key] / dict_tag[tag]) + 0.1
    else:
        return 0.1

# Outpput a table to a specified file name based on a specified training data.
# dict_tagged_token: tagged token dictionary from training data.
# dict_tag: tag dictionary from training data.
def write_table(filename, dict_tag, dict_tagged_token):
    dict_tag, dict_tagged_token = process_file("./hw2_chen7410/Klingon_Train.txt")
    s1 = "NOUN"
    s2 = "VERB"
    s3 = "CONJ"
    s4 = "PRO"
    s5 = " "
    with open(filename, "w") as file:
        file.write("%-20s|%20s|%20s|%20s|%20s\n" %(s5, s1, s2, s3, s4))
        for key in dict_tagged_token:
            c0 = key[0]
            c1 = calculate_prob(dict_tag, dict_tagged_token, key[0], 'N')
            c2 = calculate_prob(dict_tag, dict_tagged_token, key[0], 'V')
            c3 = calculate_prob(dict_tag, dict_tagged_token, key[0], 'CONJ')
            c4 = calculate_prob(dict_tag, dict_tagged_token, key[0], 'PRO')
            # print("TABLE = " + str(key))
            
            file.write("%-20s|%20.3f|%20.3f|%20.3f|%20.3f\n" %(c0, c1, c2, c3, c4))

    file.close()


def main():
    dict_tag, dict_tagged_token = process_file("./hw2_chen7410/Klingon_Train.txt")
    write_table("./hw2_chen7410/Emission.txt", dict_tag, dict_tagged_token)
    # print(dict_tag['N'], dict_tag['V'], dict_tag['PRO'], dict_tag['CONJ'])


main()

