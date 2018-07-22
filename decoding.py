from __future__ import division
import nltk
from nltk import word_tokenize
import os

# train a language from a give file name and return a 
# tag dictionary and tagged token dictionary in the flowing from.
# tag dictionary : {TAG, count}
# tagged toekn dictionary : {word/TAG, count}
# returns tag dictionary, tagged token dictionary.
def get_emission_data(filename): 
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

        # print(dict_tagged_token)
        # print(dict_tag) 
    file.close()
    return dict_tag, dict_tagged_token

        
# train a language from a give file name and return a 
# tag dictionary and bigram tag dictionary in the flowing from.
# the returned tag dictionary includes start tag '<s>' and end tag '</s>'
# bigram tag dictionary : {(TAG1, TAG2), count}
# tagged toekn dictionary : {word/TAG, count}
# returns : tag dictionary, bigram tag dictionary.
def get_transition_data(filename):
    dict_tag = {}
    dict_bigram_tag = {}
    
    with open(filename, 'r') as file:
        for line in file:
            list_tag = ["<s>"]
            
            for token in line.split():
                tagged_token = nltk.tag.str2tuple(token)
                list_tag.append(tagged_token[1])
            list_tag.append("</s>")
            # print(list_tag)

            for tag in list_tag:
                if tag in dict_tag:
                    dict_tag[tag] += 1
                else:
                    dict_tag[tag] = 1

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
def calculate_emission_prob(dict_tag, dict_tagged_token, word, tag, smooth_factor):
    key = (word, tag)
    # print(key)
    if key in dict_tagged_token:
        return (dict_tagged_token[key] / dict_tag[tag]) + smooth_factor
    else:
        return smooth_factor


# Calculate the probability of the the given word and tag in the training model.
# dict_tagged_token: tagged token dictionary from training data.
# dict_tag: tag dictionary from training data.
# smooth_factor: the smooth factor.
# return: the probability will at least be the smooth factor.
def calculate_transition_prob(dict_tag, dict_bigram_tag, tag1, tag2, smooth_factor):
    key = (tag1, tag2)
    # print(key)
    if key in dict_bigram_tag:
        return (dict_bigram_tag[key] / dict_tag[tag1]) + smooth_factor
    else:
        return smooth_factor

# calculate the emission probility of the specified tag dictionary and tagged token dictionary
# and return the probilities as a dictionary in the flowing format.
# the probability will at leat be the smooth factor.
# emission dictionary : {(word, TAG), probility}
# return: emission dictionary
def get_emission_dict(dict_tag, dict_tagged_token, smooth_factor):
    print("----------------------------Emission______________________________")
    list_tag = list(dict_tag.keys())
    # print(end="          ")
    # for tag in list_tag:
    #     print(tag, end="          ")
    # print("\n")

    dict_emission = {}
    for key in dict_tagged_token:
        c0 = key[0]
        for tag in list_tag:
            prob = calculate_emission_prob(dict_tag, dict_tagged_token, key[0], tag, smooth_factor)
            # print(prob, end="          ")
            emission_key = (c0, tag)
            dict_emission[emission_key] = prob
        # print("\n")
    print(dict_emission)
    return dict_emission

# calculate the transition probility of the specified tag dictionary and bigram tag dictionary
# and return the probilities as a dictionary in the flowing format.
# the probability will at leat be the smooth factor.
# transition dictionary : {(TAG1, TAG2), probility}
# return: transition dictionary
def get_transition_dict(dict_tag, dict_bigram_tag, smooth_factor):
    print("----------------------------Transition______________________________")
    dict_transition = {}
    for tag_i in dict_tag:
        for tag_j in dict_tag:
            transition_key = (tag_i, tag_j)
            prob = calculate_transition_prob(dict_tag, dict_bigram_tag, tag_i, tag_j, smooth_factor)
            dict_transition[transition_key] = prob
    print(dict_transition)
    return dict_transition


# Let T = # of part-of-speech tags
#       W = # of words in the sentence
# /* Initialization Step */
# for t = 1 to T
#        Score(t, 1) = Pr(Word1| Tagt) * Pr(Tagt| φ)
#        BackPtr(t, 1) = 0;
# /* Iteration Step */
# for w = 2 to W
#     for t = 1 to T
#         Score(t, w) = Pr(Wordw| Tagt) *MAXj=1,T(Score(j, w-1) * Pr(Tagt| Tagj))    
#         BackPtr(t, w) = index of j that gave the max above
# /* Sequence Identification */
# Seq(W ) = t that maximizes Score(t,W ) 
# for w = W -1 to 1
#     Seq(w) = BackPtr(Seq(w+1),w+1)
def decode(dict_emission, dict_transition, dict_tag_emssion, test_sentence):
    print("----------------------------Decode______________________________")
    list_sent = test_sentence.split()
    list_tag = list(dict_tag_emssion.keys())

    # initialize table
    table = []
    for i in range(len(list_tag)):
        table.append([])

    # calculate first column    
    word  = list_sent[0]
    j = 0
    for tag in list_tag:
        es_key = (word, tag)
        print("es_key = " + str(es_key))
        es_prob = dict_emission.get(es_key, 0)
        ts_key = ('<s>', tag)
        ts_prob = dict_transition.get(ts_key, 0)

        vb_prob = es_prob * ts_prob
        table[j].append(vb_prob)
        j += 1
    print(table)
    # end table initialization

    # iterate the table
    # for i = 1 in range(len(list_sent)):
    #     for tag in list_tag:
    #         es_key = (list_sent[i], tag)
    #         es_prob = dict_emission.get(es_key, 0)

    #         for tag in list_tag:






    # return table




def main():
    print("----------------------------start______________________________")
    current_dir = os.path.dirname(os.path.realpath(__file__))
    filename = "Klingon_Train.txt"

    dict_tag_emssion, dict_tagged_token_emssion = get_emission_data(current_dir + "/" + filename)
    dict_emission = get_emission_dict(dict_tag_emssion, dict_tagged_token_emssion, 0.1)

    dict_tag_transition, dict_bigram_tag_transition = get_transition_data(current_dir + "/" + filename)
    dict_transition = get_transition_dict(dict_tag_transition, dict_bigram_tag_transition, 0.1)

    test_sentence = "tera`ngan legh yaS"

    decode(dict_emission, dict_transition, dict_tag_emssion, test_sentence)







main()
