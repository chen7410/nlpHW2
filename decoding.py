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
                # TAG
                if tagged_token[1] in dict_tag:
                    dict_tag[tagged_token[1]] += 1
                else:
                    dict_tag[tagged_token[1]] = 1

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

    file.close()
    return dict_tag, dict_bigram_tag


# Calculate he probability of the the given word and tag in the training model.
# dict_tagged_token: tagged token dictionary from training data.
# dict_tag: tag dictionary from training data.
# smooth_factor: the smooth factor.
# return: the probability will at least be the smooth factor.
def calculate_emission_prob(dict_tag, dict_tagged_token, word, tag, smooth_factor):
    key = (word, tag)
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
    list_tag = list(dict_tag.keys())

    dict_emission = {}
    for key in dict_tagged_token:
        c0 = key[0]
        for tag in list_tag:
            prob = calculate_emission_prob(dict_tag, dict_tagged_token, key[0], tag, smooth_factor)
            emission_key = (c0, tag)
            dict_emission[emission_key] = prob

    return dict_emission

# calculate the transition probility of the specified tag dictionary and bigram tag dictionary
# and return the probilities as a dictionary in the flowing format.
# the probability will at leat be the smooth factor.
# transition dictionary : {(TAG1, TAG2), probility}
# return: transition dictionary
def get_transition_dict(dict_tag, dict_bigram_tag, smooth_factor):
    dict_transition = {}
    for tag_i in dict_tag:
        for tag_j in dict_tag:
            transition_key = (tag_i, tag_j)
            prob = calculate_transition_prob(dict_tag, dict_bigram_tag, tag_i, tag_j, smooth_factor)
            dict_transition[transition_key] = prob
    return dict_transition


# use Viterbi algorithm to generate the POS tags for the test sentence.
# dict_emission : emission dictionary {(word, TAG), probility}
# dict_transition : transition dictionary {(TAG1, TAG2), probility}
# dict_tag_emssion : tag dictionary that has all tags in emission dictionary
# return a tagged sentence as a list
def decode(dict_emission, dict_transition, dict_tag_emssion, test_sentence, smooth_factor):
    list_sent = test_sentence.split()
    list_tag = list(dict_tag_emssion.keys())

    # initialize table
    table = {}
    pointer = {}
    word  = list_sent[0]
    col_index = 0
    for tag in list_tag:
        es_key = (word, tag)
        es_prob = dict_emission.get(es_key, smooth_factor)
        ts_key = ('<s>', tag)
        ts_prob = dict_transition.get(ts_key, smooth_factor)

        vb_prob = es_prob * ts_prob
        table[(0, tag)] = vb_prob
        pointer[(0, tag)] = "0"
    col_index += 1
    # end table initialization

    # iterate the table
    for i in range(1, len(list_sent)):
        list_last_probs = [0] * len(list_tag)
        for j in range(0, len(list_tag)):
            word = list_sent[i]
            max_prob = 0
            back_pointer = ""
            
            for k in range(0, len(list_tag)):
                es_key = (word, list_tag[j])
                es_prob = dict_emission.get(es_key, smooth_factor)
                ts_key = (list_tag[k], list_tag[j])
                ts_prob = dict_transition.get(ts_key, smooth_factor)

                prev_prob = table[i - 1, list_tag[k]]
                tag_prob = es_prob * ts_prob * prev_prob
                # find max probability
                if tag_prob > max_prob:
                    max_prob = tag_prob
                    back_pointer = list_tag[k]

            # save the max probability to the table
            prob = max_prob
            pointer[col_index, list_tag[j]] = back_pointer
            table[col_index, list_tag[j]] = prob
            
            #store the probability of the last column
            list_last_probs[j] = prob
    
        col_index += 1
        
        # find the last back_pointer
        m = max(list_last_probs)
        m_index = 0
        for i in range(0, len(list_last_probs)):
            if m is list_last_probs[i]:
                m_index = i

    # # Sequence identification
    x = len(list_sent) - 1
    result = []
    last_pointer = list_tag[m_index]
    while x > -1:
        tag = pointer[(x, last_pointer)]
        tagged_token = (str(list_sent[x]) + "/" + str(last_pointer))
        last_pointer = tag
        x -= 1
        result.append(tagged_token)    

    result = list(reversed(result))
    return result


def write_result(filename, tagged_sentence):
    with open(filename, 'w') as file:
        for word in tagged_sentence:
            file.write("%s " % word)


def main():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    training_file = "Klingon_Train.txt"

    dict_tag_emssion, dict_tagged_token_emssion = get_emission_data(current_dir + "/" + training_file)
    dict_emission = get_emission_dict(dict_tag_emssion, dict_tagged_token_emssion, 0.1)

    dict_tag_transition, dict_bigram_tag_transition = get_transition_data(current_dir + "/" + training_file)
    dict_transition = get_transition_dict(dict_tag_transition, dict_bigram_tag_transition, 0.1)

    test_sentence = "tera`ngan legh yaS"
    output_file = "Tagged_sentence.txt"
    result = decode(dict_emission, dict_transition, dict_tag_emssion, test_sentence, 0.1)
    write_result(current_dir + "/" + output_file, result)

main()
