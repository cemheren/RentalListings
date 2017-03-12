import pickle
import re
import string


def sentence_to_word_array(sentence="", remove_punctuation=True, to_lowercase=True):
    return_array = list()
    split = list()

    if remove_punctuation:
        split = filter(None, re.split("[,\W\x03\x07\x0b\n\t\r.()' \-!?:/#@*^%&$<>;\"`~{}-]+", sentence))

        for word in split:

            if to_lowercase:
                word = word.lower()

            if remove_punctuation:
                word = word.translate(string.punctuation)

            if len(word) > 0:
                return_array.append(word)

    return return_array


def words_to_numbers(input_matrix=[[]], file_to_write=""):

    return_matrix = list()

    words = dict()
    current_count = 1  # start from 1

    for i in range(len(input_matrix)):
        row = input_matrix[i]

        return_matrix.append(list())

        for k in range(len(row)):
            col = row[k]

            if col not in words:
                words[col] = current_count
                current_count += 1

        return_matrix[i].append(words[col])

    if file_to_write != "":
        pickle.dump(words, open(file_to_write, 'wb'))

    return return_matrix, words


def words_to_numbers_from_old_words_dict(input_matrix=[[]], words=dict(), unk_integer=-1, file_to_read=""):

    if file_to_read != "":
        words = pickle.load(open(file_to_read, 'rb'))

    return_matrix = list()

    for i in range(len(input_matrix)):
        row = input_matrix[i]

        return_matrix.append(list())

        for k in range(len(row)):
            col = row[k]

            value = unk_integer
            if col in words:
                value = words[col]

        return_matrix[i].append(value)

    return return_matrix
