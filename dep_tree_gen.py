import numpy as np

# Generate dependency tree for single sentence
def tree_generate(sentence, processor):
    doc = processor(sentence)
    return [(token.i, token.dep_, token.head.i) for token in doc]


def string_compare(str1, str2):
    return 1 if str1 == str2 else 0

def relation_compare(re1, re2, alpha):
    return alpha if re1 == re2 else 1


def dep_matrix_gen(sentence1, sentence2, processor, alpha):
    """
    sentence1, sentence2: two input texts with lengths n, m respectively
    processor: spacy processor for english
    alpha: defined in dependency integrating formula

    return: dependency integrating matrix [(n+m), (n+m)] 
    """

    dep1 = tree_generate(sentence1, processor)
    dep2 = tree_generate(sentence2, processor)
    n, m = len(dep1), len(dep2)

    tokens1 = [token.text for token in processor(sentence1)]
    tokens2 = [token.text for token in processor(sentence2)]


    # Calculate depdency matrix [(n+m), (n+m)]
    dep_matrix = np.zeros((n+m, n+m))

    for i in range(n):
        tri1 = dep1[i]
        dep_matrix[tri1[0]][tri1[2]] = 1
        for j in range(m):
            tri2 = dep2[j]
            dep_matrix[tri2[0]+n][tri2[2]+n] = 1

            tmp = string_compare(tokens1[tri1[0]], tokens2[tri2[0]]) + string_compare(tokens1[tri1[2]], tokens2[tri2[2]])
            tmp = tmp * relation_compare(tri1[1], tri2[1], alpha)

            if relation_compare(tri1[1], tri2[1], alpha) != 1:
                tmp = max(tmp, alpha)

            dep_matrix[i][n+j] = tmp
            dep_matrix[n+j][i] = tmp
    return dep_matrix


def subwords_to_words(offset_map, spa_doc):
    """
    offset_map: offset_mapping from Bert tokenizer
    spa_doc: dependency tree from spacy

    return: sub2w - mapping subword to word
    """
    sub2w = []
    for start, end in offset_map:
        if start == end:
            sub2w.append(-1)
            continue

        for word in spa_doc:
            if word.idx <= start <= word.idx + len(word.text):
                sub2w.append(word.i)
                break

        else:
            sub2w.append(-1)
    
    return sub2w

def subword_dep_matrix(sentence1, sentence2, processor, tokenizer, alpha=2, seq_len = 512):
    """
    return dependency matrix for passing to the model [seq_len, seq_len]
    """

    spacy_doc1 = processor(sentence1)   # n
    spacy_doc2 = processor(sentence2)   # m
    
    n, m = len(spacy_doc1), len(spacy_doc2)

    tokens1 = tokenizer(sentence1, return_offsets_mapping=True, max_length=seq_len//2, truncation=True)  
    tokens2 = tokenizer(sentence2, return_offsets_mapping=True, max_length=seq_len//2, truncation=True)  

    offset_mapping1 = tokens1['offset_mapping'] # including [CLS] and [SEP]
    offset_mapping2 = tokens2['offset_mapping'] # including [CLS] and [SEP]

    sub2w1 = subwords_to_words(offset_mapping1, spacy_doc1) 
    x = len(sub2w1)
    sub2w2 = subwords_to_words(offset_mapping2, spacy_doc2) 
    y = len(sub2w2)

    init_dep_matrix = dep_matrix_gen(sentence1, sentence2, processor, alpha)    # [(n+m), (n+m)]

    matrix = np.zeros((seq_len,seq_len))

    for i in range(x-1):
        idx1 = sub2w1[i]
        
        if idx1 == -1:
            continue
        
        for j in range(x+y-1):
            if j<x-1:
                idx2 = sub2w1[j]
                if idx2 == -1:
                    continue
                matrix[i][j] = init_dep_matrix[idx1][idx2]
            else:
                idx2 = sub2w2[j-x+1]
                if idx2 == -1:
                    continue
                matrix[i][j] = init_dep_matrix[idx1][n+idx2]

    for i in range(y):
        idx1 = sub2w2[i]
        if idx1 == -1:
            continue
        for j in range(x+y-1):
            if j<x-1:
                idx2 = sub2w1[j]
                if idx2 == -1:
                    continue
                matrix[i+x-1][j] = init_dep_matrix[n+idx1][idx2]
            else:
                idx2 = sub2w2[j-x+1]
                if idx2 == -1:
                    continue
                matrix[i+x-1][j] = init_dep_matrix[n+idx1][n+idx2]
    
    return matrix


# sentence1 = "I love apples"
# sentence2 = "I enjoy bananas"

# print(subword_dep_matrix(sentence1, sentence2, alpha=2, seq_len=16))