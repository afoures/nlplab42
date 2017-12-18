import logging
import numpy
import torch

class EmbeddingsDictionary:
    '''
    This class is a container for word embedding data. Properties:
    * a list of strings mapping index to token: self.words[i] = token
    * a dictionary with the reverse mapping: self.dictionary[token] = i
    * a float tensor of shape [n_embeddings, dim]: self.emb[i] = <data> where
    <data> is the embedding of the token self.words[i]
    '''
    def __init__(self,
                 max_words=None,
                 path='data/wiki-news-300d-1M.vec',
                 normalize=True,
                 word_whitelist=None):
        self.load_embeddings(max_words, path, normalize, word_whitelist)

    def load_embeddings(self, max_words, path, normalize, word_whitelist):
        '''
        Load the embeddings contained in path.
        '''
        logging.info('Loading embeddings')
        with open(path, 'r') as f:
            first_line = f.readline().split()
            n_embeddings = int(first_line[0])
            if max_words:
                n_embeddings = min(max_words, n_embeddings)
            if word_whitelist:
                n_embeddings = min(len(word_whitelist), n_embeddings)
            dim = int(first_line[1])
            tensor = torch.zeros(n_embeddings, dim)
            dictionary = {}
            words = []
            i = 0
            for line in f:
                word, data = line.split(' ', 1)
                if word_whitelist and word not in word_whitelist:
                    continue
                data = numpy.fromstring(data, sep=' ')

                assert(len(data) == dim)
                dictionary[word] = i
                tensor[i] = torch.from_numpy(data)
                if normalize:
                    tensor[i] = tensor[i] / tensor[i].norm(2)
                words.append(word)
                i += 1
                if i > 0 and i % 50000 == 0:
                    logging.info('Loading {} / {} embeddings'.format(i, n_embeddings))
                if i == n_embeddings:
                    break
            logging.info('Loaded {} embeddings'.format(n_embeddings))
            self.words = words
            self.dictionary = dictionary
            self.emb = tensor

    def embed(self, token):
        return self.emb[self.dictionary[token]]

    def emb2neighbors(self, query_embedding, top_k=20):
        '''
        Retreive the index i of the nearest-neighbors to query_embedding.
        Input: query_embedding must be a tensor of shape [dim]
        Output: a tuple (score, index), with score a FloatTensor of shape
        [top_k] and index a LongTensor of shape [top_k] that maps to an index
        in self.words.
        '''
        return torch.mm(self.emb, query_embedding.unsqueeze(1)).squeeze(1).topk(k=top_k)

    def w2neighbors(self, query_word, top_k=20):
        '''
        Retrieve the nearest neighbors to a given word.
        Input: query_word, a string.
        Output: a list of strings neighbor_words
        Raises an error if query_word is not in self.dictionary.
        '''
        query_id = self.dictionary[query_word]
        _, neighbor_ids = self.emb2neighbors(self.emb[query_id], top_k + 1)
        neighbor_words = [self.words[i] for i in neighbor_ids if i != query_id]
        return neighbor_words

    def analogy(self, word1, word2, word3):
        ind1 = self.dictionary[word1]
        ind2 = self.dictionary[word2]
        ind3 = self.dictionary[word3]

        target_embedding = self.emb[ind1] + self.emb[ind2] - self.emb[ind3]
        _score, closest_word_index = self.emb2neighbors(target_embedding, top_k=5)
        for ind in closest_word_index:
            if (ind != ind1 and ind != ind2 and ind != ind3):
                print(self.words[ind])
                break
