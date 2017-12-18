from embeddings import EmbeddingsDictionary

emb = EmbeddingsDictionary(100000)

#1
print('#1')
print(emb.w2neighbors('geek', 10))

#2
print('#2')
emb.analogy('King', 'woman', 'man')
