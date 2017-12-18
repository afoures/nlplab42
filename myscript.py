from embeddings import EmbeddingsDictionary

emb = EmbeddingsDictionary(100000)

#1
print('#1')
print(emb.w2neighbors('geek', 10))

print()
#2
print('#2')
emb.analogy('King', 'woman', 'man')
print()
emb.analogy('sushi', 'Rome', 'Tokyo')
print()
emb.analogy('uncle', 'woman', 'man')
print()
emb.analogy('puppy', 'cat', 'dog')
