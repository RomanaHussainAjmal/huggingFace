
from sentence_transformers import SentenceTransformer
from pprint import pprint
from sklearn.metrics.pairwise import cosine_similarity


sentences = ["I like coding", "Java is easy", "India ia beutiful country"]
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
sentence_embeddings = model.encode(sentences)
print(sentence_embeddings)

pprint('Similarity between : {}  and : {}  is {}'.format(sentences[0],
       sentences[1],
       cosine_similarity(sentence_embeddings[0].reshape(1, -1),
       sentence_embeddings[1].reshape(1, -1))[0][0]))

pprint('Similarity between : {}  and : {}  is {}'.format(sentences[1],
        sentences[2],
        cosine_similarity(sentence_embeddings[1].reshape(1, -1),
        sentence_embeddings[2].reshape(1, -1))[0][0]))