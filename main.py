import math

from minitorch.matrix import Matrix
from minitorch.nn.linear import Linear

from transformer.tokenizer import Tokenizer

def main():
    t = Tokenizer("shakespeare.txt", encoding="utf-8")
    n_embedding = math.ceil(math.sqrt(math.sqrt(t.vocab_len)))
    print(n_embedding)
    l = Linear(t.vocab_len, n_embedding, False)
    test = Matrix.one_hot(3, t.vocab_len)
    print(l(test))

if __name__ == "__main__":
    main()