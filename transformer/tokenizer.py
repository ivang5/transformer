class Tokenizer:
    def __init__(self, input_txt_path: str, encoding: str | None = None) -> None:
        with open(input_txt_path, "r", encoding=encoding) as f:
            input_txt = f.read()
            self.vocab = sorted(list(set(input_txt)))
            self.vocab_len = len(self.vocab)

        self.ctoi = { c:i for i, c in enumerate(self.vocab)}
        self.itoc = { i:c for i, c in enumerate(self.vocab)}

    def encode(self, text: str) -> int:
        assert isinstance(text, str), f"Cannot encode {type(text)}. Expected: str."
        return [self.ctoi[c] for c in text]
        
    def decode(self, text_encoded: list[int]) -> str:
        assert all(isinstance(i, int) for i in text_encoded), f"Cannot encode {type(text_encoded)}. Expected: list[int]."
        return [self.itoc[i] for i in text_encoded]
