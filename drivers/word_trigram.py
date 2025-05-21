from word_bigram import BigramDriver
import pandas as pd
import polars as pl
import os
class TrigramDriver:
    """
    Driver que interpreta o modelo Unigram salvo e atribui tags a palavras.
    """
    def __init__(self, data:str = None, bigram:str = None, unigram:str = None):
        print("Initializing TrigramDriver")
        self.self_path = os.path.dirname(os.path.abspath(__file__))
        self.data_path = data
        self.bigram_path = bigram
        self.unigram_path = unigram

        if data is None:
            self.data_path = os.path.join(self.self_path, "../data/models/trigram.csv")
        if bigram is None:
            self.bigram_path = os.path.join(self.self_path, "../data/models/bigram.csv")
        if unigram is None:
            self.unigram_path = os.path.join(self.self_path, "../data/models/unigram.csv")

        self.bigram_driver = BigramDriver(self.bigram_path,self.unigram_path)
        self.train_data = pl.read_csv(self.data_path)

    def fit(self, data:str):
        """
        Carrega o modelo Trigram a partir de um arquivo CSV.
        O arquivo CSV deve conter as colunas 'first', 'second', 'third' e 'max_tag'
        """
        self.train_data = pl.read_csv(data)

    def tag(self, text:str)-> list[str]:
        """
        Gera tags para o texto fornecido.
        """

        text = "BOS BOS " + text
        tags = []
        text = text.split()
        for i in range(len(text) - 2):
            first_word = text[i] # Previous word
            second_word = text[i + 1] # Previous word
            third_word = text[i + 2] # Word to be tagged
            original_third_word = third_word

            # Verifica se o primeiro token é um número
            try:
                float(first_word)
                first_word = "numeric-word"
            except Exception:
                pass
            # Verifica se o segundo token é um número
            try:
                float(second_word)
                second_word = "numeric-word"
            except Exception:
                pass
            # Verifica se o terceiro token é um número
            try:
                float(third_word)
                third_word = "numeric-word"
            except Exception:
                pass

            tagged = False

            # Tenta encontrar a tupla completa (first_word, second_word)
            match = self.train_data.filter(
                (pl.col("first") == first_word) & (pl.col("second") == second_word) & (pl.col("third") == third_word)
            )
            if not match.is_empty():
                tag = match.get_column("max_tag")[0]
                tagged_word = "{}_{}".format(original_third_word, tag)
                #print("Found trigram: ", first_word, second_word, third_word, tag)
                tags.append(tagged_word)
                tagged = True
            
            # Em ultimo caso, voltamos para o modelo bigram
            if not tagged:
                bigram_tagged_list = self.bigram_driver.tag(second_word+" "+original_third_word) 
                #print("Using bigram: ", second_word, original_third_word, bigram_tagged_list[0])
                tags.append(bigram_tagged_list[-1])

        return tags
    
    def tag_dataset(self, dataset:str, output:str = "data/runs/trigram_dev.csv")-> pd.DataFrame:
        """
        Atribui tags a um dataset de palavras.
        Retorna um DataFrame com as colunas 'id', 'word', 'real' e 'pred'.
        """
        with open(dataset, "r", encoding="utf-8") as f:
            content = f.read()

        # Separa todos os tokens
        sentences = content.strip().split('\n')
        tagged_dataset = []
        for id,sentence in enumerate(sentences):

            cleaned_sentence = self.remove_tags(sentence)
            tagged_sentence = self.tag(cleaned_sentence)

            # TODO Isso poderia ser assincrono
            for original,tagged in zip(sentence.split(), tagged_sentence):
                tagged_dataset.append({"id": id,
                                       "word":original.rsplit("_")[0],
                                       "real": original.rsplit("_")[-1], 
                                       "pred": tagged.rsplit("_")[-1]})
                
        # Cria DataFrame
        df = pd.DataFrame(tagged_dataset)
        df.to_csv("data/runs/trigram_test.csv", index=False)
        return df
    
    def remove_tags(self, text):
        """
        Retorna o texto sem tags.
        Exemplo: "closely_RB watching_VBG" -> "closely watching"
        """
        return " ".join([word.split("_")[0] for word in text.split()])

if __name__ == "__main__":
    # Example usage
    driver = TrigramDriver()
    df = driver.tag_dataset("data/raw/Secs22-24 - testing")
    print(df.head())
    #tagged_text = driver.tag_dataset("data/raw/Secs19-21 - development")
    #print(tagged_text.head())
