from unigram import UnigramDriver
import pandas as pd
import re
class BigramDriver:
    """
    Driver que interpreta o modelo Unigram salvo e atribui tags a palavras.
    """
    def __init__(self):
        self.unigram_driver = UnigramDriver()
        self.unigram_driver.fit("data/models/unigram.csv")

    def fit(self, data:str):
        """
        Carrega o modelo Unigram a partir de um arquivo CSV.
        O arquivo CSV deve conter as colunas 'first', 'second' e 'max_tag'
        """
        self.train_data = pd.read_csv(data)

    def tag(self, text:str)-> list[str]:
        """
        Tags the input text with the specified tag.
        """
        # Adiciona BOS e EOS
        text = "BOS " + text
        tags = []
        text = text.split()
        for i in range(len(text) - 1):
            first_word = text[i] # Previous word
            second_word = text[i + 1] # Word to be tagged

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

            # Tenta encontrar a tupla completa
            try:
                tag = self.train_data.loc[(self.train_data['first'] == first_word) & (self.train_data['second'] == second_word), 'max_tag'].values[0]
                tagged_word = "{}_{}".format(second_word, tag)
                tags.append(tagged_word)
                continue
            except Exception:
                pass
            
            # Tenta encontrar apenas o primeiro token
            try:
                tag = self.train_data.loc[(self.train_data['first'] == first_word) & (self.train_data['second'] == "unk-word"), 'max_tag'].values[0]
                tagged_word = "{}_{}".format(second_word, tag)
                tags.append(tagged_word)
                continue
            except Exception:
                pass

            # Em ultimo caso, voltamos para o modelo unigram
            tagged_word = self.unigram_driver.tag(second_word)
            tags.extend(tagged_word)

        return tags
    
    # TODO
    def tag_dataset(self, dataset:str)-> pd.DataFrame:
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
        df.to_csv("data/runs/unigram_dev.csv", index=False)
        return df
    
    def remove_tags(self, text):
        """
        Retorna o texto sem tags.
        Exemplo: "closely_RB watching_VBG" -> "closely watching"
        """
        return " ".join([word.split("_")[0] for word in text.split()])

if __name__ == "__main__":
    # Example usage
    driver = BigramDriver()
    driver.fit("data/models/bigram.csv")
    tags = driver.tag("closely watching a super man")
    print(tags)
    #tagged_text = driver.tag_dataset("data/raw/Secs19-21 - development")
    #print(tagged_text.head())
