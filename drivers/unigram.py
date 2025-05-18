import polars as pl
import pandas as pd
import os
class UnigramDriver:
    """
    Driver que interpreta o modelo Unigram salvo e atribui tags a palavras.
    """
    def __init__(self, data:str = None):
        print("Initializing UnigramDriver")
        self.self_path = os.path.dirname(os.path.abspath(__file__))
        self.data_path = data

        if data is None:
            self.data_path = os.path.join(self.self_path, "../data/models/unigram.csv")

        self.train_data = pl.read_csv(self.data_path)
        self.unk_word_tag = None
        self.numeric_word_tag = None

    #TODO clean
    def fit(self, data:str):
        """
        Carrega o modelo Unigram a partir de um arquivo CSV.
        O arquivo CSV deve conter as colunas 'word' e 'max_tag'
        """
        self.train_data = pl.read_csv(data)
        # Captura as tags de unk-word e numeric-word
        try:
            self.unk_word_tag = self.train_data.filter(pl.col("word") == "unk-word").select("max_tag").item()
        except pl.exceptions.ColumnNotFoundError:
             print("Warning: 'unk-word' not found in the model. Defaulting to UNK tag.")
             self.unk_word_tag = "UNK"
        except Exception as e:
            print(f"Error fetching 'unk-word' tag: {e}. Defaulting to UNK tag.")
            self.unk_word_tag = "UNK"

        try:
            self.numeric_word_tag = self.train_data.filter(pl.col("word") == "numeric-word").select("max_tag").item()
        except pl.exceptions.ColumnNotFoundError:
            print("Warning: 'numeric-word' not found in the model. Defaulting to NUM tag.")
            self.numeric_word_tag = "NUM"
        except Exception as e:
            print(f"Error fetching 'numeric-word' tag: {e}. Defaulting to NUM tag.")
            self.numeric_word_tag = "NUM"

    def tag(self, text:str)-> list[str]:
        """
        Tags the input text with the specified tag.
        """
        tags = []

        for word in text.split():
            original_word = word
            is_numeric = False
            try:
                float(word)
                is_numeric = True
            except ValueError:
                pass

            if is_numeric:
                tagged_word = "{}_{}".format(original_word, self.numeric_word_tag)
                tags.append(tagged_word)
                continue

            # Procura a palavra no dataset
            match = self.train_data.filter(pl.col("word") == word)
            if not match.is_empty():
                tag = match.get_column("max_tag")[0]
                tagged_word = "{}_{}".format(original_word, tag)
                tags.append(tagged_word)
                continue

            # Procura a palavra em lowercase
            match = self.train_data.filter(pl.col("word") == word.lower())
            if not match.is_empty():
                tag = match.get_column("max_tag")[0]
                tagged_word = "{}_{}".format(original_word, tag)
                print(f"Achei lower {original_word} as {tagged_word}")
                tags.append(tagged_word)
                continue

            # Procura a palavra com a primeira letra maiúscula
            match = self.train_data.filter(pl.col("word") == word.capitalize())
            if not match.is_empty():
                tag = match.get_column("max_tag")[0]
                tagged_word = "{}_{}".format(original_word, tag)
                print(f"Achei cap {original_word} as {tagged_word}")
                tags.append(tagged_word)
                continue
            
            # Em último caso, atribui a tag da palavra desconhecida
            tagged_word = "{}_{}".format(original_word, self.unk_word_tag)
            tags.append(tagged_word)

        return tags
    
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

            clean_sentence = self.remove_tags(sentence)
            tagged_sentence = self.tag(clean_sentence)

            # TODO Isso poderia ser assincrono
            for original,tagged in zip(sentence.split(), tagged_sentence):
                word, real_tag = original.rsplit("_", 1)
                _, pred_tag = tagged.rsplit("_", 1) 
                tagged_dataset.append({"id": id,
                                       "word":word,
                                       "real": real_tag, 
                                       "pred": pred_tag})
                
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
    driver = UnigramDriver()
    driver.fit("data/models/unigram.csv")
    tagged_text = driver.tag_dataset("data/raw/Secs19-21 - development")
    print(tagged_text.head())