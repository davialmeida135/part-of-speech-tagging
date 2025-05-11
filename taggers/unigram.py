
import pandas as pd
class UnigramDriver:
    """
    Driver que interpreta o modelo Unigram salvo e atribui tags a palavras.
    """
    def fit(self, data:str):
        """
        Carrega o modelo Unigram a partir de um arquivo CSV.
        O arquivo CSV deve conter as colunas 'word' e 'max_tag'
        """
        self.train_data = pd.read_csv(data)

    def tag(self, text:str)-> list[str]:
        """
        Tags the input text with the specified tag.
        """
        tags = []
        unk_word_tag = self.train_data.loc[self.train_data['word'] == "unk-word", 'max_tag'].values[0]
        numeric_word_tag = self.train_data.loc[self.train_data['word'] == "numeric-word", 'max_tag'].values[0]
        for word in text.split():
            if word in self.train_data['word'].values:
                tag = self.train_data.loc[self.train_data['word'] == word, 'max_tag'].values[0]
                tagged_word = "{}_{}".format(word, tag)
                tags.append(tagged_word)
                continue
            try:
                float(word)
                tagged_word = "{}_{}".format(word, numeric_word_tag)
                tags.append(tagged_word)
                continue
            except Exception:
                pass
            
            tagged_word = "{}_{}".format(word, unk_word_tag)
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
    driver = UnigramDriver()
    driver.fit("data/models/unigram.csv")
    tagged_text = driver.tag_dataset("data/raw/Secs19-21 - development")
    print(tagged_text.head())