import polars as pl
from collections import Counter

# Gera um dataset com as colunas "first", "second", "tag" e "count"
def generate_bigram_dataset(data: str):
    """
    Gera um dataset com as colunas "first", "second", "tag" e "count".
    Itera em cada linha do arquivo e adiciona tokens BOS e EOS nos inícios e finais.
    """
    
    pairs = []
    with open(data, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Separa os tokens da linha e adiciona o token BOS no início e EOS no final
            tokens = line.split()
            processed_tokens = [("BOS", "BOS")]
            for token in tokens:
                if "_" not in token:
                    continue
                word, tag = token.rsplit("_", 1)
                try:
                    float(word)
                    word = "numeric-word"
                except Exception:
                    pass
                processed_tokens.append((word, tag))
            processed_tokens.append(("EOS", "EOS"))

            # Cria bigramas a partir dos tokens processados: first token e o token seguinte
            for i in range(len(processed_tokens) - 1):
                first_word = processed_tokens[i][0]
                second_word = processed_tokens[i + 1][0]
                second_tag = processed_tokens[i + 1][1]
                pairs.append((first_word, second_word, second_tag))
    
    counts = Counter(pairs)

    # Cria o dataset com as colunas "first", "second", "tag" e "count"
    df = pl.DataFrame([
        {"first": first, "second": second, "tag": tag, "count": count}
        for (first, second, tag), count in counts.items()
    ])

    df = df.sort("count", descending=True)
    df = df.pivot(on="tag",index=["first","second"],values="count", aggregate_function="sum")

    ## TODO Não revisado
    # Cria coluna de contagem total
    df = df.with_columns(
        appearances = pl.sum_horizontal(col for col in df.columns[2:])
    )

    df = df.fill_null(0)

    # Seleciona apenas a coluna com maior count
    tags = [col for col in df.columns if col not in ["first","second", "appearances"]]
    df = df.with_columns(
        pl.struct(tags).map_elements(lambda s: max(s, key=lambda k: s[k]), return_dtype=pl.Utf8).alias("max_tag")
    )

    df = df[["first", "second", "max_tag", "appearances"]]
        
    return df

if __name__ == "__main__":
    # Gera o dataset de bigramas a partir do arquivo de treinamento
    # e salva em um arquivo CSV
    df = generate_bigram_dataset("data/raw/Secs0-18 - training")
    df.to_pandas().to_csv("data/models/bigram.csv", index=False)