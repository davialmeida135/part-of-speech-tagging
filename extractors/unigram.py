import polars as pl
from collections import Counter

with open("data/raw/Secs0-18 - training", "r", encoding="utf-8") as f:
    content = f.read()

# Separa todos os tokens
tokens = content.strip().split()
pairs = []
for token in tokens:
    if "_" not in token:
        continue
    word, tag = token.rsplit("_", 1)
    try:
        word = float(word)
        word = "numeric-word"
    except Exception:
        pass
    pairs.append((word, tag))

counts = Counter(pairs)

df = pl.DataFrame([
    {"word": word, "tag": tag, "count": count}
    for (word, tag), count in counts.items()
])

# Transforma todas palavras com count=1 em unk-word
df = df.with_columns(
    pl.when(pl.col("count") == 1)
      .then(pl.lit("unk-word"))
      .otherwise(pl.col("word"))
      .alias("word")
)

df = df.sort(["count"], descending=True)
df = df.pivot(on="tag",index="word",values="count", aggregate_function="sum")

# Cria coluna de contagem total
df = df.with_columns(
    appearances = pl.sum_horizontal(col for col in df.columns[1:])
)

df = df.fill_null(0)

# Seleciona apenas a coluna com maior count
tags = [col for col in df.columns if col not in ["word", "appearances"]]
df = df.with_columns(
    pl.struct(tags).map_elements(lambda s: max(s, key=lambda k: s[k]), return_dtype=pl.Utf8).alias("max_tag")
)

df = df[["word", "max_tag", "appearances"]]

df.to_pandas().to_csv("data/dataset/unigram.csv")
