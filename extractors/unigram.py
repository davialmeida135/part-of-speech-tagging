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
    except:
        pass
    pairs.append((word, tag))

counts = Counter(pairs)

df = pl.DataFrame([
    {"word": word, "tag": tag, "count": count}
    for (word, tag), count in counts.items()
])

df = df.sort(["count"], descending=True)
df = df.pivot(on="tag",index="word",values="count")

# Cria coluna de contagem total
df = df.with_columns(
    appearances = pl.sum_horizontal(col for col in df.columns[1:])
)

# Transforma todas palavras com count=1 em unk-word
df = df.with_columns(

)

# Seleciona apenas a coluna com maior count


df.to_pandas().to_csv("data/dataset/unigram_pl.csv")
