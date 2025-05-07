import polars as pl
from collections import Counter

with open("data/raw/Secs0-18 - training", "r", encoding="utf-8") as f:
    content = f.read()

tokens = content.strip().split()
pairs = []
for token in tokens:
    if "_" not in token:
        continue
    word, tag = token.rsplit("_", 1)
    pairs.append((word, tag))

counts = Counter(pairs)

df = pl.DataFrame([
    {"word": word, "tag": tag, "count": count}
    for (word, tag), count in counts.items()
])

df = df.sort(["count"], descending=True)
df = df.pivot(on="tag",index="word",values="count")

print(df)
df.to_pandas().to_csv("data/dataset/unigram_pl.csv")
