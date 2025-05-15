# part-of-speech-tagging
Trabalho da disciplina de processamento de linguagem natural, UFRN 2025.1

## O objetivo
O objetivo do trabalho é implementar um part of speech tagger que classifique as palavras em uma frase de acordo com sua classe gramatical (substantivo, adjetivo etc.). 
Os experimentos foram realizados com mais de uma técnica de treinamento e inferência, como o uso de unigramas, bigramas, trigramas e o condicionamento por tag no lugar de por palavra. 

## Os dados
O [corpus utilizado](https://drive.google.com/drive/folders/19_F8mmI65lWnL6BmKvtzMX2Z_tcNlXxb) é composto por textos tratados do [Penn Treebank](https://paperswithcode.com/dataset/penn-treebank), com anotações no formato PALAVRA_TAG.

## O treinamento
A "fase de treinamento", nesse contexto, é composta pela geração de um dataset que agregue informações específicas sobre cada palavra ou série de palavras no corpus. Esses datasets são gerados pelos programas na pasta `extractors`.

## A inferência
A fase de inferência de tags é executada por um módulo de inferência chamado *driver*. O driver utiliza o dataset gerado na fase de treinamento e os conjuntos de desenvolvimento e teste do corpus para predizer a qual tag cada palavra pertence. Ao fim da fase de inferência, é gerado um dataset com as colunas `|Palavra|Tag Real|Tag Inferida|` para facilitar a análise futura dos resultados obtidos.

## Experimentos
### Unigrama
O dataset gerado pelo extractor de unigrama tem o seguinte formato:
`|Palavra|Tag Máxima|`

Em que "Tag Máxima" representa qual a tag que foi mais atribuída à palavra no corpus de treinamento.
- Alterar o threshold de palavra desconhecida

### Bigram word
- Usar unk word x ir direto pro unigram (?)
- Unk word na 1a x unk word na 2a
- Bigram tag

### Trigram word
- Trigram tag