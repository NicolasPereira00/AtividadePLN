"""Executa as três abordagens da atividade e grava os resultados observados."""

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from main import classificar_critica


PASTA = Path(__file__).resolve().parent
ROTULOS = ("NEGATIVO", "NEUTRO", "POSITIVO")
MODELO_PRE_TREINADO = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
PASTA_MODELO_LOCAL = Path.home() / ".cache" / "pln-modelo-cardiff"


def ler_csv(caminho, colunas):
    with caminho.open(encoding="utf-8-sig", newline="") as arquivo:
        leitor = csv.DictReader(arquivo)
        if leitor.fieldnames != list(colunas):
            raise ValueError(f"{caminho.name}: colunas esperadas: {', '.join(colunas)}")
        linhas = list(leitor)
    if not linhas:
        raise ValueError(f"{caminho.name} está vazio")
    for numero, linha in enumerate(linhas, start=2):
        if not linha["texto"].strip() or linha[colunas[1]] not in ROTULOS:
            raise ValueError(f"{caminho.name}, linha {numero}: texto ou rótulo inválido")
    return linhas


def conferir_dados(treino, comparacao):
    if len(treino) < 100:
        raise ValueError("O conjunto de treinamento precisa de pelo menos 100 exemplos")
    if len(comparacao) < 15:
        raise ValueError("A comparação precisa de pelo menos 15 textos")
    contagens = Counter(linha["sentimento"] for linha in treino)
    if any(contagens[rotulo] < 2 for rotulo in ROTULOS):
        raise ValueError("Cada classe precisa de pelo menos dois exemplos")
    textos_treino = [linha["texto"].strip().casefold() for linha in treino]
    textos_comparacao = [linha["texto"].strip().casefold() for linha in comparacao]
    if len(textos_treino) != len(set(textos_treino)):
        raise ValueError("Há textos repetidos nos dados de treinamento")
    if len(textos_comparacao) != len(set(textos_comparacao)):
        raise ValueError("Há textos repetidos na comparação")
    if set(textos_treino) & set(textos_comparacao):
        raise ValueError("Um texto de comparação também aparece no treinamento")
    return contagens


def treinar_e_avaliar(dados):
    textos = [linha["texto"] for linha in dados]
    rotulos = [linha["sentimento"] for linha in dados]

    # O vetor TF-IDF é ajustado apenas no treino, dentro do Pipeline.
    x_treino, x_teste, y_treino, y_teste = train_test_split(
        textos, rotulos, test_size=0.25, random_state=42, stratify=rotulos
    )
    classificador = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode"),
        LogisticRegression(max_iter=1000, random_state=42),
    )
    classificador.fit(x_treino, y_treino)
    previsoes = classificador.predict(x_teste)
    avaliacao = {
        "tamanho_treino": len(x_treino),
        "tamanho_teste": len(x_teste),
        "acuracia_teste": accuracy_score(y_teste, previsoes),
        "relatorio_classificacao": classification_report(
            y_teste, previsoes, labels=ROTULOS, output_dict=True, zero_division=0
        ),
        "rotulos_matriz": ROTULOS,
        "matriz_confusao": confusion_matrix(
            y_teste, previsoes, labels=ROTULOS
        ).tolist(),
    }
    linhas_treino = [
        {"texto": texto, "sentimento": rotulo}
        for texto, rotulo in zip(x_treino, y_treino)
    ]
    linhas_teste = [
        {"texto": texto, "sentimento": rotulo}
        for texto, rotulo in zip(x_teste, y_teste)
    ]
    return classificador, avaliacao, linhas_treino, linhas_teste


def classificar_com_pre_treinado(textos):
    # Este modelo tem três classes e foi ajustado também com textos em português.
    from transformers import AutoTokenizer, pipeline

    arquivos_locais = (
        PASTA_MODELO_LOCAL / "config.json",
        PASTA_MODELO_LOCAL / "sentencepiece.bpe.model",
        PASTA_MODELO_LOCAL / "special_tokens_map.json",
        PASTA_MODELO_LOCAL / "pytorch_model.bin",
    )
    origem = os.environ.get("PLN_MODELO_LOCAL")
    if not origem:
        origem = (
            str(PASTA_MODELO_LOCAL)
            if all(arquivo.is_file() for arquivo in arquivos_locais)
            and arquivos_locais[-1].stat().st_size == 1_112_271_561
            else MODELO_PRE_TREINADO
        )
    tokenizador = AutoTokenizer.from_pretrained(origem, use_fast=False)
    analisador = pipeline(
        "text-classification", model=origem, tokenizer=tokenizador, device=-1
    )
    saidas = analisador(textos, truncation=True, max_length=512, batch_size=8)
    traducao = {"negative": "NEGATIVO", "neutral": "NEUTRO", "positive": "POSITIVO"}
    return [
        (traducao[saida["label"].lower()], round(saida["score"], 4))
        for saida in saidas
    ]


def gravar_csv(caminho, colunas, linhas):
    with caminho.open("w", encoding="utf-8-sig", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=colunas)
        escritor.writeheader()
        escritor.writerows(linhas)


def executar(sem_pre_treinado=False):
    dados = ler_csv(PASTA / "dados_treinamento.csv", ("texto", "sentimento"))
    comparacao = ler_csv(
        PASTA / "textos_comparacao.csv", ("texto", "sentimento_esperado")
    )
    contagens = conferir_dados(dados, comparacao)
    classificador, avaliacao, linhas_treino, linhas_teste = treinar_e_avaliar(dados)
    pasta_saida = PASTA / "resultados"
    pasta_saida.mkdir(exist_ok=True)
    gravar_csv(pasta_saida / "dados_treino.csv", ("texto", "sentimento"), linhas_treino)
    gravar_csv(pasta_saida / "dados_teste.csv", ("texto", "sentimento"), linhas_teste)
    textos = [linha["texto"] for linha in comparacao]
    previsoes_proprias = classificador.predict(textos)
    previsoes_pre = (
        [("", "") for _ in textos]
        if sem_pre_treinado
        else classificar_com_pre_treinado(textos)
    )

    resultados = []
    for linha, proprio, (pre, confianca) in zip(
        comparacao, previsoes_proprias, previsoes_pre
    ):
        resultados.append({
            "texto": linha["texto"],
            "sentimento_esperado": linha["sentimento_esperado"],
            "algoritmo_manual": classificar_critica(linha["texto"])[2],
            "modelo_treinado": proprio,
            "modelo_pre_treinado": pre,
            "confianca_pre_treinado": confianca,
        })

    comparacao_acuracias = {}
    for coluna in ("algoritmo_manual", "modelo_treinado", "modelo_pre_treinado"):
        if coluna == "modelo_pre_treinado" and sem_pre_treinado:
            continue
        comparacao_acuracias[coluna] = sum(
            linha[coluna] == linha["sentimento_esperado"] for linha in resultados
        ) / len(resultados)

    gravar_csv(
        pasta_saida / "comparacao.csv",
        ("texto", "sentimento_esperado", "algoritmo_manual", "modelo_treinado",
         "modelo_pre_treinado", "confianca_pre_treinado"),
        resultados,
    )
    gravar_csv(
        pasta_saida / "matriz_confusao.csv",
        ("esperado/predito", *ROTULOS),
        [
            dict(zip(("esperado/predito", *ROTULOS), (rotulo, *valores)))
            for rotulo, valores in zip(ROTULOS, avaliacao["matriz_confusao"])
        ],
    )
    resumo = {
        "dataset": {"total": len(dados), "por_classe": dict(contagens)},
        "modelo_pre_treinado": None if sem_pre_treinado else MODELO_PRE_TREINADO,
        "avaliacao_modelo_treinado": avaliacao,
        "acuracias_comparacao_18_textos": comparacao_acuracias,
    }
    (pasta_saida / "avaliacao.json").write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Dados: {len(dados)} ({dict(contagens)})")
    print(f"Treino: {len(linhas_treino)} textos em {pasta_saida / 'dados_treino.csv'}")
    print(f"Teste: {len(linhas_teste)} textos em {pasta_saida / 'dados_teste.csv'}")
    print(f"Teste separado: {avaliacao['tamanho_teste']} textos")
    print(f"Acurácia no teste: {avaliacao['acuracia_teste']:.1%}")
    print("Acurácias nos 18 textos de comparação:")
    for metodo, acuracia in comparacao_acuracias.items():
        print(f"  {metodo}: {acuracia:.1%}")
    if not sem_pre_treinado:
        iguais = [
            linha for linha in resultados
            if len({linha["algoritmo_manual"], linha["modelo_treinado"],
                    linha["modelo_pre_treinado"]}) == 1
        ]
        diferentes = [linha for linha in resultados if linha not in iguais]
        print("Exemplos para a demonstração:")
        for linha in iguais[:1] + diferentes[:3]:
            print(f"  {linha['texto']}")
            print(
                f"    esperado={linha['sentimento_esperado']}; "
                f"manual={linha['algoritmo_manual']}; "
                f"treinado={linha['modelo_treinado']}; "
                f"pré-treinado={linha['modelo_pre_treinado']}"
            )
    print(f"Arquivos gerados em: {pasta_saida}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sem-pre-treinado", action="store_true",
        help="executa as etapas locais sem baixar o modelo da Hugging Face",
    )
    argumentos = parser.parse_args()
    executar(sem_pre_treinado=argumentos.sem_pre_treinado)
