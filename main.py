"""Atividade prática: classificador simples de sentimentos em críticas de filmes."""
# Nicolas Martins.
# Guilherme Gheno.

import string
import sys


STOPWORDS = ['o', 'a', 'os', 'as', 'e', 'de', 'do', 'da', 'foi']

POSITIVAS = [
    'bom', 'boa', 'excelente', 'maravilhoso', 'ótimo', 'ótima',
    'divertido', 'incrível',
]

NEGATIVAS = [
    'ruim', 'péssimo', 'péssima', 'horrível', 'chato',
    'terrível', 'fraco', 'decepcionante',
]

PONTUACAO = string.punctuation + '“”‘’«»—–…'


def preparar_texto(texto):
    """Converte para minúsculas, remove pontuação e separa em palavras."""
    texto = texto.lower()
    tabela = str.maketrans({sinal: ' ' for sinal in PONTUACAO})
    texto = texto.translate(tabela)
    return texto.split()


def remover_stopwords(palavras):
    """Retorna apenas as palavras que não estão na lista de stopwords."""
    return [palavra for palavra in palavras if palavra not in STOPWORDS]


def construir_vocabulario(criticas):
    """Reúne as palavras distintas na ordem em que aparecem."""
    vocabulario = []
    for critica in criticas:
        for palavra in critica:
            if palavra not in vocabulario:
                vocabulario.append(palavra)
    return vocabulario


def criar_bag_of_words(palavras, vocabulario):
    """Conta cada palavra usando o mesmo vocabulário para todas as críticas."""
    return {palavra: palavras.count(palavra) for palavra in vocabulario}


def classificar_critica(critica):
    """Retorna as contagens positivas e negativas e a classificação."""
    palavras = remover_stopwords(preparar_texto(critica))
    quantidade_positivas = sum(palavra in POSITIVAS for palavra in palavras)
    quantidade_negativas = sum(palavra in NEGATIVAS for palavra in palavras)

    if quantidade_positivas > quantidade_negativas:
        resultado = 'POSITIVO'
    elif quantidade_negativas > quantidade_positivas:
        resultado = 'NEGATIVO'
    else:
        resultado = 'NEUTRO'

    return quantidade_positivas, quantidade_negativas, resultado


def exibir_classificacao(critica):
    positivas, negativas, resultado = classificar_critica(critica)
    print(f'Crítica: {critica}')
    print(f'Palavras positivas: {positivas}')
    print(f'Palavras negativas: {negativas}')
    print(f'Resultado: {resultado}')


def demonstrar_etapas():
    critica = 'O filme foi maravilhoso e os atores excelentes!'

    print('ETAPA 1 — PREPARAÇÃO E TOKENIZAÇÃO')
    palavras = preparar_texto(critica)
    print(palavras)

    print('\nETAPA 2 — REMOÇÃO DE STOPWORDS')
    print(remover_stopwords(palavras))

    print('\nETAPA 3 — BAG OF WORDS')
    criticas_processadas = [
        ['filme', 'bom', 'excelente'],
        ['filme', 'ruim', 'chato'],
        ['filme', 'bom', 'divertido'],
    ]
    vocabulario = construir_vocabulario(criticas_processadas)
    print(f'Vocabulário: {vocabulario}')
    for numero, palavras_da_critica in enumerate(criticas_processadas, start=1):
        print(f'Crítica {numero}: {criar_bag_of_words(palavras_da_critica, vocabulario)}')

    print('\nETAPA 4 — EXEMPLO DE CLASSIFICAÇÃO')
    exibir_classificacao('O filme foi excelente e divertido')

    print('\nETAPA 5 — TESTES COM CINCO CRÍTICAS')
    criticas_teste = [
        'O filme foi maravilhoso e os atores excelentes',
        'O roteiro foi fraco e o filme muito chato',
        'O filme foi bom, mas o final foi decepcionante',
        'A história é interessante',
        'O filme é excelente, mas não é divertido',
    ]
    for critica_teste in criticas_teste:
        exibir_classificacao(critica_teste)
        print()


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    if len(sys.argv) > 1:
        exibir_classificacao(' '.join(sys.argv[1:]))
    else:
        demonstrar_etapas()
