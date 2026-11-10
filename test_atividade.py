"""Verificações dos resultados pedidos na atividade prática."""

import unittest

from main import (
    classificar_critica,
    construir_vocabulario,
    criar_bag_of_words,
    preparar_texto,
    remover_stopwords,
)


class TestarAtividade(unittest.TestCase):
    def test_tokenizacao_do_enunciado_e_pontuacao_extra(self):
        self.assertEqual(
            preparar_texto('O filme foi maravilhoso e os atores excelentes!'),
            ['o', 'filme', 'foi', 'maravilhoso', 'e', 'os', 'atores', 'excelentes'],
        )
        self.assertEqual(
            preparar_texto('Bom? Ruim, excelente.'),
            ['bom', 'ruim', 'excelente'],
        )

    def test_remocao_de_stopwords_do_enunciado(self):
        palavras = ['o', 'filme', 'foi', 'maravilhoso', 'e', 'os', 'atores', 'excelentes']
        self.assertEqual(
            remover_stopwords(palavras),
            ['filme', 'maravilhoso', 'atores', 'excelentes'],
        )
        self.assertIn('não', remover_stopwords(['o', 'filme', 'não', 'é', 'bom']))

    def test_bag_of_words_com_vocabulario_compartilhado(self):
        criticas = [
            ['filme', 'bom', 'excelente'],
            ['filme', 'ruim', 'chato'],
            ['filme', 'bom', 'divertido'],
        ]
        vocabulario = construir_vocabulario(criticas)
        self.assertEqual(
            vocabulario,
            ['filme', 'bom', 'excelente', 'ruim', 'chato', 'divertido'],
        )
        self.assertEqual(
            criar_bag_of_words(criticas[0], vocabulario),
            {'filme': 1, 'bom': 1, 'excelente': 1,
             'ruim': 0, 'chato': 0, 'divertido': 0},
        )
        self.assertEqual(
            criar_bag_of_words(['bom', 'bom'], vocabulario)['bom'], 2,
        )

    def test_classificacao_positiva_negativa_e_neutra(self):
        self.assertEqual(
            classificar_critica('O filme foi excelente e divertido'),
            (2, 0, 'POSITIVO'),
        )
        self.assertEqual(
            classificar_critica('O roteiro foi fraco e chato'),
            (0, 2, 'NEGATIVO'),
        )
        self.assertEqual(
            classificar_critica('O filme foi bom, mas decepcionante'),
            (1, 1, 'NEUTRO'),
        )

    def test_resultados_das_cinco_criticas_sugeridas(self):
        exemplos = [
            ('O filme foi maravilhoso e os atores excelentes', (1, 0, 'POSITIVO')),
            ('O roteiro foi fraco e o filme muito chato', (0, 2, 'NEGATIVO')),
            ('O filme foi bom, mas o final foi decepcionante', (1, 1, 'NEUTRO')),
            ('A história é interessante', (0, 0, 'NEUTRO')),
            ('O filme é excelente, mas não é divertido', (2, 0, 'POSITIVO')),
        ]
        for critica, esperado in exemplos:
            with self.subTest(critica=critica):
                self.assertEqual(classificar_critica(critica), esperado)


if __name__ == '__main__':
    unittest.main()
