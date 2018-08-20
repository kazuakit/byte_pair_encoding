# coding: utf-8

from collections import defaultdict
import re

import tiny_functions as t


def init_text(text):
    return "\t".join(list(text))


class Encoder:
    def __init__(self):
        self.bigram = defaultdict(lambda: 0)
        self.vocab = None

    def train(self, text, n_vocab=-1):
        new_text = init_text(text)
        max_key, max_val = self.train_bigram(new_text)

        if n_vocab > 0:
            n_tmp_vocab = len(set(new_text.split("\t")))

        while True:
            if n_vocab > 0 and n_tmp_vocab <= n_vocab:
                break

            if max_val < 2:
                break

            new_text = self.train_text(new_text, max_key)
            max_key, max_val = self.train_bigram(new_text)
            if n_vocab > 0:
                n_tmp_vocab = len(set(new_text.split("\t")))

        self.vocab = set(new_text.split("\t"))

    def train_bigram(self, text):
        gs = t.ngram(text.split("\t"), 2)

        self.bigram = defaultdict(lambda: 0)
        max_val = 0
        for g in gs:
            k = tuple(g)
            self.bigram[k] += 1
            n = self.bigram[k]

            if n > max_val:
                max_key = k
                max_val = n

        return max_key, max_val

    def train_text(self, text, max_key):
        w1, w2 = max_key
        pattern = w1 + "\t" + w2 + r"(\t|$)"
        return re.sub(pattern, w1 + w2 + r"\1", text)

    def encode(self, text):
        n = len(text)

        i = 0
        prev_i = -1
        found = False
        ws = []
        while i < n:
            j = i + 1
            while j <= min(i + 50, n):
                w = text[i:j]
                if w in self.vocab:
                    found = True
                    break
                else:
                    j += 1

            if found:
                if prev_i > 0:
                    ws.append(text[prev_i:i])
                    prev_i = -1
                ws.append(text[i:j])
                i = j
                found = False
            else:
                if prev_i < 0:
                    prev_i = i
                i += 1

        return ws
