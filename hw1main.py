# -*-coding: UTF-8 -*-
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用微軟雅黑字體


def build_word_vector(text):
    word2id = {w: i for i, w in enumerate(sorted(list(set(reduce(lambda a, b: a + b, text)))))}
    id2word = {x[1]: x[0] for x in word2id.items()}
    wvectors = np.zeros((len(word2id), len(word2id)))
    for sentence in text:
        for word1, word2 in zip(sentence[:1], sentence[1:]):
            id1, id2 = word2id[word1], word2id[word2]
            wvectors[id1, id2] += 1
            wvectors[id2, id1] += 1
    return wvectors, word2id, id2word


def cosine_sim(v1, v2):
    return np.dot(v1, v2) / (np.sqrt(np.sum(np.power(v1, 2))) * np.sqrt(np.sum(np.power(v1, 2))))


def visualize(wvectors, id2word):
    np.random.seed(10)
    fig = plt.figure()
    U, sigma, Vh = np.linalg.svd(wvectors)
    ax = fig.add_subplot(111)
    ax.axis([-1, 1, -1, 1])
    for i in id2word:
        ax.text(U[i, 0], U[i, 1], id2word[i], alpha=0.3, fontsize=20)
    plt.show()


if __name__ == '__main__':
    # ----------------------------------------------------------------------------------------------------------------------
    # 作業, 修改輸入資料
    # ----------------------------------------------------------------------------------------------------------------------
    text = [["一隻", "羊", "在玩", ],
            ["一隻", "狼", "突然竄出", ],
            ["一隻", "馬", "低頭一看"],
            ["發現是", "一隻", "狼", ],
            ["跟著", "一隻", "雞", ],
            ["驢", "停下", "腳步", ],
            ["豬", "經過", "這裏", ],
            ]

    wvectors, word2id, id2word = build_word_vector(text)

    print(word2id)

    print(id2word)

    # print(wvectors[word2id["一隻"]])
    #
    # print(cosine_sim(wvectors[word2id["dog"]], wvectors[word2id["cat"]]))
    #
    # print(cosine_sim(wvectors[word2id["dog"]], wvectors[word2id["bird"]]))

    visualize(wvectors, id2word)
