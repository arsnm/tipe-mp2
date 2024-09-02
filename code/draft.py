import itertools

import numpy as np

DIM = 2
DIM_DELIM = {
    0: "",
    1: "$",
    2: "%",
    3: "#",
    4: "@A",
    5: "@B",
    6: "@C",
    7: "@D",
    8: "@E",
    9: "@F",
}


def ch2val(c):
    if c in ".b":
        return 0
    elif c == "o":
        return 255
    elif len(c) == 1:
        return ord(c) - ord("A") + 1
    else:
        return (ord(c[0]) - ord("p")) * 24 + (ord(c[1]) - ord("A") + 25)


def val2ch(v):
    if v == 0:
        return " ."
    elif v < 25:
        return " " + chr(ord("A") + v - 1)
    else:
        return chr(ord("p") + (v - 25) // 24) + chr(ord("A") + (v - 25) % 24)


def _recur_join_st(dim, lists, row_func):
    if dim < DIM - 1:
        return DIM_DELIM[DIM - 1 - dim].join(
            _recur_join_st(dim + 1, e, row_func) for e in lists
        )
    else:
        return DIM_DELIM[DIM - 1 - dim].join(row_func(lists))


def _recur_drill_list(dim, lists, row_func):
    if dim < DIM - 1:
        return [_recur_drill_list(dim + 1, e, row_func) for e in lists]
    else:
        return row_func(lists)


def _append_stack(list1, list2, count, is_repeat=False):
    list1.append(list2)
    if count != "":
        repeated = list2 if is_repeat else []
        list1.extend([repeated] * (int(count) - 1))


def _recur_get_max_lens(dim, list1, max_lens):
    max_lens[dim] = max(max_lens[dim], len(list1))
    if dim < DIM - 1:
        for list2 in list1:
            _recur_get_max_lens(dim + 1, list2, max_lens)


def _recur_cubify(dim, list1, max_lens):
    more = max_lens[dim] - len(list1)
    if dim < DIM - 1:
        list1.extend([[]] * more)
        for list2 in list1:
            _recur_cubify(dim + 1, list2, max_lens)
    else:
        list1.extend([0] * more)


def cells2rle(A, is_shorten=True):
    values = np.rint(A * 255).astype(int).tolist()  # [[255 255] [255 0]]
    if is_shorten:
        rle_groups = _recur_drill_list(
            0,
            values,
            lambda row: [
                (len(list(g)), val2ch(v).strip()) for v, g in itertools.groupby(row)
            ],
        )
        st = _recur_join_st(
            0, rle_groups, lambda row: [(str(n) if n > 1 else "") + c for n, c in row]
        )  # "2 yO $ 1 yO"
    else:
        st = _recur_join_st(0, values, lambda row: [val2ch(v) for v in row])
    return st + "!"


def rle2cells(st):
    stacks = [[] for dim in range(DIM)]
    last, count = "", ""
    delims = list(DIM_DELIM.values())
    st = st.rstrip("!") + DIM_DELIM[DIM - 1]
    for ch in st:
        if ch.isdigit():
            count += ch
        elif ch in "pqrstuvwxy@":
            last = ch
        else:
            if last + ch not in delims:
                _append_stack(stacks[0], ch2val(last + ch) / 255, count, is_repeat=True)
            else:
                dim = delims.index(last + ch)
                for d in range(dim):
                    _append_stack(stacks[d + 1], stacks[d], count, is_repeat=False)
                    stacks[d] = []
                # print("{0}[{1}] {2}".format(last+ch, count, [np.asarray(s).shape for s in stacks]))
            last, count = "", ""
    A = stacks[DIM - 1]
    max_lens = [0 for dim in range(DIM)]
    _recur_get_max_lens(0, A, max_lens)
    _recur_cubify(0, A, max_lens)
    return np.asarray(A)


a = rle2cells(
    "22.GpKqCpKJ$20.JrXuFuUuKtDqWpA$19.pXwQxHxCxJxHvWtTrQpUO$18.pPyOxHvOwXyLyOyLwIuPtBrQqHpDG$18.yOvTrQtGwN4yOyDwQvGtTsKrDpUO$17.yOuP2.sCxE7yOxWwQvBtDqTO$16.yOwD3.tD6yOyL4yOwXtOqHE$15.2yO4.yL4yOwXvTwDxJ4yOwNsRpN$14.wLyO4.uX4yOvEqWqMrSuUxW4yOvMsHpP$13.pIyL4.qO4yOxU3.GsMwV4yOxHuPsMqEG$12.pDyL5.2yOyL2yO5.qWvOyG3yOwLuSuAtBrGpIE$12.yO5.5yOsK5.pXuNwXxJwXvJ2tDtVuPuAsRrGpUL$11.yO5.6yO6.qJtQtVsWrGpDpIrGtVwDwSwAuStBqWQ$10.yO5.5yOyL5.BqTuFsKT4.pUvE3yOyBvWtGqME$9.yOpK4.yI4yOvW5.sCvTyOyBV5.pSwV5yOvErXpD$8.pSyO5.4yOsK5.uAyGyLyOwDO5.qTwX5yOwVtDqEB$8.yO5.4yOpN6.2yOxCyOuK6.qOwLyL5yOuPrIT$7.yO5.4yOqJ6.yD4yO7.qOvBwNxExRyG2yOwSsUqCB$6.yLE4.uS3yOwA6.tQ4yO7.BrDrQrNrSsCuSxM2yOvBrNpF$5.qWvT5.4yO6.pP3yOyI6.qEsRvJrV5.vEyIyOxRtTrDpIB$5.yO5.yI3yOpI5.rGyOyI2yO6.rIxJ2yOrI5.tJyG2yOwVuFsCpS$4.yOpU4.4yOsR4.EvWyDuUvOyOqO6.yOyGxMyOqR5.tT4yOyGvJrXO$3.yOxP3.O4yOxE5.yOxRqJqCuNyO6.yDyOvRyDxHpF4.GyB5yOyBtTV$2.wLyO3.qC5yO5.yDyO2.VxCsM5.tL4yOuF5.8yOuUQ$2.yO3.pU2yOyL2yO5.pUyOqW2.uKyLJ5.4yOuP4.pX5yOxPwNxCyOuUB$.yOpS2.B5yOpU5.xJyOpPQvWyOqC4.rL4yOrD4.pK5yOwAsFqWuPyOtJ$tQwDqHJQ5yOwF5.tGyOvErSyByL4.QvOxRwAxWyOrX5.2yOxM2yOxMG2.vByOpD$uNuKtDsP6yO5.pI2yOtT2yO5.yOwAqTsFxWyO5.tB5yO4.yOwQ$qTuIwF6yO6.yOyGrXyDyOpS4.yIyO2.rNyOE5.xH4yO4.vEyO$pItO2wFwIxE2yO6.2yOGsMyOxH4.qJyOJ2.wAvW5.rL4yOpU3.pFyO$.rNrQrLsHvOyOwN5.xUyOE.wQyOO4.yIwSG.tByOpX5.xW3yOtV4.yO$4.pSuNyIqT4.sUyDuNqRxWyOsU4.yByOsPpXvJyOtO5.4yOyL4.yOrL$4.LtOsUrVO3.uXyDtLyBxEsW3.B2yOwFtJ2yOsU4.L5yO4.uSyO$6.qJsFrLqEpIuAyLsRtGsHpS4.2yOvGsC2yOrX4.B5yO5.yO$7.qOrSsHvRyOxCpN6.tLyOtV.yGyOqE5.vO4yO5.yO$8.OqJvWxMtTO6.xEyO.sPyOtV5.J4yO5.wXqH$9.qJrXrNrVpU5.qEwXsRsC2yO6.4yO5.pXyO$11.JpXrG5.qHvTwS2yO6.4yO6.yO$13.qWqE4.qJsUyGxP6.4yOpI5.yO$13.QqRqWqJpSqWrSxJuX6.4yOuN5.yO$15.pDqEsPvWyOvT6.xR4yO5.yO$17.rIwAyLrQ5.uU2yOyByOsH4.yOqE$16.GqOtLuKrV5.2yOyD2yOB3.uXvB$17.GqCpNsHpX3.5yOrI3.vGwF$20.rGsK2rN5yOxJ3.tTyO$20.JqWtB3yI3yO3.pKyO$22.tTuAuNwAyGyOtJ3.yO$22.VpAqJtVxMyOrN2OyO$24.OsUwQvEtLsWyO$24.BsKtLsWtGxHtV$25.rNqMqCtLqR$25.B!"
)
arr_list = a.tolist()

with open("species", "a") as f:
    print(arr_list)
    f.write(f"quadrium = np.array({arr_list})")
    f.close()
