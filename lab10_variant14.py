"""
Лабораторная работа № 10. Вариант 14.
Алгоритм Дейкстры (две реализации), восстановление пути и расчёт критического пути
проекта из 13 работ. Графы: разреженный (m ≈ 4n) и плотный (20 % дуг),
n = 300, 600, 1200, веса дуг 1…80. Дополнительное задание К7: диаграмма Ганта.
"""
import gc
import heapq
import json
import random
import statistics
import time
from collections import deque

INF = float("inf")
MAX_W = 80


# ------------------------------------------------------- алгоритм Дейкстры
def dijkstra_array(adj, s):
    """Дейкстра с линейным поиском минимума. Трудоёмкость Θ(n²)."""
    n = len(adj)
    dist, parent, used = [INF] * n, [-1] * n, [False] * n
    dist[s] = 0
    for _ in range(n):
        v, best = -1, INF
        for i in range(n):                       # линейный поиск минимума
            if not used[i] and dist[i] < best:
                v, best = i, dist[i]
        if v == -1:
            break
        used[v] = True
        for u, w in adj[v]:                      # релаксация исходящих дуг
            if dist[v] + w < dist[u]:
                dist[u] = dist[v] + w
                parent[u] = v
    return dist, parent


def dijkstra_heap(adj, s):
    """Дейкстра с приоритетной очередью (heapq). Трудоёмкость O(m log n)."""
    n = len(adj)
    dist, parent = [INF] * n, [-1] * n
    dist[s] = 0
    heap = [(0, s)]
    while heap:
        d, v = heapq.heappop(heap)
        if d > dist[v]:                          # устаревшая запись
            continue
        for u, w in adj[v]:
            if d + w < dist[u]:
                dist[u] = d + w
                parent[u] = v
                heapq.heappush(heap, (dist[u], u))
    return dist, parent


def dijkstra_trace(adj, s):
    """Дейкстра с протоколом шагов (для ручной проверки на малом графе)."""
    n = len(adj)
    dist, parent, used = [INF] * n, [-1] * n, [False] * n
    dist[s] = 0
    steps = []
    for _ in range(n):
        v, best = -1, INF
        for i in range(n):
            if not used[i] and dist[i] < best:
                v, best = i, dist[i]
        if v == -1:
            break
        used[v] = True
        updates = []
        for u, w in adj[v]:
            if dist[v] + w < dist[u]:
                dist[u] = dist[v] + w
                parent[u] = v
                updates.append((u, dist[u]))
        steps.append((v, best, updates, list(dist)))
    return steps, dist, parent


def restore_path(parent, target):
    """Восстановление пути по массиву предков."""
    path = []
    while target != -1:
        path.append(target)
        target = parent[target]
    return path[::-1]


def bellman_ford(adj, s):
    """Эталон для проверки (Беллман — Форд, Θ(nm))."""
    n = len(adj)
    dist = [INF] * n
    dist[s] = 0
    for _ in range(n - 1):
        changed = False
        for v in range(n):
            if dist[v] < INF:
                for u, w in adj[v]:
                    if dist[v] + w < dist[u]:
                        dist[u] = dist[v] + w
                        changed = True
        if not changed:
            break
    return dist


def generate_graph(n, m, rng, max_weight=MAX_W):
    """Случайный ориентированный взвешенный граф: n вершин, m различных дуг.
    Сначала строится остов (каждая вершина достижима из вершины 0), затем
    добавляются случайные дуги до m."""
    arcs = set()
    for i in range(1, n):
        arcs.add((rng.randrange(i), i))
    while len(arcs) < m:
        u, v = rng.randrange(n), rng.randrange(n)
        if u != v:
            arcs.add((u, v))
    adj = [[] for _ in range(n)]
    for u, v in sorted(arcs):
        adj[u].append((v, rng.randint(1, max_weight)))
    return adj


# ------------------------------------------------------- критический путь
def topological_order(tasks):
    """Топологическая сортировка (алгоритм Кана). tasks: код → (длительность, предшественники).
    При наличии цикла (некорректный сетевой график) выбрасывает ValueError."""
    indeg = {k: len(v[1]) for k, v in tasks.items()}
    succ = {k: [] for k in tasks}
    for k, (_, preds) in tasks.items():
        for p in preds:
            succ[p].append(k)
    queue = deque(sorted(k for k, d in indeg.items() if d == 0))
    order = []
    while queue:
        v = queue.popleft()
        order.append(v)
        for u in succ[v]:
            indeg[u] -= 1
            if indeg[u] == 0:
                queue.append(u)
    if len(order) != len(tasks):
        raise ValueError("Сетевой график содержит цикл")
    return order, succ


def critical_path(tasks):
    """Расчёт ES, EF, LS, LF, резервов и критических работ."""
    order, succ = topological_order(tasks)
    ES, EF = {}, {}
    for v in order:                              # прямой проход
        duration, preds = tasks[v]
        ES[v] = max((EF[p] for p in preds), default=0)
        EF[v] = ES[v] + duration
    T = max(EF.values())
    LS, LF = {}, {}
    for v in reversed(order):                    # обратный проход
        duration = tasks[v][0]
        LF[v] = min((LS[u] for u in succ[v]), default=T)
        LS[v] = LF[v] - duration
    reserve = {v: LS[v] - ES[v] for v in tasks}
    critical = [v for v in order if reserve[v] == 0]
    return T, ES, EF, LS, LF, reserve, critical, order


def all_paths_longest(tasks):
    """Проверка полным перебором: все пути от начальных работ до конечных,
    возвращает (длина самого длинного пути, список самых длинных путей)."""
    succ = {k: [] for k in tasks}
    for k, (_, preds) in tasks.items():
        for p in preds:
            succ[p].append(k)
    best, best_paths = -1, []

    def walk(v, path, length):
        nonlocal best, best_paths
        path = path + [v]
        length += tasks[v][0]
        if not succ[v]:
            if length > best:
                best, best_paths = length, [path]
            elif length == best:
                best_paths.append(path)
            return
        for u in succ[v]:
            walk(u, path, length)

    for k, (_, preds) in tasks.items():
        if not preds:
            walk(k, [], 0)
    return best, best_paths


# Сетевой график проекта (вариант 14: 13 работ; базовая таблица + работы K, L, M)
NAMES = {"A": "Сбор требований", "B": "Анализ бизнес-процессов", "C": "Проектирование базы данных",
         "D": "Проектирование интерфейса", "E": "Разработка серверной части",
         "F": "Разработка клиентской части", "G": "Интеграция модулей",
         "H": "Подготовка документации", "I": "Тестирование", "J": "Обучение пользователей",
         "K": "Развёртывание инфраструктуры", "L": "Миграция данных", "M": "Приёмка заказчиком"}
TASKS = {"A": (5, []), "B": (7, ["A"]), "C": (4, ["B"]), "D": (6, ["B"]), "E": (10, ["C"]),
         "F": (9, ["D"]), "G": (3, ["E", "F"]), "H": (4, ["C", "D"]), "I": (6, ["G"]),
         "J": (2, ["H", "I"]), "K": (3, ["B"]), "L": (5, ["C", "K"]), "M": (3, ["J", "L"])}


# ---------------------------------------------------------------- проверки
def self_test():
    """Сверка обеих реализаций Дейкстры с Беллманом — Фордом, проверка путей."""
    rng = random.Random(10)
    for _ in range(200):
        n = rng.randint(5, 60)
        m = rng.randint(n - 1, min(n * (n - 1), 5 * n))
        adj = generate_graph(n, m, rng)
        d1, p1 = dijkstra_array(adj, 0)
        d2, p2 = dijkstra_heap(adj, 0)
        d3 = bellman_ford(adj, 0)
        assert d1 == d2 == d3
        weights = {(v, u): w for v in range(n) for u, w in adj[v]}
        for t in range(n):                       # вес восстановленного пути равен расстоянию
            path = restore_path(p2, t)
            assert path[0] == 0 and sum(weights[(path[i], path[i + 1])] for i in range(len(path) - 1)) == d2[t]
    return 200


def timed(func, *args, repeats=5):
    times = []
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        func(*args)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


# ---------------------------------------------------------------- графики
def make_plots(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, kind, title in ((axes[0], "sparse", "Разреженный граф (m ≈ 4n)"),
                            (axes[1], "dense", "Плотный граф (20 % дуг)")):
        rows = [r for r in res["dijkstra"] if r["kind"] == kind]
        ns = [r["n"] for r in rows]
        ax.loglog(ns, [r["t_array"] for r in rows], "o-", label="линейный поиск минимума")
        ax.loglog(ns, [r["t_heap"] for r in rows], "s-", label="приоритетная очередь")
        ax.set_xlabel("число вершин n"); ax.set_ylabel("время, с"); ax.set_title(title)
        ax.grid(True, which="both", alpha=.3); ax.legend()
    plt.tight_layout(); plt.savefig("img/lr10_dijkstra.png", dpi=170); plt.close()

    cp = res["cpm"]
    order = sorted(cp["ES"], key=lambda k: (cp["ES"][k], k))
    plt.figure(figsize=(9, 5))
    for i, k in enumerate(order):
        crit = cp["reserve"][k] == 0
        plt.barh(i, cp["EF"][k] - cp["ES"][k], left=cp["ES"][k], color="#d62728" if crit else "#1f77b4",
                 edgecolor="k")
        if not crit:                              # полный резерв времени — светлая полоса
            plt.barh(i, cp["reserve"][k], left=cp["EF"][k], color="#c7dcef", edgecolor="#1f77b4", hatch="//")
        plt.text(cp["ES"][k] + 0.3, i, k, va="center", color="white", fontweight="bold")
    plt.yticks(range(len(order)), [f"{k}: {NAMES[k]}" for k in order], fontsize=8)
    plt.gca().invert_yaxis()
    plt.axvline(cp["T"], color="gray", ls="--")
    from matplotlib.patches import Patch
    plt.legend(handles=[Patch(color="#d62728", label="критические работы"),
                        Patch(color="#1f77b4", label="некритические работы"),
                        Patch(facecolor="#c7dcef", edgecolor="#1f77b4", hatch="//", label="полный резерв")],
               loc="upper right", fontsize=8)
    plt.xlabel("день проекта"); plt.title(f"Диаграмма Ганта проекта (T = {cp['T']} дн.)")
    plt.grid(True, axis="x", alpha=.3); plt.tight_layout(); plt.savefig("img/lr10_gantt.png", dpi=170); plt.close()


# ------------------------------------------------------------------- main
if __name__ == "__main__":
    res = {"selftest": self_test()}
    print("Самопроверка: графов", res["selftest"], "— Дейкстра (оба варианта) = Беллман — Форд")

    # --- 1. Малый тестовый граф (вершины нумеруются с 1)
    small_arcs = [(1, 2, 7), (1, 3, 9), (1, 6, 14), (2, 3, 10), (2, 4, 15), (3, 4, 11),
                  (3, 6, 2), (4, 5, 6), (6, 5, 9), (7, 6, 3)]
    adj = [[] for _ in range(7)]
    for u, v, w in small_arcs:
        adj[u - 1].append((v - 1, w))
    steps, dist, parent = dijkstra_trace(adj, 0)
    res["small"] = {"arcs": small_arcs,
                    "steps": [{"v": v + 1, "d": d, "upd": [(u + 1, x) for u, x in upd],
                               "dist": [None if x == INF else x for x in snap]}
                              for v, d, upd, snap in steps],
                    "dist": [None if x == INF else x for x in dist],
                    "path5": [v + 1 for v in restore_path(parent, 4)],
                    "path4": [v + 1 for v in restore_path(parent, 3)]}
    for st in res["small"]["steps"]:
        print(st)
    print(res["small"]["dist"], res["small"]["path5"])

    # --- 2. Сравнение реализаций: разреженный и плотный графы
    rng = random.Random(14)
    res["dijkstra"] = []
    print(f"{'n':>6}{'тип':>10}{'m':>9}{'массив, с':>12}{'куча, с':>10}{'отношение':>11}")
    for n in (300, 600, 1200):
        for kind, m in (("sparse", 4 * n), ("dense", int(0.2 * n * (n - 1)))):
            adj = generate_graph(n, m, rng)
            m_real = sum(len(a) for a in adj)
            d1, _ = dijkstra_array(adj, 0)
            d2, _ = dijkstra_heap(adj, 0)
            assert d1 == d2 and INF not in d1
            t_a, t_h = timed(dijkstra_array, adj, 0), timed(dijkstra_heap, adj, 0)
            res["dijkstra"].append({"n": n, "kind": kind, "m": m_real, "t_array": t_a, "t_heap": t_h,
                                    "ecc": max(d1)})
            print(f"{n:>6}{kind:>10}{m_real:>9}{t_a:>12.4f}{t_h:>10.4f}{t_a / t_h:>11.1f}", flush=True)

    # --- пример восстановления пути (n = 300)
    adj = generate_graph(300, 1200, random.Random(7))
    dist, parent = dijkstra_heap(adj, 0)
    target = max(range(300), key=lambda v: dist[v])
    path = restore_path(parent, target)
    res["path_example"] = {"target": target, "dist": dist[target], "path": path}
    print("Самая удалённая вершина", target, dist[target], path)

    # --- 3. Критический путь проекта (13 работ)
    T, ES, EF, LS, LF, R, crit, order = critical_path(TASKS)
    best, best_paths = all_paths_longest(TASKS)
    assert best == T, (best, T)
    assert sorted(crit) == sorted({v for p in best_paths for v in p}), "критические работы расходятся с перебором"
    res["cpm"] = {"T": T, "ES": ES, "EF": EF, "LS": LS, "LF": LF, "reserve": R, "critical": crit,
                  "order": order, "paths": best_paths, "tasks": {k: [v[0], v[1]] for k, v in TASKS.items()}}
    print(f"T = {T}", crit, best_paths)
    for k in sorted(TASKS):
        print(k, TASKS[k][0], ES[k], EF[k], LS[k], LF[k], R[k])

    # проверка обнаружения цикла в сетевом графике
    bad = dict(TASKS)
    bad["B"] = (7, ["A", "I"])
    try:
        topological_order(bad)
        res["cycle_check"] = "не обнаружен"
    except ValueError as e:
        res["cycle_check"] = str(e)
    print(res["cycle_check"])

    with open("res/lr10.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)
    make_plots(res)
