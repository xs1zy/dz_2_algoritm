import random
import math

import random
import math

class Percolation:
    def __init__(self, N):
        if N <= 0:
            raise ValueError("N must be > 0")
        self.N = N
        self.grid = [[False] * N for _ in range(N)]  # все ячейки закрыты
        self.full = [[False] * N for _ in range(N)]  # временно для DFS
        self.open_sites = 0

    def _validate(self, i, j):
        if not (1 <= i <= self.N and 1 <= j <= self.N):
            raise IndexError("Index out of bounds")

    def open(self, i, j):
        self._validate(i, j)
        if not self.grid[i - 1][j - 1]:
            self.grid[i - 1][j - 1] = True
            self.open_sites += 1

    def isOpen(self, i, j):
        self._validate(i, j)
        return self.grid[i - 1][j - 1]

    def isFull(self, i, j):
        self._validate(i, j)
        self._mark_full_sites()
        return self.full[i - 1][j - 1]

    def percolates(self):
        self._mark_full_sites()
        for j in range(self.N):
            if self.full[self.N - 1][j]:
                return True
        return False

    def numberOfOpenSites(self):
        return self.open_sites

    def _mark_full_sites(self):
        self.full = [[False] * self.N for _ in range(self.N)]
        for j in range(self.N):
            if self.grid[0][j]:
                self._dfs(0, j)

    def _dfs(self, i, j):
        if i < 0 or i >= self.N or j < 0 or j >= self.N:
            return
        if not self.grid[i][j] or self.full[i][j]:
            return
        self.full[i][j] = True
        self._dfs(i - 1, j)
        self._dfs(i + 1, j)
        self._dfs(i, j - 1)
        self._dfs(i, j + 1)


class PercolationStats:
    def __init__(self):
        self.results = []

    def doExperiment(self, N, T):
        if N <= 0 or T <= 0:
            raise ValueError("N and T must be > 0")
        self.results = []

        for _ in range(T):
            perc = Percolation(N)
            while not perc.percolates():
                i, j = random.randint(1, N), random.randint(1, N)
                perc.open(i, j)
            opened_ratio = perc.numberOfOpenSites() / (N * N)
            self.results.append(opened_ratio)

        self._calculate_stats()

    def _calculate_stats(self):
        T = len(self.results)
        self._mean = sum(self.results) / T
        self._stddev = math.sqrt(sum((x - self._mean) ** 2 for x in self.results) / (T - 1)) if T > 1 else float('nan')
        margin = 1.96 * self._stddev / math.sqrt(T)
        self._confidence = (self._mean - margin, self._mean + margin)

    def mean(self):
        return self._mean

    def stddev(self):
        return self._stddev

    def confidence(self):
        return self._confidence


# Пример использования:
if __name__ == "__main__":
    N = 20  # размер решетки
    T = 30  # количество экспериментов

    ps = PercolationStats()
    ps.doExperiment(N, T)
    # Пример использования:
    print(f"Решетка: {N}x{N}, Экспериментов: {T}")
    print(f"mean                    = {ps.mean()}")
    print(f"stddev                  = {ps.stddev()}")
    conf = ps.confidence()
    print(f"95% confidence interval = ({conf[0]}, {conf[1]})")
   # Код для анализа времени выполнения

class QuickFindUF:
    def __init__(self, n):
        self.id = list(range(n))

    def find(self, p):
        return self.id[p]

    def connected(self, p, q):
        return self.find(p) == self.find(q)

    def union(self, p, q):
        pid = self.find(p)
        qid = self.find(q)
        if pid == qid:
            return
        for i in range(len(self.id)):
            if self.id[i] == pid:
                self.id[i] = qid

class PercolationQuickFind:
    def __init__(self, N):
        if N <= 0:
            raise ValueError("N must be > 0")
        self.N = N
        self.grid = [[False]*N for _ in range(N)]
        self.uf = QuickFindUF(N * N + 2)
        self.top = N * N
        self.bottom = N * N + 1
        for j in range(N):
            self.uf.union(self.top, self._index(0, j))
            self.uf.union(self.bottom, self._index(N-1, j))

    def _index(self, i, j):
        return i * self.N + j

    def open(self, i, j):
        i -= 1
        j -= 1
        if not (0 <= i < self.N and 0 <= j < self.N):
            raise IndexError("Index out of bounds")
        if self.grid[i][j]:
            return
        self.grid[i][j] = True
        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.N and 0 <= nj < self.N and self.grid[ni][nj]:
                self.uf.union(self._index(i, j), self._index(ni, nj))

    def isOpen(self, i, j):
        return self.grid[i-1][j-1]

    def isFull(self, i, j):
        return self.uf.connected(self.top, self._index(i-1, j-1))

    def percolates(self):
        return self.uf.connected(self.top, self.bottom)


class PercolationStatsQF:
    def __init__(self):
        pass

    def doExperiment(self, N, T):
        times = []
        results = []
        start = time.time()
        for _ in range(T):
            perc = PercolationQuickFind(N)
            opened = 0
            while not perc.percolates():
                i, j = random.randint(1, N), random.randint(1, N)
                if not perc.isOpen(i, j):
                    perc.open(i, j)
                    opened += 1
            results.append(opened / (N * N))
        end = time.time()

        mean = sum(results) / T
        stddev = math.sqrt(sum((x - mean) ** 2 for x in results) / (T - 1))
        conf = 1.96 * stddev / math.sqrt(T)
        return {
            "mean": mean,
            "stddev": stddev,
            "confidence": (mean - conf, mean + conf),
            "time_sec": end - start
        }
    # Ответы на вопросы по QuickFind:
    # 1.  Удвоение N
    # Если N удваивается, размер сетки увеличивается в 4 раза ((2N)² = 4N²), а каждая операция union становится в 4 раза дороже.
    # Приблизительно время выполнения увеличивается в 16 раз
    # 2. Удвоение T
    # Число экспериментов увеличивается в 2 раза — следовательно, общее время выполнения тоже увеличится примерно в 2 раза
    # 3. Приблизительная формула времени выполнения
    # Time(N, T) ≈ k × T × N⁴
    class WeightedQuickUnionUF:
        def __init__(self, n):
            self.parent = list(range(n))
            self.size = [1] * n

        def find(self, p):
            root = p
            while root != self.parent[root]:
                root = self.parent[root]
            # Path compression
            while p != root:
                next_p = self.parent[p]
                self.parent[p] = root
                p = next_p
            return root

        def connected(self, p, q):
            return self.find(p) == self.find(q)

        def union(self, p, q):
            rootP = self.find(p)
            rootQ = self.find(q)
            if rootP == rootQ:
                return
            if self.size[rootP] < self.size[rootQ]:
                self.parent[rootP] = rootQ
                self.size[rootQ] += self.size[rootP]
            else:
                self.parent[rootQ] = rootP
                self.size[rootP] += self.size[rootQ]

    class PercolationQuickUnion:
        def __init__(self, N):
            if N <= 0:
                raise ValueError("N must be > 0")
            self.N = N
            self.grid = [[False] * N for _ in range(N)]
            self.uf = WeightedQuickUnionUF(N * N + 2)
            self.top = N * N
            self.bottom = N * N + 1
            for j in range(N):
                self.uf.union(self.top, self._index(0, j))
                self.uf.union(self.bottom, self._index(N - 1, j))

        def _index(self, i, j):
            return i * self.N + j

        def open(self, i, j):
            i -= 1
            j -= 1
            if not (0 <= i < self.N and 0 <= j < self.N):
                raise IndexError("Index out of bounds")
            if self.grid[i][j]:
                return
            self.grid[i][j] = True
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < self.N and 0 <= nj < self.N and self.grid[ni][nj]:
                    self.uf.union(self._index(i, j), self._index(ni, nj))

        def isOpen(self, i, j):
            return self.grid[i - 1][j - 1]

        def isFull(self, i, j):
            return self.uf.connected(self.top, self._index(i - 1, j - 1))

        def percolates(self):
            return self.uf.connected(self.top, self.bottom)

    class PercolationStatsQU:
        def __init__(self):
            pass

        def doExperiment(self, N, T):
            results = []
            start = time.time()
            for _ in range(T):
                perc = PercolationQuickUnion(N)
                opened = 0
                while not perc.percolates():
                    i, j = random.randint(1, N), random.randint(1, N)
                    if not perc.isOpen(i, j):
                        perc.open(i, j)
                        opened += 1
                results.append(opened / (N * N))
            end = time.time()

            mean = sum(results) / T
            stddev = math.sqrt(sum((x - mean) ** 2 for x in results) / (T - 1))
            conf = 1.96 * stddev / math.sqrt(T)
            return {
                "mean": mean,
                "stddev": stddev,
                "confidence": (mean - conf, mean + conf),
                "time_sec": end - start}
        # Ответы на вопросы по QuickUnion:
        # 1.  Удвоение N
        # При удвоении N, время увеличивается примерно в 4 × log(2N)/log(N) раза.
        # 2. Удвоение T
        # При увеличении числа экспериментов T в 2 раза общее время выполнения увеличится примерно в 2 раза, так как каждый эксперимент независим.
        # 3. Приблизительная формула времени выполнения
        # Time(N, T) ≈ k × T × N² × log N
