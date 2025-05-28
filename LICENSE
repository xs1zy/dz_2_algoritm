import random
import math

class Percolation:
    def init(self, N):
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
        # очищаем предыдущие метки
        self.full = [[False] * self.N for _ in range(self.N)]
        # запускаем DFS от всех открытых ячеек верхнего ряда
        for j in range(self.N):
            if self.grid[0][j]:
                self._dfs(0, j)

    def _dfs(self, i, j):
        if i < 0 or i >= self.N or j < 0 or j >= self.N:
            return
        if not self.grid[i][j] or self.full[i][j]:
            return
        self.full[i][j] = True
        self._dfs(i - 1, j)  # вверх
        self._dfs(i + 1, j)  # вниз
        self._dfs(i, j - 1)  # влево
        self._dfs(i, j + 1)  # вправо


class PercolationStats:
    def init(self):
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
