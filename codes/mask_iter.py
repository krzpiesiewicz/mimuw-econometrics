class MaskIter():
    def __init__(self, n, min_ones = 0, max_ones = None):
        self.n = n
        self.max_ones = max_ones if max_ones is not None else n
        self.min_ones = min_ones
        
    def __iter__(self):
        self.mask = [0 for _ in range(self.n)]
        self.visited = [0 for _ in range(self.n)]
        self.pos = 0
        self.ones = 0
        self.backward = False
        self.first = True
        return self
    
    def __next__(self):
        if self.first:
            self.first = False
            if self.min_ones == 0:
                return mask
        if self.pos == self.n:
            self.pos -= 1
            self.backward = True
        while self.pos >= 0 \
            and ((self.ones == self.max_ones and self.mask[self.pos] == 0) or self.mask[self.pos] == 1):
            if self.mask[self.pos] == 1:
                self.mask[self.pos] = 0
                self.ones -= 1
            self.pos -= 1
            self.backward = True
        if self.pos == -1:
            raise StopIteration
        while self.pos < self.n and self.mask[self.pos] == 0 and self.ones < self.max_ones:
            if self.backward:
                self.mask[self.pos] = 1
                self.ones += 1
                self.pos += 1
                self.backward = False
                if self.ones >= self.min_ones:
                    return self.mask
            else:
                self.pos += 1
                self.backward = False
        return self.__next__()
