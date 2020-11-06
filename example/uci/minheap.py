class MinHeap(object):
    """a heap designed for top K problem.
    Attributes:
        capacity: the max size of the heap as well as the number K in top K problem.
        size: the number of the elements in the heap.
        heap: the heap itself, the elements of which are lists consists of items like [key, val],
              If keys are ignored and only vals are considered, the heap is a min heap.
    """
    def __init__(self, capacity):
        """Init the heap."""
        self.capacity = capacity
        self.size = 0
        self.heap = [[None, -float('inf')]] * capacity

    def min_heapify(self, index, key, val):
        """Set the index'th element with [key, val],
        then modify the heap to make it still a min heap.
        """
        while index > 0:
            p = (index-1)//2
            if self.heap[p][1] > val:
                self.heap[index] = self.heap[p].copy()
                index = p
            else:
                break
        self.heap[index] = [key, val]

    def push(self, key, val):
        """Push a new element [key, val] into the heap.
        If the heap is full, the original root will be deleted.
        """
        if self.size < self.capacity:
            self.min_heapify(self.size, key, val)
            self.size += 1
        elif self.minimum() < val:
            self.pop()
            self.push(key, val)

    def pop(self):
        """Delete the root of the heap,
        then modify the heap to make it still a min heap.
        Returns:
             the element at root, whose val is minimum of the heap.
        """
        ans = self.heap[0]
        if self.size:
            self.size -= 1
            index = 0
            while 2*index+1 < self.size:
                a, b = 2*index+1, 2*index+2
                if b < self.size and self.heap[b][1] < self.heap[a][1]:
                    a = b
                if self.heap[a][1] > self.heap[self.size][1]:
                    break
                self.heap[index] = self.heap[a].copy()
                index = a
            self.heap[index] = self.heap[self.size].copy()
        return ans

    def items(self):
        """Return the items in the heap."""
        return self.heap[:self.size]

    def minimum(self):
        """Return the minimum of the heap."""
        return self.heap[0][1] if not self.isempty() else float('inf')

    def isempty(self):
        """Return whether the heap is empty."""
        return self.size == 0

    def reset(self):
        """Reset the heap."""
        self.size = 0
