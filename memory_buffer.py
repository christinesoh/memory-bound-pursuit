from random import sample

class Node:
    def __init__(self, word):
        self.word = word
        self.next = None
        self.prev = None

    def equals(self, other):
        return self.word == other.word

    def remove(self):
        p = self.prev
        n = self.next
        p.next = n
        n.prev = p

    def get_word(self):
        return self.word

    def get_next(self):
        return self.next

    def get_prev(self):
        return self.prev

    def set_next(self, other):
        self.next = other

    def set_prev(self, other):
        self.prev = other


class MemoryBuffer:
    def __init__(self, size):
        self.cache_limit = size
        self.cache = {}
        self.head = Node("")
        self.tail = Node("")
        self.head.set_next(self.tail)
        self.tail.set_prev(self.head)

    def contains(self, word):
        for node in self.cache:
            if node == word:
                return True
        return False

    def put(self, word):
        node = Node(word)
        self.cache[word] = node
        p = self.tail.get_prev()
        p.set_next(node)
        self.tail.set_prev(node)
        node.set_prev(p)
        node.set_next(self.tail)

    def reshuffle(self, word):
        assert self.contains(word)
        current = self.cache[word]
        current.remove()
        self.put(word)

    def get(self):
        if len(self.cache) == 0:
            raise IndexError("Nothing to remove")
        word = sample(self.cache, 1)[0]
        self.cache.remove(word)
        return word

    def full(self):
        return len(self.cache) >= self.cache_limit

    def __str__(self):
        string = ""
        a = self.head.get_next()
        while not a.equals(self.tail):
            string = string + a.get_word() + ", "
            a = a.get_next()
        return string
