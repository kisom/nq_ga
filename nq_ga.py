#!/usr/bin/env python3
"""Solving the n-Queens problem using genetic algorithms."""

import datetime
import math
import random
import sys
import time


class Queen:
    def __init__(self, row, col):
        self.row = row
        self.col = col
    
    def safe(self, other):
        if other.row == self.row and other.col == self.col:
            return True
        if other.row == self.row:
            return False
        if other.col == self.col:
            return False
        slope = abs((self.row - other.row) / (self.col - other.col))
        if slope == 1:
            return False
        return True

    def __str__(self):
        return str(self.col + 1)
        

class Board:
    def __init__(self, state):
        """Initialise the board from a state string."""
        self._queens = []
        self._size = len(state)
        tState = [int(c) for c in state]
        assert(max(tState) <= self._size)
        for i in range(0, self._size):
            self._queens.append(Queen(i, tState[i] - 1))
        self.max_threats = math.factorial(self._size) / (math.factorial(self._size - 2) * 2)
        self._fitness = -1
        
    def mutate(self, p):
        s = str(self)
        ms = ''
        
        for c in s:
            x = random.random()
            if x <= p:
                new_c = c
                while new_c == c:
                    new_c = str(random.randint(1, len(s)))
                ms += new_c
            else:
                ms += c
    
        return Board(ms)
        
    def fitness(self):
        if self._fitness > -1:
            return self._fitness
        
        threats = 0
        for q in self._queens:
            for o in self._queens:
                if not q.safe(o):
                    threats += 1
        self._fitness = self.max_threats - (threats / 2)
        return self._fitness
    
    def mate(self, other, i):
        sqs = ''.join(str(q) for q in self._queens[:i])
        oqs = ''.join(str(q) for q in other._queens[i:])
        return Board(sqs + oqs)
    
    def __str__(self):
        return ''.join([str(q) for q in self._queens])
    
    def __len__(self):
        return self._size
        
    def goal_test(self):
        s = str(self)
        if len(s) != len(set(s)):
            return False
            
        if s.fitness() < self.max_threats:
            return False
            
        return True


def random_board(n=8):
    return ''.join([str(random.randint(1, n)) for _ in range(n)])

def make_boards(*args):
    boards = []
    for arg in args:
        boards.append(Board(arg))
    return boards


def crossover(a, b, i=-1):
    assert(a._size == b._size)
    if i == -1:
        i = random.randint(1, len(a) - 1)
    return a.mate(b, i), b.mate(a, i)
    
    
class Population:
    def __init__(self, members, mutation_rate=0.2):
        self._members = members[:]
        self._mr = mutation_rate

    def select(self, n=4):
        assert(n % 2 == 0)
        scores = [parent.fitness() for parent in self._members]
        total = sum(scores)
        pcts = [round(score / total * 100) for score in scores]
        cutoff = 0
        cutoffs = []
        for pct in pcts:
            cutoff += pct
            cutoffs.append(cutoff)
    
        chosen = []
        while len(chosen) < n:
            x = random.randint(1, 100)
            for i in range(len(cutoffs)):
                if x <= cutoffs[i]:
                    chosen.append(self._members[i])
                    break
        return chosen
        
    def iterate(self, n=4, i=-1):
        chosen = self.select(n)
        assert(len(chosen) != 0)
        children = []
        for j in range(0, len(chosen), 2):
            children.extend(crossover(chosen[j], chosen[j+1], i))
            
        for j in range(0, len(children)):
            child = children[j]
            children[j] = child.mutate(self._mr)
            
        return Population(children)
        
    def display(self):
        s = ''
        for member in self._members:
            s += '{}: {}'.format(member, member.fitness())
        return s
            
    def size(self):
        return len(self._members)
        
    def best(self):
        best = None
        best_score = -1
        
        for member in self._members:
            score = member.fitness()
            if score > best_score:
                best_score = score
                best = member
        
        return best

    def max_fitness(self):
        return self.best().fitness()
        

class Generation:
    def __init__(self, parents, i=4, p=0.2):
        self._parents = parents
        self._children = None
        self._best_parent = parents.best()
        self._best_child = None
        self.n = parents.size()
        self.i = i
        self.p = p
    
    def score(self):
        if self._best_child is None:
            self.run()
        return self._best_child.max_fitness()
    
    def run(self):
        if self._children is not None:
            return self._children
        
        self._children = self._parents.iterate(self.n, self.i)
        self._best_child = self._children.best()
        return self._children
        
    def __str__(self):
        if self._children is None:
            self.run()
        
        s = """
Parents:\t Children:
"""
        for i in range(0, self.n):
            p = self._parents._members[i]
            c = self._children._members[i]
            s += "{} ({})\t| {} ({})\n".format(p, p.fitness(), c, c.fitness())
        
        bpf = self._best_parent.fitness()
        bcf = self._best_child.fitness()
        s += """
Best:
    Parent: {}
    Child:  {}
""".format(bpf, bcf)

        s += "Generation has "
        if bpf < bcf:
            s += "improved"
        elif bpf == bcf:
            s += "stagnated"
        else:
            s += "regressed"
            
        return s + ".\n"
        
    def progress(self):
        bpf = self._best_parent.fitness()
        bcf = self._best_child.fitness()
        return bcf - bpf

class Experiment:
    def __init__(self, starting=None, popsize=4):
        if starting is None:
            starting = Population([Board(random_board()) for _ in range(popsize)])
            
        self.generations = []
        self.latest = starting
        
    def completed(self):
        if self.latest is None:
            return False
        
        if self.latest.max_fitness() < self.latest._members[0].max_threats:
            return False
            
        return True
        
    def score(sef):
        return self.latest.max_fitness()
        
    def step(self):
        if self.completed():
            return 0

        g = Generation(self.latest)
        self.latest = g.run()
        self.generations.append(g)
        return g.progress()
        
    def run(self, max_steps=0, verbose=False, quiet=False):
        n = len(self.generations)
        stop_at = 0
        stop_cond = lambda : False
        if max_steps > 0:
            stop_at = n + max_steps
            stop_cond = lambda :  len(self.generations) >= stop_at
        
        cpr = 0 # chars printed
        while not stop_cond():
            if self.completed():
                break
            progress = self.step()
            if verbose:
                print(self.generations[:-1])
            elif not quiet:
                if progress < 0:
                    sys.stdout.write('<')
                elif progress == 0:
                    sys.stdout.write('-')
                else:
                    sys.stdout.write('>')
                sys.stdout.flush()
                
                cpr += 1
                if cpr == 73:
                    cpr = 0
                    sys.stdout.write('\n')

        if not verbose and not quiet:
            sys.stdout.write('\n')
        return self.completed()
            
    def __str__(self):
        if self.completed():
            return 'Solution: {} in {} generations.'.format(self.best(), len(self.generations))
        else:
            return 'No solution yet.'
        
    def best(self):
        return self.latest.best()
        
    def __len__(self):
        return len(self.generations)
        

def random_experiment():
    exp = Experiment()
    start = time.process_time()
    exp.run()
    stop = time.process_time()
    print(exp)
    print('Elapsed:', datetime.timedelta(seconds=stop-start))
    
if __name__ == '__main__':
    random_experiment()