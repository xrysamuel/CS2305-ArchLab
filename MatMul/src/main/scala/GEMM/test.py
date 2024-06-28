from copy import copy

class Element:
    def __init__(self, name="A", row=0, col=0, zero=False):
        self.name = name
        self.row = row
        self.col = col
        self.zero = zero

    def __repr__(self):
        if self.zero:
            return f"0"
        else:
            return f"{self.name}[{self.row}][{self.col}]"


class SA:
    def __init__(self, matSize):
        self.matSize = matSize
        self.cells = [[[] for j in range(self.matSize)] for i in range(self.matSize)]
        self.horizontals = [[Element(zero=True) for j in range(self.matSize)] for i in range(self.matSize)]
        self.verticals = [[Element(zero=True) for i in range(self.matSize)] for j in range(self.matSize)]

    def step(self, aIn, bIn):
        self.horizontals[0] = aIn
        self.verticals[0] = bIn
        for i in range(self.matSize):
            for j in range(self.matSize):
                self.cells[i][j].append((self.horizontals[j][i], self.verticals[i][j]))
        self.horizontals = self.horizontals[:-1]
        self.verticals = self.verticals[:-1]
        self.horizontals.insert(0, [Element(zero=True) for j in range(self.matSize)])
        self.verticals.insert(0, [Element(zero=True) for i in range(self.matSize)])

    def flush(self):
        self.cells = [[[] for j in range(self.matSize)] for i in range(self.matSize)]
        self.horizontals = [[Element(zero=True) for j in range(self.matSize)] for i in range(self.matSize)]
        self.verticals = [[Element(zero=True) for i in range(self.matSize)] for j in range(self.matSize)]

        
class State:
    def __init__(self) -> None:
        self.dataIn = Element()
        self.matSize = 4
        self.counter = 1234
        self.index = 1234
        self.jMax = 15
        self.iStart = 4
        self.kStart = 8
        self.aIn = [Element(zero=True) for i in range(self.matSize)]
        self.bIn = [Element(zero=True) for i in range(self.matSize)]
        self.sa = SA(self.matSize)

class Result:
    def __init__(self, next_step, stall=False, busy=False, flush=False):
        self.next_step = next_step
        self.stall = stall
        self.busy = busy
        self.flush = flush



def step0(s: State):
    return Result(next_step=1, flush=True)

    
def step1(s: State):
    s.counter = 0
    return Result(next_step=2, stall=True, busy=True)


def step2(s: State):
    s.aIn = [Element(zero=True) for i in range(s.matSize)]
    s.bIn = [Element(zero=True) for i in range(s.matSize)]
    s.index = max(s.counter - s.jMax, 0)
    return Result(next_step=3, busy=True)


def step3(s: State):
    s.dataIn = Element("A", s.iStart + s.index, s.counter - s.index)
    return Result(next_step=5, stall=True, busy=True)


def step4(s: State):
    s.bIn[s.index - 1] = s.dataIn
    s.dataIn = Element("A", s.iStart + s.index, s.counter - s.index)
    return Result(next_step=5, stall=True, busy=True)


def step5(s: State):
    s.aIn[s.index] = s.dataIn
    s.dataIn = Element("B", s.counter - s.index, s.kStart + s.index)
    if s.index == min(s.counter, s.matSize - 1):
        if s.counter == s.matSize - 1 + s.jMax:
            return Result(next_step=7, stall=True, busy=True)
        else:
            s.counter += 1
            return Result(next_step=6, stall=True, busy=True)
    else:
        s.index += 1
        return Result(next_step=4, stall=True, busy=True)
    

def step6(s: State):
    s.bIn[s.index] = s.dataIn
    return Result(next_step=2, stall=True, busy=True)


def step7(s: State):
    s.bIn[s.index] = s.dataIn
    s.counter = 0
    return Result(next_step=8, stall=True, busy=True)

            
def step8(s: State):
    print(f"\nC[{s.iStart + s.counter // s.matSize}][{s.kStart + s.counter % s.matSize}] = ", end='')
    for m1, m2 in s.sa.cells[s.counter // s.matSize][s.counter % s.matSize]:
        print(f"({m1} * {m2}) + ", end='')
    print("\n")
    s.aIn = [Element(zero=True) for i in range(s.matSize)]
    s.bIn = [Element(zero=True) for i in range(s.matSize)]
    if s.counter == s.matSize * s.matSize - 1:
        return Result(next_step=0, busy=True)
    else:
        s.counter += 1
        return Result(next_step=8, busy=True)
    

steps = [step0, step1, step2, step3, step4, step5, step6, step7, step8]
cur_step = 0
state = State()
for i in range(2):
    for clock in range(1000):
        prev_state = copy(state)
        result = steps[cur_step](state)
        if result.stall == False:
            state.sa.step(prev_state.aIn, prev_state.bIn)
            print(clock, cur_step, prev_state.aIn, prev_state.bIn)
        else:
            print(clock, cur_step)
        if result.flush == True:
            state.sa.flush()
        if result.busy == False and clock > 0:
            break
        cur_step = result.next_step