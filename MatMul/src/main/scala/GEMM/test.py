from copy import copy

class State:
    def __init__(self) -> None:
        self.dataOut = ("A", 0, 0)
        self.matSize = 4
        self.counter = 1234
        self.index = 1234
        self.jMax = 14
        self.iStart = 0
        self.kStart = 0
        self.aIn = [0 for i in range(self.matSize)]
        self.bIn = [0 for i in range(self.matSize)]

class Result:
    def __init__(self, next_step, stall=False, busy=False):
        self.next_step = next_step
        self.stall = stall
        self.busy = busy


def step0(s: State):
    return Result(next_step=1)

    
def step1(s: State):
    s.counter = 0
    return Result(next_step=2, stall=True, busy=True)


def step2(s: State):
    s.aIn = [0 for i in range(s.matSize)]
    s.bIn = [0 for i in range(s.matSize)]
    s.index = max(s.counter - s.jMax, 0)
    return Result(next_step=3, busy=True)


def step3(s: State):
    s.dataOut = ("A", s.iStart + s.index, s.counter - s.index)
    return Result(next_step=5, stall=True, busy=True)


def step4(s: State):
    s.bIn[s.index - 1] = s.dataOut
    s.dataOut = ("A", s.iStart + s.index, s.counter - s.index)
    return Result(next_step=5, stall=True, busy=True)


def step5(s: State):
    s.aIn[s.index] = s.dataOut
    s.dataOut = ("B", s.counter - s.index, s.kStart + s.index)
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
    s.bIn[s.index] = s.dataOut
    return Result(next_step=2, stall=True, busy=True)


def step7(s: State):
    s.bIn[s.index] = s.dataOut
    s.counter = 0
    return Result(next_step=8, stall=True, busy=True)

            
def step8(s: State):
    if s.counter == s.matSize:
        return Result(next_step=0, busy=True)
    else:
        s.counter += 1
        s.aIn = [0 for i in range(s.matSize)]
        s.bIn = [0 for i in range(s.matSize)]
        return Result(next_step=8, busy=True)
    

steps = [step0, step1, step2, step3, step4, step5, step6, step7, step8]
cur_step = 0
state = State()
for i in range(1000):
    prev_state = copy(state)
    result = steps[cur_step](state)
    if result.stall == False:
        print(i, cur_step, prev_state.aIn, prev_state.bIn)
    else:
        print(i, cur_step)
    if result.busy == False and i > 0:
        break
    cur_step = result.next_step