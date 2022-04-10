import numpy as np

start_state = np.zeros([2,3,3])
start_state[0,1,:2] = 1
start_state[1,0,1] = 1
start_state[1,1,2] = 1

def check_win(state):
    row = np.argwhere(np.sum(state, axis=1)==3)
    column = np.argwhere(np.sum(state, axis=2)==3)
    diag = np.argwhere(np.trace(state,axis1=1,axis2=2)==3)
    flip_diag = np.argwhere(np.trace(np.flip(state,1),axis1=1,axis2=2)==3)
    row_any = row.size > 0
    column_any = column.size > 0
    diag_any = diag.size > 0
    flip_diag_any = flip_diag.size > 0
#     print("{}, {}, {}, {}".format(row, column, diag, flip_diag))
    if row_any or column_any or diag_any or flip_diag_any:
        x = np.append(row, column)
        y = np.append(diag, flip_diag)
        return True, np.append(x,y)[0]
    elif np.sum(state) == 9:
        return True, None
    else:
        return False, None

def show_board(state):
    for i in range(3):
        for j in range(3):
            if state[0,i,j] == 1:
                print("x", end=' ')
            elif state[1,i,j] == 1:
                print("o", end=' ')
            else:
                print(" ", end=' ')
        print()

def available_move(state):
    return np.argwhere(state[0] + state[1] == 0).tolist()

def update_state(state, agent, action):
    current_state = state.copy()
    current_state[agent][tuple(available_move(current_state)[int(action)])] = 1
    return current_state
    
def purely_random(state):
    state_list = available_move(state)
    if state_list:
        random_index = np.random.randint(len(state_list))
        return random_index
    else:
        return None

def UCB(wins, n_i, N):
    c = np.sqrt(2)
    if np.any(n_i == 0):
        return np.argwhere(n_i == 0)[0]
    else:
        a = np.argmax(wins/n_i + c*np.sqrt(np.log(N)/n_i))
    return a

def UCT(start_state,k):
    T = 1000
    n_i = np.zeros(k)
    wins = np.zeros(k)
    for t in range(T):
        a_t = UCB(wins, n_i, t)
        n_i[a_t] += 1
        s = start_state.copy()
        s = update_state(s, 0, a_t)
        win = False
        agent2 = True
        while not win:
            win, agent = check_win(s)
            if agent == 0:
                wins[a_t] += 1
            # roll-out
            a = purely_random(s)
            if a != None:
                if agent2:
                    s = update_state(s, 1, a)
                    agent2 = False
                else:
                    s = update_state(s, 0, a)
                    agent2 = True
#     print(wins/n_i)
#     print(n_i)
    return np.argmax(wins/n_i)

def UCT2(start_state,k):
    T = 1000
    n_i = np.zeros(k)
    wins = np.zeros(k)
    opposite_wins = np.zeros(k)
    for t in range(T):
        a_t = UCB(wins, n_i, t)
        n_i[a_t] += 1
        s = start_state.copy()
        s = update_state(s, 1, a_t)
        win = False
        agent2 = True
        while not win:
            win, agent = check_win(s)
            if agent == 1:
                wins[a_t] += 1
            elif agent == 0:
                opposite_wins[a_t] += 1
            # roll-out
            a = purely_random(s)
            if a != None:
                if agent2:
                    s = update_state(s, 0, a)
                    agent2 = False
                else:
                    s = update_state(s, 1, a)
                    agent2 = True
    q = wins/n_i
    q_opposite = opposite_wins/n_i
    
    min_action = []
    min_a = np.argmin(q_opposite)
    for index, i in enumerate(q_opposite):
        if i == q_opposite[min_a]:
            min_action.append(index)
    max_min_index = np.argmax(q[min_action])
    return min_action[max_min_index]

#simulation
episode = 1000
action_count = np.zeros((5))
win_count = np.zeros((5))
for i in range(episode):
    a_1 = UCT(start_state, len(available_move(start_state)))
    state = update_state(start_state, 0, a_1)
    win, agent = check_win(state)
#     show_board(state)
    agent2 = True
    while not win:
        if agent2:
            a_2 = purely_random(state)
            state = update_state(state, 1, a_2)
            win, agent = check_win(state)
#             show_board(state)
            agent2 = False
        else:
            a_2 = UCT(state, len(available_move(state)))
            state = update_state(state, 0, a_2)
            win, agent = check_win(state)
#             show_board(state)
            agent2 = True
#     print(agent)
    action_count[a_1] += 1
    if agent == 0:
        win_count[a_1] += 1
print(action_count)
print(win_count)
        
    
# start_state
# available_move(start_state)
#simulation
episode = 1000
action_count = np.zeros((5))
win_count = np.zeros((5))
for i in range(episode):
    a_1 = UCT2(start_state, len(available_move(start_state)))
    state = update_state(start_state, 1, a_1)
    win, agent = check_win(state)
#     show_board(state)
    agent2 = True
    while not win:
        if agent2:
            a_2 = purely_random(state)
            state = update_state(state, 0, a_2)
            win, agent = check_win(state)
#             show_board(state)
            agent2 = False
        else:
            a_2 = UCT2(state, len(available_move(state)))
            state = update_state(state, 1, a_2)
            win, agent = check_win(state)
#             show_board(state)
            agent2 = True
#     print(agent)
    action_count[a_1] += 1
    if agent == 1:
        win_count[a_1] += 1
print(action_count)
print(win_count)
#simulation
episode = 1000
action_count = np.zeros((5))
win_count = np.zeros((5))
for i in range(episode):
    a_1 = UCT(start_state, len(available_move(start_state)))
    state = update_state(start_state, 0, a_1)
    win, agent = check_win(state)
#     show_board(state)
    agent2 = True
    while not win:
        if agent2:
            a_2 = UCT2(state, len(available_move(state)))
            state = update_state(state, 1, a_2)
            win, agent = check_win(state)
#             show_board(state)
            agent2 = False
        else:
            a_2 = UCT(state, len(available_move(state)))
            state = update_state(state, 0, a_2)
            win, agent = check_win(state)
#             show_board(state)
            agent2 = True
#     print(agent)
    action_count[a_1] += 1
    if agent == 0:
        win_count[a_1] += 1
print(action_count)
print(win_count)