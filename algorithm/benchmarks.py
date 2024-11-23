import random

ACTIONS = range(6)

def rnd_policy(s):
    actions = []
    for a in ACTIONS:
        if s[0, a] != 0:
            actions.append(a)
            
    return random.choice(actions)

def max_play_policy(s):
    actions = []
    for a in ACTIONS:
        if s[0, a] != 0 and s[0, a] == (6-a):
            actions.append(a)
            
    if actions:
        return actions[-1]
    return rnd_policy(s)

def always_gain_policy(s):
    actions = []
    for a in ACTIONS:
        if s[0, a] >= 6-a:
            actions.append(a)
            
    if actions:
        return random.choice(actions)
    return rnd_policy(s)

def capture_policy(s):
    actions = []
    for a in ACTIONS:
        if s[0, a] != 0 and (s[0, a] + a) % 12 < 6 and s[0, (s[0, a] + a)%12] == 0 and s[1, 5-(s[0, 1, a] + a)%12] != 0:
            actions.append(a)
            
    if actions:
        actions.sort(key=lambda a: s[1, 5-(s[0, 1, a] + a)%12])
        return actions[-1]
    return rnd_policy(s)

def united_policy(s):
    actions = []
    for a in ACTIONS:
        if s[0, a] != 0 and s[0, a] == (6-a):
            actions.append(a)
            
    if actions:
        return actions[-1]
    
    for a in ACTIONS:
        if s[0, a] != 0 and (s[0, a] + a) % 12 < 6 and s[0, (s[0, a] + a)%12] == 0 and s[1, 5-(s[0, a] + a)%12] != 0:
            actions.append(a)
            
    if actions:
        actions.sort(key=lambda a: s[1, 5-(s[0, a] + a)%12])
        return actions[-1]
    
    for a in ACTIONS:
        if s[0, a] >= 6-a:
            actions.append(a)
            
    if actions:
        return random.choice(actions)
    return rnd_policy(s)
    
    
def united_policy_2(s):
    actions = []
    for a in ACTIONS:
        if s[0, a] != 0 and (s[0, a] + a) % 12 < 6 and s[0, (s[0, a] + a)%12] == 0 and s[1, 5-(s[0, a] + a)%12] != 0:
            actions.append(a)
            
    if actions:
        actions.sort(key=lambda a: s[1, 5-(s[0, a] + a)%12])
        return actions[-1]
    
    for a in ACTIONS:
        if s[0, a] != 0 and s[0, a] == (6-a):
            actions.append(a)
            
    if actions:
        return actions[-1]
    
    for a in ACTIONS:
        if s[0, a] >= 6-a:
            actions.append(a)
            
    if actions:
        return random.choice(actions)
    return rnd_policy(s)