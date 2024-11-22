import numpy as np
import gymnasium as gym

N_HOLES = 6

class MancalaEnv(gym.Env):
    def __init__(self):
        self._first_player_holes = [4] * N_HOLES + [0]
        self._second_player_holes = [4] * N_HOLES + [0]
        
        self.player = 0
        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Box(
            low=0,           
            high=30,             
            shape=(2, 7),       
            dtype=np.int32       
        )
       
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._first_player_holes = [4] * N_HOLES + [0]
        self._second_player_holes = [4] * N_HOLES + [0]
        self.player = 0
        return self.state(), None

    def step(self, action):
        # Ustalamy, które dziury są nasze, a które przeciwnika
        our_holes = self._first_player_holes if self.player == 0 else self._second_player_holes
        opponent_holes = self._second_player_holes if self.player == 0 else self._first_player_holes
        # Kontrolujemy ostatnią dziurę, do której wrzuciliśmy kulę 
        last = False
        # Indeks wybranej dziury
        idx = action
        previous_h_n = our_holes[idx] 
        our_holes[idx] = 0 
        
        # Proces wykonywania ruchu
        i = idx+1
        for _ in range(previous_h_n):
            # Jeśli dotarliśmy do domku przeciwnika, indeksujemy od początku
            if i == 2*N_HOLES + 1:
                i = 0
                
            # Jeśli kulka spada po naszej stronie
            if i // (N_HOLES + 1) == 0:
                our_holes[i] += 1  
            # Jeśli kulka spada po stronie przeciwnika
            else:
                opponent_holes[i % (N_HOLES + 1)] += 1
            
            # Jeśli kulka spadła do naszego domku
            if i == N_HOLES:
                last = True
            else:
                last = False
            
            # Przechodzimy do kolejnej dziury     
            i += 1
              
        # Jeśli ostatni kamień spadł do domku, mamy kolejny ruch
        if last is True:
            return self.state(), self.reward(), self.terminated(), None
        # Jeśli nie, to sprawdzamy czy doszło do przejęcia kulek przeciwnika
        elif i - 1 < N_HOLES and our_holes[i-1] == 1 and opponent_holes[N_HOLES-i] != 0:
            our_holes[-1] += 1 + opponent_holes[N_HOLES-i]
            our_holes[i-1] = 0
            opponent_holes[N_HOLES-i] = 0
        self.player = int(not self.player)
        return self.state(), self.reward(), self.terminated(), None
    
    def state(self):
        if self.player == 0:
            return self.player, np.array([self._first_player_holes, self._second_player_holes])
        else:
            return self.player, np.array([self._second_player_holes, self._first_player_holes])
        
    def terminated(self):
        if (self._first_player_holes[-1] > 24) or (self._second_player_holes[-1] > 24):
            return True
        if (sum(self._first_player_holes[:N_HOLES]) == 0) or (sum(self._second_player_holes[:N_HOLES]) == 0):
            return True
        return False
    
    def reward(self):
        # Gra nieskończona
        if not self.terminated():
            return 0
        
        # Remis
        if sum(self._first_player_holes) == sum(self._second_player_holes):
            return 0
        
        sign = 1
        if self.player == 1:
            sign = -1
            
        if sum(self._first_player_holes) > sum(self._second_player_holes):
            return 1*sign
        else:
            return -1*sign
        
    def set_state(self, s):
        a1 = s[0, :]
        a2 = s[1, :]
        if self.player == 0:
            self._first_player_holes = np.copy(a1)
            self._second_player_holes = np.copy(a2)
        else:
            self._first_player_holes = np.copy(a2)
            self._second_player_holes = np.copy(a1)
            
    def possible_moves(self):
        if self.player == 0:
            return [i for i in range(6) if self._first_player_holes[i] != 0]
        else:
            return [i for i in range(6) if self._second_player_holes[i] != 0]
        
        
 
    
gym.envs.registration.register(
    id="Mancala-v0",
    entry_point="library.enviroment:MancalaEnv"
)     