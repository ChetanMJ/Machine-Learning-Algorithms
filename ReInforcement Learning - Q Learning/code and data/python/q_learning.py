from environment import MountainCar
import sys
import numpy as np
from random import randint

def main(args):
    pass

def q_val(state, weight, bias, mode):
    if mode == 'raw':
        s = np.zeros(2)
    else:
        s = np.zeros(2048)
    for key,val in state.items():
        s[key] = val
    return (np.dot(np.array(s), weight) + bias), np.array(s)

if __name__ == "__main__":
    main(sys.argv)
    
    mode = sys.argv[1]
    weight_out = sys.argv[2]
    returns_out = sys.argv[3]
    episodes = int(sys.argv[4])
    max_iterations = int(sys.argv[5])
    epsilon = float(sys.argv[6])
    gamma = float(sys.argv[7])
    learning_rate = float(sys.argv[8])

    alpha = learning_rate

    
    x = MountainCar(mode)
    if mode == 'raw': 
        w = np.zeros([2,3],dtype=float)
    else:
        w = np.zeros([2048,3],dtype=float)
    bias = 0.0
    
    ##print(x.state)
    
    
    ##print(q(x.state,1, w, bias))
    '''
    q = q_val(x.state, w, bias)
    for a in range(3):
        next_state, reward, done = x.step(a)
        q_next = q_val(next_state, w, bias)
        current_state = np.array([x.state[0], x.state[1]])
        ##print(alpha * (q[a] -(reward + (gamma * max(q_next)))) * current_state)
        w[:,0] = w[:,0] - (alpha * (q[a] -(reward + (gamma * max(q_next)))) * current_state)

    print(w)
    
    '''
    
    rng = np.random.RandomState()
    seed = rng.randint(2**31 - 1)
    rng.seed(seed)
    
    
    returns_out_file= open(returns_out,"w")
    
    for e in range(episodes):
       state = x.reset()
       #print(state)
       total_rewards = 0
       ##q = q_val(state, w, bias)          
       ##a = np.argmax(q)
       
       
       for i in range(max_iterations):
           
           q, current_state = q_val(state, w, bias, mode)          
           a = np.argmax(q)
           next_state, reward, done = x.step(a)
           q_next, next_state_np = q_val(next_state, w, bias, mode)
           next_random_action = rng.randint(0,2+1)
           ##current_state = np.array([x.state[0], x.state[1]])
           
           update = float(alpha) * (q[a] - (float(reward) + (float(gamma) * ((max(q_next) * (1.0 - epsilon)) + (q_next[next_random_action] * epsilon )))))
           ##w[:,a] = w[:,a] - (alpha * ((q[a] -(reward + (gamma * ((max(q_next) * (1.0 - epsilon)) + (q_next[next_random_action] * epsilon))))) * current_state))
           w[:,a] = w[:,a] - (update * current_state)
           
           ##w[:,next_random_action] = w[:,next_random_action] - (alpha * (q[next_random_action] -(reward + (gamma * q_next[next_random_action] * (epsilon)))) * current_state)
          
           
           bias = bias - update
           
           state = next_state
           ##q = q_next
           ##a = np.argmax(q) 
           if done :
               total_rewards = total_rewards + reward
               break
           total_rewards = total_rewards + reward
           
           
       returns_out_file.write(str(total_rewards) + "\n")
       
    
    #print(bias)       
    w_flat = w.flatten(order='C')
    weights = np.insert(w_flat,0,bias)
    np.savetxt(weight_out, weights)
    returns_out_file.close()
    
    
    
           
    
    