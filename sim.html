import pulp
import pandas as pd
import numpy as np

# Read data from Excel file
df = pd.read_excel('data.xlsx')

# Parameters
n = len(df)  # Number of DMUs
m = 3  # Number of inputs (x_1, x_2, x_3)
s = 3  # Number of outputs (y_1, y_2, y_3)
L = 2  # Number of flexible measures (z_1, z_2)
M = 10000  # Large positive number for constraints
epsilon = 0.002  # Epsilon value to prevent zero weights

# Define the DEA model
def solve_dea_epsilon(exclude_dmus=[]):
    model = pulp.LpProblem("DEA_Epsilon_Model", pulp.LpMaximize)
    
    # Variables
    t = pulp.LpVariable("t", lowBound=0)
    v = [pulp.LpVariable(f"v_{i}", lowBound=epsilon) for i in range(m)]
    u = [pulp.LpVariable(f"u_{r}", lowBound=epsilon) for r in range(s)]
    gamma = [pulp.LpVariable(f"gamma_{l}", lowBound=0) for l in range(L)]
    delta = [pulp.LpVariable(f"delta_{l}", lowBound=0) for l in range(L)]
    d = [pulp.LpVariable(f"d_{l}", cat='Binary') for l in range(L)]
    b = [pulp.LpVariable(f"b_{j}", cat='Binary') for j in range(n)]
    
    # Objective function
    model += t
    
    # Constraints
    for j in range(n):
        input_sum = pulp.lpSum(v[i] * df[f'x_{i+1}'][j] for i in range(m)) + \
                    pulp.lpSum(gamma[l] * df[f'z_{l+1}'][j] for l in range(L))
        output_sum = pulp.lpSum(u[r] * df[f'y_{r+1}'][j] for r in range(s)) + \
                     pulp.lpSum(delta[l] * df[f'z_{l+1}'][j] for l in range(L))
        
        model += input_sum <= 1
        model += output_sum - input_sum <= M * b[j]
        model += output_sum - input_sum >= t + M * (b[j] - 1)
    
    model += pulp.lpSum(b[j] for j in range(n)) == 1
    
    for l in range(L):
        model += delta[l] <= M * d[l]
        model += gamma[l] <= M * (1 - d[l])
        model += delta[l] + gamma[l] >= epsilon
    
    # Exclude previously identified efficient DMUs
    for j in exclude_dmus:
        model += b[j] == 0
    
    # Solve the model
    model.solve()
    
    # Collect results
    if pulp.LpStatus[model.status] == 'Optimal':
        return {
            't': t.value(),
            'v': [v[i].value() for i in range(m)],
            'u': [u[r].value() for r in range(s)],
            'gamma': [gamma[l].value() for l in range(L)],
            'delta': [delta[l].value() for l in range(L)],
            'd': [d[l].value() for l in range(L)],
            'b': [b[j].value() for j in range(n)],
            'status': 'Optimal'
        }
    else:
        return {'status': 'Infeasible'}

# Run the algorithm to rank efficient DMUs
efficient_dmus = []
results = []

while True:
    result = solve_dea_epsilon(exclude_dmus=[j-1 for j in efficient_dmus])
    if result['status'] == 'Infeasible' or result['t'] == 0:
        break
    
    # Identify efficient DMU
    for j in range(n):
        if result['b'][j] == 1:
            efficient_dmus.append(j + 1)
            results.append({
                'DMU': j + 1,
                't': result['t'],
                'v': result['v'],
                'u': result['u'],
                'gamma': result['gamma'],
                'delta': result['delta'],
                'd': result['d']
            })
            break

# Print results
print("Ranking of Efficient DMUs:")
for i, res in enumerate(results):
    print(f"Rank {i+1}: DMU {res['DMU']}")
    print(f"  t*: {res['t']}")
    print(f"  v*: {res['v']}")
    print(f"  u*: {res['u']}")
    print(f"  gamma*: {res['gamma']}")
    print(f"  delta*: {res['delta']}")
    print(f"  d*: {res['d']}")
print(f"Number of Efficient DMUs: {len(efficient_dmus)}")