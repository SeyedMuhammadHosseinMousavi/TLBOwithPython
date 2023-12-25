import numpy as np
import matplotlib.pyplot as plt

# Cost Function
def CostFunction(x):
    return np.sum(x**2)

nVar = 10          # Number of Unknown Variables
VarSize = (1, nVar) # Unknown Variables Matrix Size
VarMin = -10       # Unknown Variables Lower Bound
VarMax = 10       # Unknown Variables Upper Bound

# TLBO Parameters
MaxIt = 1000        # Maximum Number of Iterations
nPop = 50           # Population Size

# Empty Structure for Individuals
class Individual:
    def __init__(self):
        self.Position = None
        self.Cost = None

# Initialize Population Array
pop = np.empty(nPop, dtype=object)

# Initialize Best Solution
BestSol = Individual()
BestSol.Cost = np.inf

# Initialize Population Members
for i in range(nPop):
    pop[i] = Individual()
    pop[i].Position = np.random.uniform(VarMin, VarMax, VarSize)
    pop[i].Cost = CostFunction(pop[i].Position)
    if pop[i].Cost < BestSol.Cost:
        BestSol = pop[i]

# Initialize Best Cost Record
BestCosts = np.zeros(MaxIt)

# TLBO Main Loop
for it in range(MaxIt):
    # Calculate Population Mean
    Mean = np.zeros(VarSize)
    for i in range(nPop):
        Mean += pop[i].Position
    Mean /= nPop

    # Select Teacher
    Teacher = pop[0]
    for i in range(1, nPop):
        if pop[i].Cost < Teacher.Cost:
            Teacher = pop[i]

    # Teacher Phase
    for i in range(nPop):
        # Create Empty Solution
        newsol = Individual()

        # Teaching Factor
        TF = np.random.randint(1, 3)

        # Teaching (moving towards teacher)
        newsol.Position = pop[i].Position + np.random.rand(*VarSize) * (Teacher.Position - TF * Mean)

        # Clipping
        newsol.Position = np.maximum(newsol.Position, VarMin)
        newsol.Position = np.minimum(newsol.Position, VarMax)

        # Evaluation
        newsol.Cost = CostFunction(newsol.Position)

        # Comparision
        if newsol.Cost < pop[i].Cost:
            pop[i] = newsol
            if pop[i].Cost < BestSol.Cost:
                BestSol = pop[i]

    # Learner Phase
    for i in range(nPop):
        A = np.arange(nPop)
        A = np.delete(A, i)
        j = np.random.choice(A)
        Step = pop[i].Position - pop[j].Position
        if pop[j].Cost < pop[i].Cost:
            Step = -Step

        # Create Empty Solution
        newsol = Individual()

        # Teaching (moving towards teacher)
        newsol.Position = pop[i].Position + np.random.rand(*VarSize) * Step

        # Clipping
        newsol.Position = np.maximum(newsol.Position, VarMin)
        newsol.Position = np.minimum(newsol.Position, VarMax)

        # Evaluation
        newsol.Cost = CostFunction(newsol.Position)

        # Comparision
        if newsol.Cost < pop[i].Cost:
            pop[i] = newsol
            if pop[i].Cost < BestSol.Cost:
                BestSol = pop[i]

    # Store Record for Current Iteration
    BestCosts[it] = BestSol.Cost

    # Show Iteration Information
    print('Iteration', it, ': Best Cost =', BestCosts[it])

# Results
plt.figure()
plt.semilogy(BestCosts, linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Best Cost')
plt.grid(True)
plt.show()


