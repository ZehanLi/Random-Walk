from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


# TD Lambda
def tdupdate(alpha, _lambda, state_sequence, values):

    # we have 7 possible states, with two of them being end states (A,G)

    eligibility = np.zeros(7)
    updates = np.zeros(7)

    for t in range(0, len(state_sequence) - 1):
        current_state = state_sequence[t]
        next_state = state_sequence[t + 1]

        eligibility[current_state] += 1.0

        td = alpha * (values[next_state] - values[current_state])
        updates += td * eligibility
        eligibility *= _lambda

    return updates


states = ['A', 'B', 'C', 'D', 'E', 'F', 'G']


def simulate():
    states = [3]  # Starting point
    while states[-1] not in [0, 6]:
        states.append(states[-1] + (1 if random.choice([True, False]) else -1))

    return states


random.seed(101)
truth = [1 / 6, 1 / 3, 1 / 2, 2 / 3, 5 / 6]

dtype = np.float

num_train_sets = 100
num_sequences = 10  # or episodes

training_sets = [[simulate() for i in range(num_sequences)] for i in range(num_train_sets)]

# Figure 3
alphas = np.array([0.01], dtype=dtype)
lambdas = np.array([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0], dtype=dtype)

results = []

for _lambda in lambdas:
    print('current lambda is', _lambda)
    for alpha in alphas:
        rmses = []
        for training_set in training_sets:
            values = np.zeros(7, dtype=dtype)
            iterations = 0

            while 1:
                iterations += 1
                before = np.copy(values)
                updates = np.zeros(7, dtype=dtype)
                values[6] = 1.0

                for sequence in training_set:
                    updates += tdupdate(alpha, _lambda, sequence, values)

                values += updates
                diff = np.sum(np.absolute(before - values))

                if diff < .000001:
                    break

            estimate = np.array(values[1:-1], dtype=dtype)
            error = (truth - estimate)
            rms = np.sqrt(np.average(np.power(error, 2)))
            rmses.append(rms)

        result = [_lambda, alpha, np.mean(rmses)]
        results.append(result)

data = pd.DataFrame(results)
data.columns = ["lambda", "alpha", "rms"]
data.head()

data = data[data.groupby(['lambda'])['rms'].transform(min) == data['rms']].set_index(keys=['lambda'])
data.drop('alpha', 1, inplace=True,)

# sns.set_style("white")
plt.figure(num=None, figsize=(10, 6), dpi=72)
plt.margins(.05)
plt.xlabel(r"$\lambda$")
plt.ylabel("RMS")
plt.title("Figure 3")
plt.xticks([i * .1 for i in range(0, 10)])
plt.yticks([i * .01 for i in range(10, 19)])
plt.plot(data,marker='o');

plt.savefig('fig3.png')


alphas = [0.05 * i for i in range(0,16)]
lambdas = [0.05 * i for i in range(0, 21)]

results = []

for _lambda in lambdas:
    print('current lambda is', _lambda)
    for alpha in alphas:
        rms_vals = []
        for training_set in training_sets:

            values = np.array([0.5 for i in range(7)])

            for sequence in training_set:
                values[0] = 0.0
                values[6] = 1.0
                values += tdupdate(alpha, _lambda, sequence, values)

            estimate = np.array(values[1:-1])
            error = (truth - estimate)
            rms   = np.sqrt(np.average(np.power(error, 2)))

            rms_vals.append(rms)

        result = [_lambda, alpha, np.mean(rms_vals)]
        results.append(result)

data = pd.DataFrame(results)

data.columns = ["lambda", "alpha", "rms"]
# print(data.to_string())

data4 = data.drop("lambda", 1).set_index(keys=['alpha'])
# print(data4.head())
plt.figure(num=None, figsize=(10, 6), dpi=80)
plt.plot(data4.head(12), marker='o')
plt.plot(data4.iloc[96:108], marker='o')
plt.plot(data4.iloc[256:268], marker='o')
plt.plot(data4.iloc[320:332], marker='o')
plt.ylim(0, 0.8)
plt.legend((r"$\lambda=0$", r"$\lambda=0.3$", r"$\lambda=0.8$", r"$\lambda=1$"))
plt.margins(.10)
plt.xticks([i * .1 for i in range(0, 6)])
plt.yticks([i * .1 for i in range(0, 8)])
plt.xlabel(r"$\alpha$")
plt.ylabel("RMS")
plt.title("Figure 4 ")
plt.text(-.25,.204, "ERROR",size=12)

plt.savefig('fig4.png')

data5 = data[data.groupby(['lambda'])['rms'].transform(min) == \
            data['rms']].set_index(keys=['lambda'])

data5 = data5.drop('alpha', 1)
# print(data5.head())

plt.figure(num=None, figsize=(10, 6), dpi=80)
plt.plot(data5, marker='o')
plt.margins(.10)
plt.xlabel(r"$\lambda$")
plt.ylabel("RMS")
plt.title("Figure 5 ")
plt.text(-.25,.204, "ERROR\nUSING\nBEST α",size=12)

plt.savefig('fig5.png')