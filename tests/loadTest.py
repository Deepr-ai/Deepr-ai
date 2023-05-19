from deeprai import models as model
import numpy as np

net = model.FeedForward()
net.load("log10.deepr")

for i in range(1,1000):
    out = net.run(np.array([i/100]))*100
    exp = np.log10(i)
    error = np.abs((out-exp)/exp)*100
    print(f"    Log {i} | Neural Net Prediction:{np.round(out, 4)} | Expected: {np.round(exp, 4)} | Relative Error: {np.round(error, 3)}% \n")
