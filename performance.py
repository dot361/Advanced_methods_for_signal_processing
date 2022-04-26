import json
import matplotlib.pyplot as plt
import numpy as np

f = open('KLT_performance.json')
performance_rez = json.load(f)["results"]




SIdx = 0
fig = plt.figure()

means = list(map (lambda x:x['mean'],performance_rez))
Ss = list(map (lambda x: str(x['S']),performance_rez))
Ns = list(map (lambda x: str(x['N']),performance_rez))


plt.scatter(Ns,Ss, c=means, cmap='Oranges', s=500)
plt.xlabel("N")
plt.ylabel("S")
plt.colorbar()
plt.grid()
plt.minorticks_on()
plt.show()