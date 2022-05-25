import matplotlib.pyplot as plt
import numpy as np

err = np.loadtxt("errors.txt");
time = np.loadtxt("time.txt");
vx = np.loadtxt("Vx.txt");
vy = np.loadtxt("Vy.txt");
vz = np.loadtxt("Vz.txt");
vyaw = np.loadtxt("Vyaw.txt");

plt.plot(time,err)
plt.ylabel('Error Promedio (pixeles)')
plt.xlabel('Tiempo (s)')
plt.show()

fig, ax = plt.subplots()
ax.plot(time, vx, label='$V_x$')
ax.plot(time,vy,label='$V_y$')
ax.plot(time,vz,label='$V_z$')
ax.plot(time,vyaw,label='$W_z$')
legend = ax.legend(loc='upper center', shadow=True)

plt.ylabel('Velocidades (u/s)')
plt.xlabel('Tiempo (s)')
plt.show()

