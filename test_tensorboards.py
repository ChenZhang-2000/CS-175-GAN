import matplotlib.pyplot as plt
import pandas as pd

g_loss = pd.read_csv("logs/202211290202_g_loss.csv", header=0)
d_loss = pd.read_csv("logs/202211290202_d_loss.csv", header=0)

# print(g_loss)

plt.plot(g_loss["Step"], g_loss["Value"])
plt.title("Generator Loss")
plt.show()
plt.plot(d_loss["Step"], d_loss["Value"])
plt.title("Discriminator Loss")
plt.show()

