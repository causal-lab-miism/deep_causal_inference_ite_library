import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
ax = plt.figure().gca()
ax.yaxis.set_major_locator(MaxNLocator(integer=True))


def plot_outcom_dist(y0, y1):
    y0, y1 = y0.numpy(), y1.numpy()
    plt.hist(y0, 30, alpha=0.7, label='y(t=0)')
    plt.hist(y1, 30, alpha=0.7, label='y(t=1)')
    plt.xlabel('outcome y(t)')
    plt.ylabel('# occurences')
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.show()