from matplotlib import pyplot as plt
import numpy as np


def plot_mu_1():
    plt.figure()
    x = list(np.arange(0, 1., .1))
    x = np.array(x)

    y1 = [0.572, 0.654, 0.693, 0.741, 0.734, 0.732, 0.724, 0.729, 0.727, 0.725]
    e_y1 = np.array([0.022, 0.0094, 0.01, 0.0093, 0.014, 0.014, 0.019, 0.0021, 0.0022, 0.0021])

    plt.errorbar(x, y1, yerr=e_y1, fmt='g--*', label='otsc+lasso')

    y2 = [0.622, 0.644, 0.65, 0.673, 0.714, 0.729, 0.734, 0.728, 0.732, 0.745]
    e_y2 = [0.022, 0.014, 0.015, 0.0173, 0.0124, 0.0249, 0.0234, 0.028, 0.012, 0.0145]
    plt.errorbar(x, y2, yerr=e_y2, fmt='k--*', label='otsc+elastic')

    plt.title('Regularization parameter: $\mu_1$; $\mu_2=0.4$')
    plt.ylabel('Classification accuracy')
    plt.xlabel('$\mu_1$')
    plt.xticks(x)
    plt.legend()
    plt.savefig('mu_1.png',dpi=600)
    # plt.show()


def plot_mu_2():
    plt.figure()
    x = list(np.arange(0, 1., .1))
    x = np.array(x)

    y1 = [0.652, 0.674, 0.693, 0.683, 0.704, 0.692, 0.684, 0.699, 0.682, 0.691]
    e_y1 = np.array([0.022, 0.0094, 0.01, 0.0093, 0.014, 0.014, 0.019, 0.0021, 0.0022, 0.0021])

    plt.errorbar(x, y1, yerr=e_y1, fmt='g--*', label='otsc+lasso')

    y2 = [0.662, 0.664, 0.685, 0.693, 0.724, 0.709, 0.718, 0.715, 0.692, 0.71]
    e_y2 = [0.022, 0.014, 0.015, 0.0173, 0.0124, 0.0249, 0.0234, 0.028, 0.012, 0.0145]
    plt.errorbar(x, y2, yerr=e_y2, fmt='k--*', label='otsc+elastic')

    plt.title('Sparse regularization parameter: $\mu_2$; $\mu_1=0.6$')
    plt.ylabel('Classification accuracy')
    plt.xlabel('$\mu_2$')

    plt.legend()
    # plt.show()
    plt.xticks(x)
    plt.savefig('mu_2.png',dpi=600)

def plot_examples():
    plt.figure()
    x = list(range(0, 1000, 100))
    x = np.array(x)

    plt.plot(x, x * 0 + 0.877, 'r-*', label='svm')
    plt.plot(x, x * 0 + 0.712, 'b-*', label='stanford')

    y1 = [0.612, 0.634, 0.653, 0.683, 0.704, 0.714, 0.724, 0.729, 0.732, 0.741]
    e_y1 = np.array([0.022, 0.0094, 0.01, 0.0093, 0.014, 0.014, 0.019, 0.0021, 0.012, 0.021])

    plt.errorbar(x, y1, yerr=e_y1, fmt='g--*', label='otsc+lasso')

    y2 = [0.622, 0.644, 0.65, 0.673, 0.714, 0.729, 0.734, 0.728, 0.732, 0.745]
    e_y2 = [0.022, 0.014, 0.015, 0.0173, 0.0124, 0.0249, 0.0234, 0.028, 0.022, 0.0245]
    plt.errorbar(x, y2, yerr=e_y2, fmt='k--*', label='otsc+elastic')

    plt.title('Classification accuracy vs # of labeled examples; $\mu_1=0.6$ and $\mu_2=0.4$')
    plt.ylabel('Classification accuracy')
    plt.xlabel('# of labeled examples')
    plt.xticks(x)

    plt.legend()
    # plt.show()

    plt.savefig('examples.png',dpi=600)

if __name__ == '__main__':
    plot_mu_1()
    plot_mu_2()
    plot_examples()

    print('Done..')
