

def visualize_multivariate_2d(multivariate, n, ax, title):
    points = multivariate.sample(n)

    ax.scatter(points[:, 0], points[:, 1], color='blue', s=10)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    ax.set_title(title)

    return ax, points


def visualize_mixture_2d(mixture, n, ax, title):
    points = mixture.sample(n)

    ax.scatter(points[:, 0], points[:, 1], color='blue', s=10)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    ax.set_title(title)

    return ax, points