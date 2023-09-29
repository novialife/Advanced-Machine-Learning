import random
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np

log_dict_pi = {}
log_dict_tau = {}
log_dict_lambd = {}
log_dict_fact_s = {}
log_2_pi = np.log(np.pi)


def calculate_log_r_nk_numerator(x_1, x_2, s, pi, lambd, mu_1, tau_1, mu_2, tau_2):
    # prior part
    try:
        sum = log_dict_pi[pi]
    except KeyError:
        sum = log_dict_pi[pi] = np.log(pi)

    # gauss part
    for mu_d, tau_d, x_d in zip([mu_1, mu_2], [tau_1, tau_2], [x_1, x_2]):
        try:
            sum += log_dict_tau[tau_d] * 0.5
        except KeyError:
            log_dict_tau[tau_d] = np.log(tau_d)
            sum += log_dict_tau[tau_d] * 0.5
        sum -= log_2_pi
        sum -= tau_d * ((x_d - mu_d) ** 2) / 2

    # poisson part
    try:
        sum += s * log_dict_tau[lambd]
    except:
        log_dict_tau[lambd] = np.log(lambd)
        sum += s * log_dict_tau[lambd]
    sum -= lambd

    # Not needed and causes issues

    # try:
    #     sum -= log_dict_fact_s[s]
    # except KeyError:
    #     log_dict_fact_s[s] = np.log(math.factorial(s))
    #     sum -= log_dict_fact_s[s]

    return sum


def display_distributions(x_cords, y_cords, magnitude, theta):
    # Taken from https://stackoverflow.com/questions/9215658/plot-a-circle-with-pyplot
    # and adapted to the question at hand
    s_factor = 10
    ax = plt.gca()
    ax.cla()  # clear things for fresh plot
    for x, y, s in zip(x_cords, y_cords, magnitude):
        ax.scatter(x, y, s=s*s_factor, c='k', alpha=0.5)
    for theta_k in theta:
        circle_std_1 = Ellipse(xy=(theta_k['mu_x'], theta_k['mu_y']),
                               width=(1/theta_k['tau_x'])**0.5,
                               height=(1/theta_k['tau_y'])**0.5,
                               color='r',
                               alpha=0.2)
        circle_std_2 = Ellipse(xy=(theta_k['mu_x'], theta_k['mu_y']),
                               width=2*((1 / theta_k['tau_x']) ** 0.5),
                               height=2*((1 / theta_k['tau_y']) ** 0.5),
                               color='r',
                               alpha=0.2)
        circle_std_3 = Ellipse(xy=(theta_k['mu_x'], theta_k['mu_y']),
                               width=3 * ((1 / theta_k['tau_x']) ** 0.5),
                               height=3 * ((1 / theta_k['tau_y']) ** 0.5),
                               color='r',
                               alpha=0.2)
        ax.add_patch(circle_std_1)
        ax.add_patch(circle_std_2)
        ax.add_patch(circle_std_3)
        ax.scatter(theta_k['mu_x'], theta_k['mu_y'], s=theta_k['lambda'] * s_factor, c='red', alpha=1.0)
        ax.scatter(theta_k['mu_x'], theta_k['mu_y'], s=theta_k['lambda'] * s_factor, c='yellow', marker='x', alpha=1.0)
    plt.show()


def generate_data(n=30):
    datapoints_a = []
    datapoints_b = []
    datapoints_c = []
    x_cords = []
    y_cords = []
    magnitude = []
    datapoints_lists = [datapoints_a, datapoints_b, datapoints_c]

    true_theta_dict = {
        'a': [14.8, 3.25, 6.5, 4.42, 30],  # mu1, tau1, mu2, tau2, lamda
        'b': [28.3, 5.62, 12.1, 3.59, 10],
        'c': [24.1, 3.24, 46.1, 4.36, 100]
    }

    true_theta = []

    for k in true_theta_dict:
        x = true_theta_dict[k][0]
        tau_x = (1 / true_theta_dict[k][1])**2
        y = true_theta_dict[k][2]
        tau_y = (1 / true_theta_dict[k][3])**2
        lambd = true_theta_dict[k][4]
        pi = 1. / len(true_theta_dict)  # Assuming initializing all pis are equally likely is a good idea :S
        true_theta.append(
            {
                'mu_x': x,
                'tau_x': tau_x,
                'mu_y': y,
                'tau_y': tau_y,
                'lambda': lambd,
                'pi': pi,
            }
        )

    for i in range(n):
        for theta_k, datapoints_list in zip(true_theta_dict.values(), datapoints_lists):
            x_cords.append(random.gauss(theta_k[0], theta_k[1]))
            y_cords.append(random.gauss(theta_k[2], theta_k[3]))
            magnitude.append(np.random.poisson(theta_k[4]))
            datapoints_list.append([x_cords[-1], y_cords[-1], magnitude[-1]])

    for i in range(len(datapoints_a)):
        plt.scatter(datapoints_a[i][0], datapoints_a[i][1], s=datapoints_a[i][2], c='r', alpha=0.5)
        plt.scatter(datapoints_b[i][0], datapoints_b[i][1], s=datapoints_b[i][2], c='b', alpha=0.5)
        plt.scatter(datapoints_c[i][0], datapoints_c[i][1], s=datapoints_c[i][2], c='g', alpha=0.5)
    plt.show()
    display_distributions(x_cords, y_cords, magnitude, true_theta)
    return x_cords, y_cords, magnitude


def generate_initial_values(x_cords, y_cords, magnitude, k=3):
    n_samples = len(x_cords)
    n_classes = k
    theta = []
    for k in range(n_classes):
        x = random.uniform(min(x_cords), max(x_cords))
        tau_x = 1.
        y = random.uniform(min(y_cords), max(y_cords))
        tau_y = 1.
        lambd = random.uniform(min(magnitude), max(magnitude))
        pi = 1. / n_classes  # Assuming initializing all pis are equally likely is a good idea :S
        theta.append(
            {
                'mu_x': x,
                'tau_x': tau_x,
                'mu_y': y,
                'tau_y': tau_y,
                'lambda': lambd,
                'pi': pi,
            }
        )
    return theta

def gmm_em(x_cords, y_cords, magnitude, k=3):
    # Generate initial values
    n_samples = len(x_cords)
    n_classes = k
    theta = generate_initial_values(x_cords, y_cords, magnitude, k=n_classes)
    i = 0
    theta_history = []
    theta_history.append(theta.copy())
    for iteration, old_theta in enumerate(theta_history):
        print(f"iteration: {iteration}")
        for k, theta_k in enumerate(old_theta):
            print(f"theta_{k}: {theta_k}")
    while True:
        new_initial_values = False
        if i % 50 == 0:
            display_distributions(x_cords, y_cords, magnitude, theta)
        if i == 500:
            print("classification failed")
            print(type(theta[0]['mu_x']))
            break
        i += 1
        r = np.zeros((n_classes, n_samples))
        for theta_k, c in zip(theta, range(n_classes)):
            for x_n, y_n, s_n, n in zip(x_cords, y_cords, magnitude, range(n_samples)):
                r[c, n] = calculate_log_r_nk_numerator(x_n,
                                                       y_n,
                                                       s_n,
                                                       theta_k['pi'],
                                                       theta_k['lambda'],
                                                       theta_k['mu_x'],
                                                       theta_k['tau_x'],
                                                       theta_k['mu_y'],
                                                       theta_k['tau_y'],
                                                       )

        r = np.exp(r)
        r = np.divide(r, r.sum(axis=0))
        max_change = 0
        for k in range(n_classes):
            old_values = theta[k].values()
            r_k = r[k].sum()
            pi_k = r_k/ n_samples
            mu_1_k = np.multiply(r[k], np.array(x_cords)).sum() / r_k
            tau_1_k = (r_k / np.multiply(r[k], np.square(np.array(x_cords) - theta[k]['mu_x'])).sum())**2
            mu_2_k = np.multiply(r[k], np.array(y_cords)).sum() / r_k
            tau_2_k = (r_k / np.multiply(r[k], np.square(np.array(y_cords) - theta[k]['mu_y'])).sum())**2
            lamda_k = np.multiply(r[k], np.array(magnitude)).sum() / r_k
            if abs(mu_1_k) > max(x_cords)*100 or abs(mu_2_k) > max(y_cords)*100:
                new_initial_values = True
            else:
                theta[k] = {
                    'mu_x': mu_1_k,
                    'tau_x': tau_1_k,
                    'mu_y': mu_2_k,
                    'tau_y': tau_2_k,
                    'lambda': lamda_k,
                    'pi': pi_k,
                }
                for old_param, new_param in zip(old_values, theta[k].values()):
                    max_change = max(abs(old_param-new_param), max_change)
                    if np.isnan(new_param):
                        new_initial_values = True
        if max_change < 0.001:
            break
        if new_initial_values:
            theta = generate_initial_values(x_cords, y_cords, magnitude, k=n_classes)
        theta_history.append(theta.copy())
    for k, theta_k in enumerate(theta):
        print(f"theta_{k}: {theta_k}")
    display_distributions(x_cords, y_cords, magnitude, theta)
    for iteration, old_theta in enumerate(theta_history):
        print(f"iteration: {iteration}")
        for k, theta_k in enumerate(old_theta):
            print(f"\ttheta_{k}: {theta_k}")

if __name__ == '__main__':
    x_cords, y_cords, magnitude = generate_data(200)
    gmm_em(x_cords, y_cords, magnitude, k=2)