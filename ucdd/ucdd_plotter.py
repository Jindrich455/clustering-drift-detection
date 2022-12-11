from matplotlib import pyplot as plt


def plot_predicted(df_ref_plus, df_ref_minus, df_test_plus, df_test_minus):
    c_ref = 'g'
    c_test = 'r'
    marker_plus = ','
    marker_minus = 'v'
    plt.scatter(df_ref_plus.iloc[:, 0], df_ref_plus.iloc[:, 1], c=c_ref, marker=marker_plus)
    plt.scatter(df_ref_minus.iloc[:, 0], df_ref_minus.iloc[:, 1], c=c_ref, marker=marker_minus)
    plt.scatter(df_test_plus.iloc[:, 0], df_test_plus.iloc[:, 1], c=c_test, marker=marker_plus)
    plt.scatter(df_test_minus.iloc[:, 0], df_test_minus.iloc[:, 1], c=c_test, marker=marker_minus)

    plt.title("Predicted labels")
    plt.show()


def plot_u_w0_w1(df_u, w0, w1):
    # U and W0 always come from the reference window, W1 is always from detection

    print('plotting u w0 w1')

    c_ref = 'g'
    c_test = 'r'
    marker_plus = ','
    marker_minus = 'v'

    plt.scatter(df_u.iloc[:, 0], df_u.iloc[:, 1], c=c_ref, marker=marker_plus)
    plt.scatter(w0.iloc[:, 0], w0.iloc[:, 1], c=c_ref, marker=marker_minus)
    plt.scatter(w1.iloc[:, 0], w1.iloc[:, 1], c=c_test, marker=marker_minus)

    plt.title("Nearest neighbours found")
    plt.show()
