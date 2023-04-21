from nolitsa import data, lyapunov
import matplotlib.pyplot as plt
import time


if __name__ == "__main__":
    sample = 0.01
    x0 = [0.62225717, -0.08232857, 30.60845379]
    x = data.lorenz(length=10000, sample=sample, x0=x0,
                    sigma=16.0, beta=4.0, rho=45.92)[1][:, 0]

    # Choose appropriate Theiler window.
    window = 60

    # Time delay.
    tau = 13

    # Embedding dimension.
    dim = [5]

    # experiments by data_length
    res = []
    res_fast = []
    data_len = [1000*i for i in range(1, 10+1)]

    for l in data_len:
        print("l: {}".format(l))
        start = time.time()
        _ = lyapunov.mle_embed(x[:l], dim=dim, tau=tau, maxt=300, window=window)[0]
        end = time.time()
        res.append(end - start)

        start = time.time()
        _ = lyapunov.fast_euclidean_mle_embed(x[:l], dim=dim, tau=tau, maxt=300, window=window)[0]
        end = time.time()
        res_fast.append(end - start)

    plt.figure()
    plt.title("Experiment by data length")
    plt.xlabel("Data length")
    plt.ylabel("Calculation time (sec)")
    plt.plot(data_len, res, label="original")
    plt.plot(data_len, res_fast, label="fast")
    plt.legend()
    plt.show()

    # experiments by maxt
    res = []
    res_fast = []
    max_t = [300*i for i in range(1, 10 + 1)]

    for t in max_t:
        print("max_t: {}".format(t))
        start = time.time()
        _ = lyapunov.mle_embed(x[:5000], dim=dim, tau=tau, maxt=t, window=window)[0]
        end = time.time()
        res.append(end - start)

        start = time.time()
        _ = lyapunov.fast_euclidean_mle_embed(x[:5000], dim=dim, tau=tau, maxt=t, window=window)[0]
        end = time.time()
        res_fast.append(end - start)

    plt.figure()
    plt.title("Experiment by maxt")
    plt.xlabel("maxt")
    plt.ylabel("Calculation time (sec)")
    plt.plot(max_t, res, label="original")
    plt.plot(max_t, res_fast, label="fast")
    plt.legend()
    plt.show()


    # experiments by embedding dimension
    res = []
    res_fast = []
    embed_dim = [[2*i] for i in range(1, 5+1)]

    for dim in embed_dim:
        print("dim: {}".format(dim))
        start = time.time()
        _ = lyapunov.mle_embed(x[:5000], dim=dim, tau=tau, maxt=300, window=window)[0]
        end = time.time()
        res.append(end - start)

        start = time.time()
        _ = lyapunov.fast_euclidean_mle_embed(x[:5000], dim=dim, tau=tau, maxt=300, window=window)[0]
        end = time.time()
        res_fast.append(end - start)

    plt.figure()
    plt.title("Experiment by embedding dimension")
    plt.xlabel("Embedding dimension")
    plt.ylabel("Calculation time (sec)")
    plt.plot(embed_dim, res, label="original")
    plt.plot(embed_dim, res_fast, label="fast")
    plt.legend()
    plt.show()
