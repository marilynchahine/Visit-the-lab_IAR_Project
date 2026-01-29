
import numpy as np
import os
import time

def plot_rewards_ma(results, switch_trial, out_pdf, window=200):
    import matplotlib.pyplot as plt

    def moving_average(x, w):
        return np.convolve(x, np.ones(w) / w, mode="valid")

    plt.figure(dpi=300)
    plt.grid(linestyle="--")

    for agent_name, runs in results.items():
        runs = np.asarray(runs, dtype=float)   # (n_runs, trials)
        mean = runs.mean(axis=0)
        std  = runs.std(axis=0)

        mean_s = moving_average(mean, window)
        std_s  = moving_average(std, window)

        x = np.arange(len(mean_s)) + (window - 1)

        plt.plot(x, mean_s, label=agent_name)
        plt.fill_between(
            x,
            mean_s - std_s,
            mean_s + std_s,
            alpha=0.25
        )

    plt.axvline(switch_trial, linestyle="--", color="k", label="Human change")
    plt.xlabel("Trial")
    plt.ylabel("Reward (mean ± std, moving average)")
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()


def main():
    # npy_path = r"../data/non_stationary_train_test_1769717581.6607442.npy"
    npy_path = r"../data/non_stationary_train_test_1769720582.2531784.npy"

    train_trials = 300000
    test_trials  = 300000
    switch_train = 130000
    switch_test  = 130000

    results = np.load(npy_path, allow_pickle=True).item()

    # results[agent] = list of full_curve
    full = {a: np.asarray(curves) for a, curves in results.items()}

    
    train_results = {
        a: curves[:, :train_trials]
        for a, curves in full.items()
    }

    test_results = {
        a: curves[:, train_trials:train_trials + test_trials]
        for a, curves in full.items()
    }

    
    ts = str(time.time()).replace(".", "_")
    out_train = f"../data/non_stationary_train_{ts}.pdf"
    out_test  = f"../data/non_stationary_test_{ts}.pdf"

    
    plot_rewards_ma(
        train_results,
        switch_trial=switch_train,
        out_pdf=out_train,
        window=200
    )

    plot_rewards_ma(
        test_results,
        switch_trial=switch_test,
        out_pdf=out_test,
        window=200
    )

    print("Saved train figure:", out_train)
    print("Saved test figure:", out_test)


if __name__ == "__main__":
    main()
