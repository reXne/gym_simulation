import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss
import time
import hydra
from omegaconf import DictConfig

def kl_divergence(p, q, epsilon=1e-8):
    p += epsilon
    q += epsilon
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def js_divergence(p, q, epsilon=1e-8):
    p += epsilon
    q += epsilon
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, epsilon) + 0.5 * kl_divergence(q, m, epsilon)

def evaluate_additional_metrics(real_data, simulated_data):
    num_variables = real_data.shape[1]
    kl_values = []
    js_values = []
    ks_p_values = []

    for i in range(num_variables):
        p = np.histogram(real_data[:, i], bins=50, density=True)[0]
        q = np.histogram(simulated_data[:, i], bins=50, density=True)[0]
        kl_values.append(kl_divergence(p, q))
        js_values.append(js_divergence(p, q))
        ks_stat, ks_p_value = stats.ks_2samp(real_data[:, i], simulated_data[:, i])
        ks_p_values.append(ks_p_value)

    t_stat, p_value = stats.ttest_1samp(kl_values, popmean=0)
    print(f"KL Divergence - T-statistic: {t_stat}, p-value: {p_value}")

    t_stat, p_value = stats.ttest_1samp(js_values, popmean=0)
    print(f"JS Divergence - T-statistic: {t_stat}, p-value: {p_value}")

    t_stat, p_value = stats.ttest_1samp(ks_p_values, popmean=0.5)
    print(f"Kolmogorov-Smirnov p-values - T-statistic: {t_stat}, p-value: {p_value}")

def evaluate_simulation(real_tuples, generated_tuples):
    S_real, A_real, R_real, Sp_real, D_real, eligibility_matrix_real = real_tuples
    S_gen, A_gen, R_gen, Sp_gen, D_gen = generated_tuples

    # Evaluate states (Sp) using Mean Squared Error (MSE) and Mean Absolute Error (MAE)
    mse_sp = mean_squared_error(Sp_real, Sp_gen)
    mae_sp = mean_absolute_error(Sp_real, Sp_gen)
    
    # Evaluate rewards (R) using Mean Squared Error (MSE) and Mean Absolute Error (MAE)
    mse_r = mean_squared_error(R_real, R_gen)
    mae_r = mean_absolute_error(R_real, R_gen)

    # Evaluate done (D) using Binary Cross-Entropy (BCE)
    bce_d = log_loss(D_real, D_gen)

    print(f"State MSE: {mse_sp:.4f}")
    print(f"State MAE: {mae_sp:.4f}")
    print(f"Reward MSE: {mse_r:.4f}")
    print(f"Reward MAE: {mae_r:.4f}")
    print(f"Done BCE: {bce_d:.4f}")

    # Evaluate additional metrics (KL divergence, JS divergence, and Kolmogorov-Smirnov test)
    evaluate_additional_metrics(Sp_real, Sp_gen)

@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    start_time = time.time()
    # Load real and generated tuples
    real_tuples = pd.read_pickle(cfg.data.files.tuples_train_resampled_file)
    generated_tuples = pd.read_pickle('./data/40_simulator/simulated_tuples.pkl')

    # Evaluate the performance of the simulation
    evaluate_simulation(real_tuples, generated_tuples)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
