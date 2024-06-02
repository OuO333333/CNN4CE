import math
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

fontsize = 25

def calculate_snr(n):
    if n < 0:
        raise ValueError("The input n must be a non-negative integer.")
    p_signal = 150
    p_noise = 12
    snr = p_signal / (p_noise + 0.01 * n)
    return snr

def calculate_snr_db(snr):
    snr_db = 10 * math.log10(snr)
    return snr_db

def estimate_NMSE(x_input):
    # 创建插值函数
    x = [-10, -5, 0, 5, 10, 15, 20]
    y = [0.039645493030548096, 0.02848387509584427, 0.02177749201655388, 0.017665809020400047, 0.015258733183145523, 0.01451767235994339, 0.013874311000108719]
    f = interp1d(x, y, kind='linear', fill_value="extrapolate")
    # 对输入的 x 值进行插值估计 y 值
    y_estimate = f(x_input)
    return y_estimate

def estimate_sum_rate(x_input):
    # 创建插值函数
    x = [-10, -5, 0, 5, 10, 15, 20]
    y = [15.49264954985823, 15.594047444695512, 15.65463062254004, 15.691648785427166, 15.713276056208183, 15.719927910290641, 15.725700256494086]
    f = interp1d(x, y, kind='linear', fill_value="extrapolate")
    # 对输入的 x 值进行插值估计 y 值
    y_estimate = f(x_input)
    return y_estimate

# 用于存储 n 和 NMSE 的列表
n_values = range(17, 600)
nmse_values = []
sum_rate_values = []

# 计算每个 n 对应的 NMSE
for n in n_values:
    snr = calculate_snr(n - 16)
    snr_db = calculate_snr_db(snr)
    print("SNR(db) = ", snr_db)
    nmse = estimate_NMSE(snr_db)
    print("NMSE = ", nmse)
    sum_rate = estimate_sum_rate(snr_db)
    print("sum_rate = ", sum_rate)
    nmse_values.append(nmse)
    sum_rate_values.append(sum_rate)

# 绘制 n 和 NMSE 的关系图
plt.plot(n_values, nmse_values, marker='o')
plt.xlabel('Server numbers', fontsize=fontsize)
plt.ylabel('NMSE', fontsize=fontsize)
plt.title('Comparison of NMSE of different server numbers', fontsize=fontsize)
plt.grid(True)
plt.show()

# 绘制 n 和 Sum Rate 的关系图
plt.plot(n_values, sum_rate_values, marker='o')
plt.xlabel('Server numbers', fontsize=fontsize)
plt.ylabel('Sum Rate', fontsize=fontsize)
plt.title('Comparison of Sum Rate of different server numbers', fontsize=fontsize)
plt.grid(True)
plt.show()