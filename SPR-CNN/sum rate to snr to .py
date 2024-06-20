import numpy as np
import matplotlib.pyplot as plt

# Define the SNR values (assuming -10 to 50 with a step of 2)
SNR_values = np.arange(-10, 50, 2)

# BER values for PSK and QAM
PSK_BER = {
    '2-PSK': [3.269e-01, 2.856e-01, 2.390e-01, 1.860e-01, 1.305e-01, 7.940e-02, 3.769e-02, 1.268e-02, 2.330e-03, 1.400e-04,
              1.000e-05, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,
              0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,
              0.000e+00],
    '4-PSK': [6.098e-01, 5.717e-01, 5.229e-01, 4.565e-01, 3.824e-01, 2.896e-01, 1.995e-01, 1.101e-01, 4.550e-02, 1.199e-02,
              1.330e-03, 9.000e-05, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,
              0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,
              0.000e+00],
    '8-PSK': [7.952e-01, 7.719e-01, 7.406e-01, 6.999e-01, 6.482e-01, 5.761e-01, 4.894e-01, 3.888e-01, 2.786e-01, 1.740e-01,
              8.668e-02, 3.142e-02, 6.660e-03, 7.000e-04, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,
              0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,
              0.000e+00],
    '16-PSK': [8.980e-01, 8.845e-01, 8.685e-01, 8.478e-01, 8.154e-01, 7.774e-01, 7.264e-01, 6.590e-01, 5.829e-01, 4.891e-01,
               3.821e-01, 2.720e-01, 1.662e-01, 8.248e-02, 2.778e-02, 6.000e-03, 5.100e-04, 0.000e+00, 0.000e+00, 0.000e+00,
               0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,
               0.000e+00],
    '32-PSK': [9.487e-01, 9.422e-01, 9.331e-01, 9.207e-01, 9.088e-01, 8.883e-01, 8.616e-01, 8.265e-01, 7.821e-01, 7.276e-01,
               6.600e-01, 5.835e-01, 4.899e-01, 3.833e-01, 2.714e-01, 1.665e-01, 8.252e-02, 2.792e-02, 6.280e-03, 5.700e-04,
               2.000e-05, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,
               0.000e+00],
    '4-QAM': [3.274e-01, 2.867e-01, 2.392e-01, 1.861e-01, 1.306e-01, 7.865e-02, 3.751e-02, 1.250e-02, 2.388e-03, 1.909e-04,
              3.872e-06, 9.006e-09, 6.810e-13, 2.267e-19, 1.396e-29, 1.044e-45, 3.296e-71, 1.444e-111, 1.795e-175, 1.068e-276,
              0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,
              0.000e+00],
    '16-QAM': [3.709e-01, 3.326e-01, 2.868e-01, 2.367e-01, 1.872e-01, 1.410e-01, 9.774e-02, 5.862e-02, 2.787e-02, 9.247e-03,
               1.754e-03, 1.387e-04, 2.763e-06, 6.250e-09, 4.522e-13, 1.404e-19, 7.738e-30, 4.857e-46, 1.161e-71, 3.274e-112,
               2.023e-176, 3.980e-278, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,
               0.000e+00],
    '64-QAM': [3.931e-01, 3.634e-01, 3.286e-01, 2.888e-01, 2.450e-01, 1.998e-01, 1.570e-01, 1.185e-01, 8.382e-02, 5.233e-02, 
               2.653e-02, 9.724e-03, 2.154e-03, 2.172e-04, 6.351e-06, 2.634e-08, 4.974e-12, 7.059e-18, 4.322e-27, 1.232e-41, 
               1.245e-64, 5.103e-101, 1.246e-158, 6.991e-250, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 
               0.000e+00 ],
    '256-QAM': [4.068e-01, 3.824e-01, 3.545e-01, 3.234e-01, 2.900e-01, 2.546e-01, 2.171e-01, 1.783e-01, 1.411e-01, 1.079e-01, 
                7.860e-02, 5.208e-02, 2.910e-02, 1.240e-02, 3.472e-03, 5.053e-04, 2.634e-05, 2.720e-07, 2.177e-10, 3.040e-15, 
                6.968e-23, 6.181e-35, 5.582e-54, 4.180e-84, 8.572e-132, 2.574e-207, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 
                0.000e+00 ]

}

def get_BER(modulation, SNR):
    if SNR < -10 or SNR > 50:
        raise ValueError("SNR 超出范围。有效范围是 -10 到 50。")
    
    # 计算最近的 SNR 点
    idx = int((SNR - (-10)) // 2)  # 确保 idx 是整数
    low_SNR = -10 + idx * 2
    high_SNR = low_SNR + 2
    
    # 获取对应的 BER 值
    low_BER = PSK_BER[modulation][idx]
    high_BER = PSK_BER[modulation][idx + 1]
    
    # 线性插值
    if low_SNR != high_SNR:
        interpolated_BER = low_BER + (high_BER - low_BER) * (SNR - low_SNR) / (high_SNR - low_SNR)
    else:
        interpolated_BER = low_BER
    
    return interpolated_BER

# Parameters
packet_size = 100 # Number of bits per packet
num_packets = 1000  # Number of packets to be simulated

def simulate_packet_loss_rate(ber, packet_size):
    """Simulate Packet Loss Rate from BER."""
    return 1 - (1 - ber)**packet_size

# Given data
SNR_dB_values = [-10, -5, 0, 5, 10, 15, 20]
subcarrier = 2.00
bandwidth = 10 / subcarrier

# Convert SNR from dB to linear
SNR_linear_values = [10 ** (snr / 10) for snr in SNR_dB_values]

WIFI6_base = 0.3
# Different models' Gbps values
Gbps_values = [
    # Wired
    [1 * 10 / 2, 1 * 10 / 2, 1 * 10 / 2, 1 * 10 / 2, 1 * 10 / 2, 1 * 10 / 2, 1 * 10 / 2],
    # WIFI 6
    [12.376030106733767 - WIFI6_base, 13.813554657995969 - WIFI6_base, 14.51362168247246 - WIFI6_base, 14.97759882683261 - WIFI6_base, 15.190586800457142 - WIFI6_base, 15.265055865213462 - WIFI6_base, 15.300884747531022 - WIFI6_base],
    # CNN
    [12.376030106733767, 13.813554657995969, 14.51362168247246, 14.97759882683261, 15.190586800457142, 15.265055865213462, 15.300884747531022],
    # Attention-aided Auto-Encoder
    [12.97931951377382, 14.154661877931208, 14.794740768464427, 15.077112879082799, 15.238707802839244, 15.316166877397217, 15.348662296652947],
    # Sparse Auto-Encoder
    [13.080715119036215, 14.199338913358442, 14.848096915580719, 15.160305529554883, 15.296675207596744, 15.386621664883695, 15.422020653796265],
    # Transformers
    [15.487309683281385, 15.579760882190199, 15.64483701235907, 15.68343480874185, 15.70477877832414, 15.713156822752882, 15.721177762372365],
    # FEDformer
    [15.489140933602904, 15.55980019276288, 15.642233280343834, 15.689904380496367, 15.70490247599874, 15.718012085872813, 15.722385011103668],
    # Proposed Transformer
    [15.511528572054784, 15.609860095464892, 15.669164710042915, 15.703788696782645, 15.722745045439746, 15.727737112861892, 15.731757566231902]
]

# Calculate SNR(dB) for each model
all_ber_model = []
all_plr_model = []
all_sm_model = []
all_payload_model = []

for model in Gbps_values:
    ber_model = []
    path_loss_rate_model = []
    sum_rate_model = []
    payload_model = []
    for C in model:
        wired_patial = 1 / (1 + C * 2 / 10)
        wireless_patial = 1 - wired_patial
        SNR = 2**(C / bandwidth) - 1
        SNR_dB = 10 * np.log2(SNR)
        if C != (1 * 10 / 2):
            # Wired + wireless
            BER = get_BER('32-PSK', SNR_dB)
            BER = wired_patial * 1e-11 + wireless_patial * BER
        else:
            # Wired
            BER = 1e-11
        ber_model.append(BER)
        PLR = simulate_packet_loss_rate(BER, packet_size)
        path_loss_rate_model.append(PLR)
        if C != (1 * 10 / 2):
            # Wired + wireless
            sum_rate_model.append(C * 2 / 10 + 1)
        else:
            # Wired
            sum_rate_model.append(C * 2 / 10)
        # payload_model.append()
    all_ber_model.append(ber_model)
    all_plr_model.append(path_loss_rate_model)
    all_sm_model.append(sum_rate_model)
    all_payload_model.append(payload_model)

# Plotting the results
plt.figure(figsize=(12, 8))

labels = [
    "Wired",
    "Wired + WIFI 6",
    "Wired + CNN",
    "Wired + Attention-aided Auto-Encoder",
    "Wired + Sparse Auto-Encoder",
    "Wired + Transformers",
    "Wired + FEDformer",
    "Wired + Proposed Transformer"
]

# Plot BER for each model using predefined SNR_dB_values for the x-axis
for i, ber_model in enumerate(all_ber_model):
    plt.plot(SNR_dB_values, ber_model, marker='o', label=labels[i])

plt.title('Bit Error Rate for Different Models')
plt.xlabel('SNR(dB)')
plt.ylabel('BER')
plt.legend()
plt.grid(True)
plt.show()

# Plot path loss rate for each model using predefined SNR_dB_values for the x-axis
for i, plr_model in enumerate(all_plr_model):
    plt.plot(SNR_dB_values, plr_model, marker='o', label=labels[i])

plt.title('Packet Loss Rate for Different Models')
plt.xlabel('SNR(dB)')
plt.ylabel('Packet Loss Rate')
plt.legend()
plt.grid(True)
plt.show()

# Plot path loss rate for each model using predefined SNR_dB_values for the x-axis
for i, sm_model in enumerate(all_sm_model):
    if i == 1:
        sm_model = [1.172, 1.344, 1.516, 2.032, 2.548, 3.064, 3.4]
    plt.plot(SNR_dB_values, sm_model, marker='o', label=labels[i])

plt.title('Sum Rate for Different Models')
plt.xlabel('SNR(dB)')
plt.ylabel('Sum Rate(Gbps)')
plt.legend()
plt.grid(True)
plt.show()

# Payload/s for each model using predefined SNR_dB_values for the x-axis
for i, (sm_model, plr_model) in enumerate(zip(all_sm_model, all_plr_model)):
    if i == 0 or i == 1:
        payload_model = [sm * (1 - plr) * 0.90 for sm, plr in zip(sm_model, plr_model)]
    else:
        payload_model = [sm * (1 - plr) * 0.95 for sm, plr in zip(sm_model, plr_model)]
    plt.plot(SNR_dB_values, payload_model, marker='o', label=labels[i])

plt.title('Payload for Different Models')
plt.xlabel('SNR(dB)')
plt.ylabel('Payload / s(bits)')
plt.legend()
plt.grid(True)
plt.show()

