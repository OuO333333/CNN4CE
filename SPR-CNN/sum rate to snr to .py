import numpy as np
import matplotlib.pyplot as plt

# Define the SNR values (assuming -10 to 50 with a step of 2)
SNR_values = np.arange(-10, 52, 2)

# BER values for PSK and QAM
PSK_BER = {
    '2-PSK': [3.269e-01 2.856e-01 2.390e-01 1.860e-01 1.305e-01 7.940e-02 3.769e-02 1.268e-02 2.330e-03 1.400e-04,
              1.000e-05 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00,
              0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00,
              0.000e+00 ],
    '4-PSK': [6.098e-01 5.717e-01 5.229e-01 4.565e-01 3.824e-01 2.896e-01 1.995e-01 1.101e-01 4.550e-02 1.199e-02,
              1.330e-03 9.000e-05 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00,
              0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00,
              0.000e+00 ],
    '8-PSK': [7.952e-01 7.719e-01 7.406e-01 6.999e-01 6.482e-01 5.761e-01 4.894e-01 3.888e-01 2.786e-01 1.740e-01 8.668e-02 3.142e-02 6.660e-03 7.000e-04 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 ],
    '16-PSK': [8.975e-01, 8.847e-01, 8.697e-01, 8.460e-01, 8.168e-01, 7.734e-01, 7.253e-01, 6.622e-01, 5.833e-01, 4.876e-01, 3.852e-01, 2.737e-01],
    '32-PSK': [9.492e-01, 9.420e-01, 9.335e-01, 9.222e-01, 9.067e-01, 8.855e-01, 8.591e-01, 8.266e-01, 7.837e-01, 7.295e-01, 6.609e-01, 5.808e-01],
    '4-QAM': [3.274e-01, 2.867e-01, 2.392e-01, 1.861e-01, 1.306e-01, 7.865e-02, 3.751e-02, 1.250e-02, 2.388e-03, 1.909e-04, 3.872e-06, 9.006e-09],
    '16-QAM': [3.709e-01, 3.326e-01, 2.868e-01, 2.367e-01, 1.872e-01, 1.410e-01, 9.774e-02, 5.862e-02, 2.787e-02, 9.247e-03, 1.754e-03, 1.387e-04],
    '64-QAM': [3.931e-01, 3.634e-01, 3.286e-01, 2.888e-01, 2.450e-01, 1.998e-01, 1.570e-01, 1.185e-01, 8.382e-02, 5.233e-02, 2.653e-02, 9.724e-03],
    '256-QAM': [4.068e-01, 3.824e-01, 3.545e-01, 3.234e-01, 2.900e-01, 2.546e-01, 2.171e-01, 1.783e-01, 1.411e-01, 1.079e-01, 7.860e-02, 5.208e-02]
}

def get_BER(modulation, SNR):
    if SNR < -10 or SNR > 50:
        raise ValueError("SNR out of range. Valid range is -10 to 50.")
    
    # Determine the closest SNR points
    idx = (SNR - (-10)) // 2
    low_SNR = -10 + idx * 2
    high_SNR = low_SNR + 2
    
    # Interpolate BER values based on the SNR points
    low_BER = PSK_BER[modulation][idx]
    high_BER = PSK_BER[modulation][idx + 1]
    
    # Linear interpolation
    if low_SNR != high_SNR:
        interpolated_BER = low_BER + (high_BER - low_BER) * (SNR - low_SNR) / (high_SNR - low_SNR)
    else:
        interpolated_BER = low_BER
    
    return interpolated_BER

# Example usage:
modulation = '16-QAM'
input_SNR = 20  # Example SNR value
resulting_BER = get_BER(modulation, input_SNR)
print(f"For SNR = {input_SNR} dB and {modulation}, interpolated BER = {resulting_BER:.2e}")

# Given data
SNR_dB_values = [-10, -5, 0, 5, 10, 15, 20]
subcarrier = 2
bandwidth = 10 / subcarrier

# Convert SNR from dB to linear
SNR_linear_values = [10 ** (snr / 10) for snr in SNR_dB_values]

# Different models' Gbps values
Gbps_values = [
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
snr_db_values = []

for model in Gbps_values:
    snr_db_model = []
    for C in model:
        SNR = 2**(C / bandwidth) - 1
        SNR_dB = 10 * np.log2(SNR)
        snr_db_model.append(SNR_dB)
    snr_db_values.append(snr_db_model)

# Plotting the results
plt.figure(figsize=(12, 8))

labels = [
    "CNN",
    "Attention-aided Auto-Encoder",
    "Sparse Auto-Encoder",
    "Transformers",
    "FEDformer",
    "Proposed Transformer"
]

for i, snr_db_model in enumerate(snr_db_values):
    plt.plot(snr_db_model, marker='o', label=labels[i])

plt.title('SNR(dB) for Different Models')
plt.xlabel('Data Point')
plt.ylabel('SNR (dB)')
plt.legend()
plt.grid(True)
plt.show()
