import numpy as np
import matplotlib.pyplot as plt

# Given data
SNR_dB_values = [-10, -5, 0, 5, 10, 15, 20]
BER_values = [0.2132280184, 0.1583683188, 0.0786496035, 0.0228784076, 0.0011908930, 0.0000033627, 0.0000000225]
models = ['Wired', 'Wired + WIFI 6 ', 'Wired + CNN', 'Wired + Attention-aided Auto-Encoder', 'Wired + Sparse Auto-Encoder', 'Wired + Transformers', 'Wired + FEDformer', 'Wired + Proposed Transformer']
const_x = 0.00005
# 各SNR值對應的NMSE和Gbps列表
NMSE_values = [
    [1, 1.5419357114932095, 0.6419357114932095, 0.5412381862428629, 0.5238966308505519, 0.040231116116046906, 0.04003031924366951, 0.037573281675577164],
    [1, 1.0948697806398535, 0.3948697806398535, 0.33254079647163465, 0.32426746977979914, 0.030061252415180206, 0.032262492924928665, 0.026736173778772354],
    [1, 0.7653386565881674, 0.2653386565881674, 0.2115293457419186, 0.20119746725457452, 0.02286352589726448, 0.023152129724621773, 0.02016441710293293],
    [1, 0.47596123561662536, 0.17596123561662536, 0.1564142778360277, 0.13996944454347623, 0.018579063937067986, 0.017859801650047302, 0.016315100714564323],
    [1, 0.33396011468502073, 0.13396011468502073, 0.12438449687660279, 0.11280704789072017, 0.016204895451664925, 0.01619112305343151, 0.014203720726072788],
    [1, 0.31912793777809844, 0.11912793777809844, 0.10890364229279535, 0.09475028820637019, 0.015272018499672413, 0.014731141738593578, 0.013647235929965973],
    [1, 0.31196449103168699, 0.11196449103168699, 0.10238436697803815, 0.08761300950468075, 0.014378390274941921, 0.014243846759200096, 0.013198921456933022]
]

Gbps_values = [
    [1, 0.5, 12.376030106733767 * 2 / 10, 12.97931951377382 * 2 / 10, 13.080715119036215 * 2 / 10, 15.487309683281385 * 2 / 10, 15.489140933602904 * 2 / 10, 15.511528572054784 * 2 / 10],
    [1, 0.5, 13.813554657995969 * 2 / 10, 14.154661877931208 * 2 / 10, 14.199338913358442 * 2 / 10, 15.579760882190199 * 2 / 10, 15.55980019276288 * 2 / 10, 15.609860095464892 * 2 / 10],
    [1, 0.5, 14.51362168247246 * 2 / 10, 14.794740768464427 * 2 / 10, 14.848096915580719 * 2 / 10, 15.64483701235907 * 2 / 10, 15.642233280343834 * 2 / 10, 15.669164710042915 * 2 / 10],
    [1, 0.5, 14.97759882683261 * 2 / 10, 15.077112879082799 * 2 / 10, 15.160305529554883 * 2 / 10, 15.68343480874185 * 2 / 10, 15.689904380496367 * 2 / 10, 15.703788696782645 * 2 / 10],
    [1, 0.5, 15.190586800457142 * 2 / 10, 15.238707802839244 * 2 / 10, 15.296675207596744 * 2 / 10, 15.70477877832414 * 2 / 10, 15.70490247599874 * 2 / 10, 15.722745045439746 * 2 / 10],
    [1, 0.5, 15.265055865213462 * 2 / 10, 15.316166877397217 * 2 / 10, 15.386621664883695 * 2 / 10, 15.713156822752882 * 2 / 10, 15.718012085872813 * 2 / 10, 15.727737112861892 * 2 / 10],
    [1, 0.5, 15.300884747531022 * 2 / 10, 15.348662296652947 * 2 / 10, 15.422020653796265 * 2 / 10, 15.721177762372365 * 2 / 10, 15.722385011103668 * 2 / 10, 15.731757566231902 * 2 / 10]
]

# Parameters
packet_size = 1024  # Number of bits per packet
num_packets = 1000  # Number of packets to be simulated

def simulate_ber(nmse, gbps):
    """Simulate Bit Error Rate (BER) from NMSE."""
    wired_patial = 1 / (1 + gbps)
    wireless_patial = 1 - wired_patial
    if nmse == 1:
        return 1e-11  # 為 Wired 的 BER
    else:
        return 1e-11 * wired_patial + nmse * const_x * wireless_patial

def simulate_packet_loss_rate(ber, packet_size):
    """Simulate Packet Loss Rate from BER."""
    return 1 - (1 - ber)**packet_size

def calculate_payload(sum_rate, packet_loss_rate, num_packets, packet_size):
    """Calculate effective payload based on the sum rate, packet loss rate, number of packets, and packet size."""
    effective_num_packets = num_packets * (1 - packet_loss_rate)
    total_payload = sum_rate / packet_size * (1 - packet_loss_rate) * packet_size  # Correct calculation
    return total_payload

# Loop through each SNR value
for i, snr_dB in enumerate(SNR_dB_values):
    # Initialize results
    results = []

    nmse_values_at_snr = NMSE_values[i]
    gbps_values_at_snr = Gbps_values[i]

    # Loop through each set of sum rate and NMSE values
    for model, nmse_val, gbps_val in zip(models, nmse_values_at_snr, gbps_values_at_snr):
        # Convert Gbps to bps for sum_rate
        sum_rate = gbps_val * 1e9  # Convert Gbps to bps

        # Adjust sum rate for models other than 'Wired'
        if model != 'Wired':
            sum_rate += 1e9

        # Simulate BER from NMSE
        ber = simulate_ber(nmse_val, gbps_val)

        # Simulate packet loss rate from BER
        plr = simulate_packet_loss_rate(ber, packet_size)

        # Calculate effective payload
        payload = calculate_payload(sum_rate, plr, num_packets, packet_size)

        # Store results
        results.append({
            'Model': model,
            'BER': ber,
            'Packet Loss Rate': plr,
            'Payload': payload,
            'Sum Rate': sum_rate
        })

    # Plotting the results for this SNR value
    plt.figure(figsize=(16, 10))

    # Plot BER for each model
    plt.subplot(2, 2, 1)
    models_list = [r['Model'] for r in results]
    ber_values = [r['BER'] for r in results]
    plt.bar(models_list, ber_values, color='b', alpha=0.7)
    plt.ylabel('BER')
    plt.title(f'BER vs Models at SNR = {snr_dB} dB')
    plt.xticks(rotation=45, ha='right')

    # Plot Packet Loss Rate for each model
    plt.subplot(2, 2, 2)
    plr_values = [r['Packet Loss Rate'] for r in results]
    plt.bar(models_list, plr_values, color='g', alpha=0.7)
    plt.ylabel('Packet Loss Rate')
    plt.title(f'Packet Loss Rate vs Models at SNR = {snr_dB} dB')
    plt.xticks(rotation=45, ha='right')

    # Plot Payload for each model
    plt.subplot(2, 2, 3)
    payload_values = [r['Payload'] for r in results]
    plt.bar(models_list, payload_values, color='r', alpha=0.7)
    plt.ylabel('Payload (bits)')
    plt.title(f'Payload vs Models at SNR = {snr_dB} dB')
    plt.xticks(rotation=45, ha='right')

    # Plot Sum Rate for each model
    plt.subplot(2, 2, 4)
    sum_rate_values = [r['Sum Rate'] for r in results]
    plt.bar(models_list, sum_rate_values, color='m', alpha=0.7)
    plt.ylabel('Sum Rate (bps)')
    plt.title(f'Sum Rate vs Models at SNR = {snr_dB} dB')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(f'performance_at_SNR_{snr_dB}_dB.png')  # Save the figure
    plt.show()
    for result in results:
        print(result)
    print("")
