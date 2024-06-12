import numpy as np
import matplotlib.pyplot as plt

# Given data
models = ['Wired', 'CNN', 'Attention-aided Auto-Encoder', 'Sparse Auto-Encoder', 'Transformers', 'FEDformer', 'Proposed Transformer']
nmse = [0.001, 0.11196449103168699, 0.10238436697803815, 0.08761300950468075, 0.014378390274941921, 0.014243846759200096, 0.013198921456933022]
gbps = [1.00000, 3.47520, 3.59586, 3.61614, 4.09746, 4.09782, 4.10230]  # in Gbps

# Parameters
packet_size = 1024  # Number of bits per packet
num_packets = 1000  # Number of packets to be simulated

def simulate_ber(nmse):
    """Simulate Bit Error Rate (BER) from NMSE."""
    return 0.003 * nmse  # Simplified example relationship

def simulate_packet_loss_rate(ber, packet_size):
    """Simulate Packet Loss Rate from BER."""
    return 1 - (1 - ber)**packet_size

def calculate_payload(sum_rate, packet_loss_rate, num_packets, packet_size):
    """Calculate effective payload based on the sum rate, packet loss rate, number of packets, and packet size."""
    effective_num_packets = num_packets * (1 - packet_loss_rate)
    total_payload = sum_rate * (1 - packet_loss_rate) * packet_size * num_packets  # Correct calculation
    return total_payload

# Initialize results
results = []

# Loop through each set of sum rate and NMSE values
for model, nmse_val, gbps_val in zip(models, nmse, gbps):
    # Convert Gbps to bps for sum_rate
    sum_rate = gbps_val * 1e9  # Convert Gbps to bps
    
    # Simulate BER from NMSE
    ber = simulate_ber(nmse_val)
    
    # Simulate packet loss rate from BER
    plr = simulate_packet_loss_rate(ber, packet_size)
    
    # Calculate effective payload
    payload = calculate_payload(sum_rate, plr, num_packets, packet_size)
    
    # Store results
    results.append({
        'model': model,
        'nmse': nmse_val,
        'sum_rate': sum_rate,
        'ber': ber,
        'packet_loss_rate': plr,
        'payload': payload
    })

# Convert results to numpy array for easy plotting
results_np = np.array([(r['model'], r['nmse'], r['sum_rate'], r['ber'], r['packet_loss_rate'], r['payload']) for r in results],
                      dtype=[('model', 'U50'), ('nmse', float), ('sum_rate', float), ('ber', float), ('packet_loss_rate', float), ('payload', float)])

# Plotting results in four separate figures
plt.figure(figsize=(16, 10))

# Plot NMSE without 'Wired'
nmse_filtered = results_np[results_np['model'] != 'Wired']
plt.subplot(2, 2, 1)
plt.bar([r[0] for r in nmse_filtered], nmse_filtered['nmse'], color='skyblue')
plt.xlabel('Models')
plt.ylabel('NMSE')
plt.title('NMSE by Model')
plt.xticks(rotation=45)
plt.grid(axis='y')

# Plot Packet Loss Rate with 'Wired'
plt.subplot(2, 2, 2)
plt.bar([r[0] for r in results_np], results_np['packet_loss_rate'], color='lightgreen')
plt.xlabel('Models')
plt.ylabel('Packet Loss Rate')
plt.title('Packet Loss Rate by Model')
plt.xticks(rotation=45)
plt.grid(axis='y')

# Plot Sum Rate with 'Wired'
plt.subplot(2, 2, 3)
plt.bar([r[0] for r in results_np], results_np['sum_rate'] / 1e9, color='salmon')  # Convert bps back to Gbps for plotting
plt.xlabel('Models')
plt.ylabel('Sum Rate (Gbps)')
plt.title('Sum Rate by Model')
plt.xticks(rotation=45)
plt.grid(axis='y')

# Plot Effective Payload with 'Wired'
plt.subplot(2, 2, 4)
plt.bar([r[0] for r in results_np], results_np['payload'], color='lightcoral')
plt.xlabel('Models')
plt.ylabel('Effective Payload (bits)')
plt.title('Effective Payload by Model')
plt.xticks(rotation=45)
plt.grid(axis='y')

plt.tight_layout()
plt.show()

# Print results
for res in results:
    print(res)
