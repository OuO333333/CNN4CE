import random

def calculate_data_center_delays(packet_size_bytes, transmission_rate_gbps, distance_meters, propagation_speed_mps):
    # 轉換單位
    packet_size_bits = packet_size_bytes * 8  # 將字節轉換為位元
    transmission_rate_bps = transmission_rate_gbps * (10**9)  # 將Gbps轉換為bps

    # 計算傳輸延遲
    transmission_delay = packet_size_bits / transmission_rate_bps

    # 計算傳播延遲
    propagation_delay = distance_meters / propagation_speed_mps

    return transmission_delay + propagation_delay

def simulate_data_transfer(total_data_tb, sum_rate_gbps, packet_loss_rate, packet_size_bytes, protocol, queue_count, queueing_delay, packet_delay):
    # 轉換單位
    total_data_bits = total_data_tb * (10**12) * 8
    sum_rate_bps = sum_rate_gbps * (10**9) * 8
    packet_size_bits = packet_size_bytes * 8
    
    # 計算傳輸一個封包所需的時間（秒）
    time_per_packet = packet_size_bits / sum_rate_bps + packet_delay
    
    # 初始化變數
    total_packets = total_data_bits / packet_size_bits
    successful_packets = 0
    total_time = 0
    queue = [0] * queue_count

    while successful_packets < total_packets:
        # 處理每個隊列中的封包
        for i in range(queue_count):

            if protocol == "UDP":
                # UDP: 封包可能遺失
                if random.random() > packet_loss_rate:
                    successful_packets += 1
                else:
                    successful_packets += 1
            elif protocol == "TCP":
                # TCP: 封包遺失會重傳
                if random.random() > packet_loss_rate:
                    successful_packets += 1

        total_time += (time_per_packet * queue_count + queueing_delay)

    return total_time

# for simulate_data_transfer
total_data_tb = 1  # 傳輸資料大小，單位 TB
sum_rate_gbps = 10  # 總傳輸速率，單位 Gbps
packet_loss_rate = 0.01  # 封包遺失率
packet_size_bytes = 1500  # 封包大小，單位 Bytes
protocol = "UDP"  # 傳輸協議 ("TCP" 或 "UDP")
queue_count = 1000  # 節點中的隊列數量
queueing_delay = 0.000000001

# for calculate_data_center_delays(packet_delay)
distance_meters = 100  # 傳輸距離，單位米
propagation_speed_mps = 2 * (10**8)  # 傳播速度，單位米/秒

# 模擬資料傳輸
packet_delay = calculate_data_center_delays(packet_size_bytes, sum_rate_gbps, distance_meters, propagation_speed_mps)
total_time = simulate_data_transfer(total_data_tb, sum_rate_gbps, packet_loss_rate, packet_size_bytes, protocol, queue_count, queueing_delay, packet_delay)
print(f"Total transfer time: {total_time} seconds")
