import matplotlib.pyplot as plt

# data
x = [-10, -5, 0, 5, 10, 15, 20]
# CNN, epochs = 200, lr = 0.0001, batch_size = 32
y1 = [0.6419357114932095, 0.3948697806398535, 0.2653386565881674, 0.16621672269013613, 0.12786874896302208, 0.1266478453073014, 0.11196449103168699]

# Transformers(Encoder * 2 + Decoder * 2), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
y2 = [0.03987005725502968, 0.029254980385303497, 0.02019014209508896, 0.018056659027934074, 0.015157107263803482, 0.014463509432971478, 0.013804900459945202]

# draw
plt.plot(x, y1, marker='o', label='CNN')
plt.plot(x, y2, marker='o', label='Transformers')

# add title and axis labels
plt.title('Comparison of NMSE of CNN and Transformers')
plt.xlabel('SNR (dB)')
plt.ylabel('NMSE')

# add legend
plt.legend()

# save the plot
plt.savefig('Comparison of NMSE of CNN and Transformers')

# display the plot
plt.grid(True)
plt.show()
####################################################################################################################
# data
x = [-10, -5, 0, 5, 10, 15, 20]
# CNN, epochs = 200, lr = 0.0001, batch_size = 32
y1 = [12.376030106733767, 13.813554657995969, 14.51362168247246, 15.027294189530807, 15.221216724747757, 15.227348121333408, 15.300884747531022]

# Transformers(Encoder * 2 + Decoder * 2), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
y2 = [15.490602115597438, 15.58706522935487, 15.66893307641061, 15.688134016492416, 15.714188500585596, 15.720413966130947, 15.726322933376025]

# draw
plt.plot(x, y1, marker='o', label='CNN')
plt.plot(x, y2, marker='o', label='Transformers')

# add title and axis labels
plt.title('Comparison of sum rate of CNN and Transformers')
plt.xlabel('SNR (dB)')
plt.ylabel('sum rate(bandwith = 10)')

# add legend
plt.legend()

# save the plot
plt.savefig('Comparison of sum rate of CNN and Transformers')

# display the plot
plt.grid(True)
plt.show()