import matplotlib.pyplot as plt

# data
x = [-10, -5, 0, 5, 10, 15, 20]
# CNN, epochs = 200, lr = 0.0001, batch_size = 32
y1 = [0.6419357114932095, 0.3948697806398535, 0.2653386565881674, 0.16621672269013613, 0.12786874896302208, 0.1266478453073014, 0.11196449103168699]

# Transformers(Encoder * 9), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
y2 = [0.039898015558719635, 0.039575472474098206, 0.03917403519153595, 0.01579899713397026, 0.015518935397267342, 0.015378423035144806, 0.013160046190023422]

# Transformers(Encoder * 2 + Decoder * 2), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
y3 = [0.03957320749759674, 0.0313139446079731, 0.018902089446783066, 0.015466086566448212, 0.013048394583165646, 0.012472232803702354, 0.01183935534209013]

# draw
plt.plot(x, y1, marker='o', label='CNN')
# plt.plot(x, y2, marker='o', label='Transformers(Encoder)')
plt.plot(x, y3, marker='o', label='Transformers')

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

# Transformers(Encoder * 9), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
y2 = [15.490347333089433, 15.493287869347391, 15.496947047258363, 15.708424724468355, 15.710939771297618, 15.712201425164704, 15.732106112054788]

# Transformers(Encoder * 2 + Decoder * 2), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
y3 = [15.493308527872895, 15.568404974771003, 15.680528308855868, 15.711414292671664, 15.733107185589379, 15.73827208055814, 15.74394318113315]

# draw
plt.plot(x, y1, marker='o', label='CNN')
# plt.plot(x, y2, marker='o', label='Transformers(Encoder)')
plt.plot(x, y3, marker='o', label='Transformers')

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