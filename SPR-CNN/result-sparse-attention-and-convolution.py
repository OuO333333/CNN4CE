import matplotlib.pyplot as plt

# data
x = [-10, -5, 0, 5, 10, 15, 20]

# Transformers(Encoder * 2 + Decoder * 2), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
# Full Self Attention
y1 = [0.03965744748711586, 0.028810562565922737, 0.02021711692214012, 0.015879059210419655, 0.01424616202712059, 0.013560600578784943, 0.013357477262616158]

# Transformers(Encoder * 2 + Decoder * 2), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
# Atrous Self Attention
y2 = [0.03856229782104492, 0.027260668575763702, 0.020888376981019974, 0.019930638372898102, 0.013781104236841202, 0.013130326755344868, 0.012763812206685543]

# Not yet!
# Transformers(Encoder * 2 + Decoder * 2), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
# Local Self Attention
y3 = [0.03928970918059349, 0.028576577082276344, 0.02001512609422207, 0.015685435384511948, 0.013992800377309322, 0.013519784435629845, 0.013477599248290062]

# Not yet!
# Transformers(Encoder * 2 + Decoder * 2), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
# Stride Sparse Self Attention(2, 2), no(6, 2), (4, 2)because is worse
y4 = [0.039743758738040924, 0.027776280418038368, 0.019192421808838844, 0.014876648783683777, 0.012890099547803402, 0.012358461506664753, 0.012501830235123634]

# draw
plt.plot(x, y1, marker='o', label='Full Self Attention')
plt.plot(x, y2, marker='o', label='Atrous Self Attention')
plt.plot(x, y3, marker='o', label='Local Self Attention')
plt.plot(x, y4, marker='o', label='Stride Sparse Self Attention')

# set y-axis to log scale
plt.yscale('log')

# add title and axis labels
plt.title('Comparison of NMSE of different Sparse Attention')
plt.xlabel('SNR (dB)')
plt.ylabel('NMSE (log scale)')

# add legend
plt.legend()

# save the plot
plt.savefig('Comparison of NMSE of different Sparse Attention')

# display the plot
plt.grid(True)
plt.show()
####################################################################################################################
# data
x = [-10, -5, 0, 5, 10, 15, 20]

# Transformers(Encoder * 2 + Decoder * 2), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
# Full Self Attention
y1 = [15.492540660394557, 15.59108972424448, 15.668690136608705, 15.707705615158078, 15.722364224883243, 15.72851409896477, 15.730335747297762]

# Transformers(Encoder * 2 + Decoder * 2), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
# Atrous Self Attention
y2 = [15.502521089647496, 15.60511641373273, 15.662643520026297, 15.671269878086171, 15.72653634091522, 15.732372591531984, 15.735658495479676]

# Not yet!
# Transformers(Encoder * 2 + Decoder * 2), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
# Local Self Attention
y3 = [15.495892652680121, 15.593208143779389, 15.670509142513932, 15.709444554223042, 15.724637358010378, 15.728880197789383, 15.729258483011803]

# Not yet!
# Transformers(Encoder * 2 + Decoder * 2), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
# Stride Sparse Self Attention
y4 = [15.491753832581484, 15.600451582711955, 15.677915513269225, 15.716706038110257, 15.734526410074793, 15.739291727207052, 15.738006738750139]

# draw
plt.plot(x, y1, marker='o', label='Full Self Attention')
plt.plot(x, y2, marker='o', label='Atrous Self Attention')
plt.plot(x, y3, marker='o', label='Local Self Attention')
plt.plot(x, y4, marker='o', label='Stride Sparse Self Attention')

# set y-axis to log scale
plt.yscale('log')

# add title and axis labels
plt.title('Comparison of sum rate of different Sparse Attention')
plt.xlabel('SNR (dB)')
plt.ylabel('sum rate(bandwith = 10) (log scale)')

# add legend
plt.legend()

# save the plot
plt.savefig('Comparison of sum rate of different Sparse Attention')

# display the plot
plt.grid(True)
plt.show()
####################################################################################################################

# 比例值
values = [1.0, 0.5, 125/1024, 574/1024]
labels = ['Full Self Attention', 'Atrous Self Attention', 'Local Self Attention', 'Stride Sparse Self Attention']

# 创建条形图
plt.bar(labels, values)

# 添加标题和标签
plt.title('Comparison of calculation loading ratio of different Sparse Attention and Full Self Attention')
plt.xlabel('Categories')
plt.ylabel('Proportion')

# 显示图形
plt.show()
####################################################################################################################
# data
x = [-10, -5, 0, 5, 10, 15, 20]

# CNN, epochs = 200, lr = 0.0001, batch_size = 32
y1 = [0.6419357114932095, 0.3948697806398535, 0.2653386565881674, 0.16621672269013613, 0.12786874896302208, 0.1266478453073014, 0.11196449103168699]

# Transformers(Encoder * 2 + Decoder * 2), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
# Full Self Attention
y2 = [0.03965744748711586, 0.028810562565922737, 0.02021711692214012, 0.015879059210419655, 0.01424616202712059, 0.013560600578784943, 0.013357477262616158]

# Transformers(Encoder * 2 + Decoder * 2), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
# Atrous Self Attention
y3 = [0.03856229782104492, 0.027260668575763702, 0.020888376981019974, 0.019930638372898102, 0.013781104236841202, 0.013130326755344868, 0.012763812206685543]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 400, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, Atrous Self Attention, ff layer 用 Conv1D
y4 = [0.03798374906182289, 0.024973848834633827, 0.020242398604750633, 0.015106507577002048, 0.013501455076038837, 0.012612905353307724, 0.012577512301504612]

# draw
# plt.plot(x, y1, marker='o', label='CNN')
plt.plot(x, y2, marker='o', label='Full Self Attention')
plt.plot(x, y3, marker='o', label='Atrous Self Attention')
plt.plot(x, y4, marker='o', label='Atrous Self Attention with Convolution layer')

# set y-axis to log scale
plt.yscale('log')

# add title and axis labels
plt.title('Comparison of NMSE of Atrous Self Attention with Convolution layer')
plt.xlabel('SNR (dB)')
plt.ylabel('NMSE (log scale)')

# add legend
plt.legend()

# save the plot
plt.savefig('Comparison of NMSE of Atrous Self Attention with Convolution layer')

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
# Full Self Attention
y2 = [15.492540660394557, 15.59108972424448, 15.668690136608705, 15.707705615158078, 15.722364224883243, 15.72851409896477, 15.730335747297762]

# Transformers(Encoder * 2 + Decoder * 2), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
# Atrous Self Attention
y3 = [15.502521089647496, 15.60511641373273, 15.662643520026297, 15.671269878086171, 15.72653634091522, 15.732372591531984, 15.735658495479676]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 400, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, Atrous Self Attention, ff layer 用 Conv1D
y4 = [15.507790908162843, 15.625787491983507, 15.668462428433932, 15.714642721214037, 15.729044567012842, 15.73701119848958, 15.737328425151173]

# draw
# plt.plot(x, y1, marker='o', label='CNN')
plt.plot(x, y2, marker='o', label='Full Self Attention')
plt.plot(x, y3, marker='o', label='Atrous Self Attention')
plt.plot(x, y4, marker='o', label='Atrous Self Attention with Convolution layer')

# set y-axis to log scale
plt.yscale('log')

# add title and axis labels
plt.title('Comparison of sum rate of Atrous Self Attention with Convolution layer')
plt.xlabel('SNR (dB)')
plt.ylabel('sum rate(bandwith = 10) (log scale)')

# add legend
plt.legend()

# save the plot
plt.savefig('Comparison of sum rate of Atrous Self Attention with Convolution layer')

# display the plot
plt.grid(True)
plt.show()