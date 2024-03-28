import matplotlib.pyplot as plt

# data
x = [-10, -5, 0, 5, 10, 15, 20]

# Transformers(Encoder * 2 + Decoder * 2), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
# Full Self Attention
y1 = [0.03987005725502968, 0.029254980385303497, 0.02019014209508896, 0.018056659027934074, 0.015157107263803482, 0.014463509432971478, 0.013804900459945202]

# Transformers(Encoder * 2 + Decoder * 2), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
# Atrous Self Attention
y2 = [0.03990931808948517, 0.027489781379699707, 0.020104501396417618, 0.01623721234500408, 0.014446022920310497, 0.0135115347802639, 0.01348433643579483]

# Not yet!
# Transformers(Encoder * 2 + Decoder * 2), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
# Local Self Attention
y3 = [0.039571523666381836, 0.02931174635887146, 0.020710840821266174, 0.01748591847717762, 0.015006369911134243, 0.014497745782136917, 0.014882778748869896]

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
y1 = [15.490602115597438, 15.58706522935487, 15.66893307641061, 15.688134016492416, 15.714188500585596, 15.720413966130947, 15.726322933376025]

# Transformers(Encoder * 2 + Decoder * 2), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
# Atrous Self Attention
y2 = [15.490244286888137, 15.603043863477247, 15.669704313731152, 15.704488501364377, 15.72057086775953, 15.728954210281074, 15.729198126988027]

# Not yet!
# Transformers(Encoder * 2 + Decoder * 2), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
# Local Self Attention
y3 = [15.493323801466287, 15.586551055087641, 15.664242912388673, 15.693266228231288, 15.715541668319254, 15.720106725025246, 15.716651054125423]

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
y2 = [0.03987005725502968, 0.029254980385303497, 0.02019014209508896, 0.018056659027934074, 0.015157107263803482, 0.014463509432971478, 0.013804900459945202]

# Transformers(Encoder * 2 + Decoder * 2), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
# Atrous Self Attention
y3 = [0.03990931808948517, 0.027489781379699707, 0.020104501396417618, 0.01623721234500408, 0.014446022920310497, 0.0135115347802639, 0.01348433643579483]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 400, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, Atrous Self Attention, ff layer 用 Conv1D
y4 = [0.03733016178011894, 0.026352835819125175, 0.01988144963979721, 0.015685485675930977, 0.014073392376303673, 0.013301283121109009, 0.01246847864240408]

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
y2 = [15.490602115597438, 15.58706522935487, 15.66893307641061, 15.688134016492416, 15.714188500585596, 15.720413966130947, 15.726322933376025]

# Transformers(Encoder * 2 + Decoder * 2), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
# Atrous Self Attention
y3 = [15.490244286888137, 15.603043863477247, 15.669704313731152, 15.704488501364377, 15.72057086775953, 15.728954210281074, 15.729198126988027]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 400, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, Atrous Self Attention, ff layer 用 Conv1D
y4 = [15.513741995396554, 15.613326189757453, 15.67171283671444, 15.70944412424403, 15.723914389956803, 15.730839697733403, 15.738305735345495]

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