import matplotlib.pyplot as plt

# data
x = [-10, -5, 0, 5, 10, 15, 20]

# Transformers(Encoder * 2 + Decoder * 2), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
# Full Self Attention
y1 = [0.03957320749759674, 0.0313139446079731, 0.018902089446783066, 0.015466086566448212, 0.013048394583165646, 0.012472232803702354, 0.01183935534209013]

# Transformers(Encoder * 2 + Decoder * 2), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
# Atrous Self Attention
y2 = [0.03793514147400856, 0.02725178748369217, 0.018875764682888985, 0.014742224477231503, 0.012882505543529987, 0.011958373710513115, 0.011849570088088512]

# Transformers(Encoder * 2 + Decoder * 2), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
# Local Self Attention
y3 = [0.03928970918059349, 0.028576577082276344, 0.02001512609422207, 0.015685435384511948, 0.013992800377309322, 0.013519784435629845, 0.013477599248290062]

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
y1 = [15.493308527872895, 15.568404974771003, 15.680528308855868, 15.711414292671664, 15.733107185589379, 15.73827208055814, 15.74394318113315]

# Transformers(Encoder * 2 + Decoder * 2), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
# Atrous Self Attention
y2 = [15.50823359352834, 15.605196758612788, 15.680765231318805, 15.717912628820924, 15.734594474390205, 15.742876849423679, 15.743851673854186]

# Transformers(Encoder * 2 + Decoder * 2), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
# Local Self Attention
y3 = [15.495892652680121, 15.593208143779389, 15.670509142513932, 15.709444554223042, 15.724637358010378, 15.728880197789383, 15.729258483011803]

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
y2 = [0.03957320749759674, 0.0313139446079731, 0.018902089446783066, 0.015466086566448212, 0.013048394583165646, 0.012472232803702354, 0.01183935534209013]

# Transformers(Encoder * 2 + Decoder * 2), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
# Atrous Self Attention
y3 = [0.03793514147400856, 0.02725178748369217, 0.018875764682888985, 0.014742224477231503, 0.012882505543529987, 0.011958373710513115, 0.011849570088088512]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 400, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, Atrous Self Attention, ff layer 用 Conv1D
y4 = [0.03621919825673103, 0.0272029098123312, 0.017789877951145172, 0.014526165090501308, 0.012576806358993053, 0.011850962415337563, 0.011727375909686089]

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
y2 = [15.493308527872895, 15.568404974771003, 15.680528308855868, 15.711414292671664, 15.733107185589379, 15.73827208055814, 15.74394318113315]

# Transformers(Encoder * 2 + Decoder * 2), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
# Atrous Self Attention
y3 = [15.50823359352834, 15.605196758612788, 15.680765231318805, 15.717912628820924, 15.734594474390205, 15.742876849423679, 15.743851673854186]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 400, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, Atrous Self Attention, ff layer 用 Conv1D
y4 = [15.523851843084003, 15.605638873813945, 15.690533174195762, 15.71985169518842, 15.73733479108248, 15.74383919925955, 15.744946353832843]

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