import matplotlib.pyplot as plt

# data
x = [-10, -5, 0, 5, 10, 15, 20]
# CNN, epochs = 200, lr = 0.0001, batch_size = 32
y1 = [0.8902655945595724, 0.5751199399355023, 0.36848606844546833, 0.29410193315144156, 0.2553683132455284, 0.23459051626799857, 0.2303461244047422]

# Transformers(Encoder * 9), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
y2 = [0.03453879430890083, 0.026762498542666435, 0.021355926990509033, 0.01854751817882061, 0.016703907400369644, 0.015434942208230495, 0.013476668857038021]

# Transformers(Encoder * 3 + Decoder * 3), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
y3 = [0.031660910695791245, 0.023208769038319588, 0.017391914501786232, 0.014003201387822628, 0.01258099265396595, 0.012034251354634762, 0.011669430881738663]

# draw
plt.plot(x, y1, marker='o', label='CNN')
plt.plot(x, y2, marker='o', label='Transformers(Encoder)')
plt.plot(x, y3, marker='o', label='Transformers(Encoder + Decoder)')

# add title and axis labels
plt.title('Comparison of NMSE of CNN and Transformers(Encoder) and Transformers(Encoder + Decoder)')
plt.xlabel('SNR (dB)')
plt.ylabel('NMSE')

# add legend
plt.legend()

# save the plot
plt.savefig('Comparison of NMSE of CNN and Transformers(Encoder) and Transformers(Encoder + Decoder)')

# display the plot
plt.grid(True)
plt.show()
####################################################################################################################
# data
x = [-10, -5, 0, 5, 10, 15, 20]
# CNN, epochs = 200, lr = 0.0001, batch_size = 32
y1 = [12.940970626378812, 13.972075582297652, 14.508584726671309, 14.825802753405233, 14.9719951899972, 15.106508174427415, 15.093083672359931]

# Transformers(Encoder * 9), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
y2 = [15.539312784548422, 15.609762987385716, 15.658542552063942, 15.68381586680459, 15.700382733433553, 15.711774756083067, 15.729337311794527]

# Transformers(Encoder * 3 + Decoder * 3), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
y3 = [15.56542539475691, 15.641844298837185, 15.69420245522329, 15.72461722262373, 15.73736293004918, 15.742259803203385, 15.74552633958734]

# draw
plt.plot(x, y1, marker='o', label='CNN')
plt.plot(x, y2, marker='o', label='Transformers(Encoder)')
plt.plot(x, y3, marker='o', label='Transformers(Encoder + Decoder)')

# add title and axis labels
plt.title('Comparison of sum rate of CNN and Transformers(Encoder) and Transformers(Encoder + Decoder)')
plt.xlabel('SNR (dB)')
plt.ylabel('sum rate(bandwith = 10)')

# add legend
plt.legend()

# save the plot
plt.savefig('Comparison of sum rate of CNN and Transformers(Encoder) and Transformers(Encoder + Decoder)')

# display the plot
plt.grid(True)
plt.show()
####################################################################################################################
# data
x = [-10, -5, 0, 5, 10, 15, 20]

# Transformers(Encoder * 3 + Decoder * 3), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
# Full Self Attention
y1 = [0.031660910695791245, 0.023208769038319588, 0.017391914501786232, 0.014003201387822628, 0.01258099265396595, 0.012034251354634762, 0.011669430881738663]

# Transformers(Encoder * 3 + Decoder * 3), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
# Atrous Self Attention
y2 = [0.03277484327554703, 0.022869588807225227, 0.01983608677983284, 0.013921788893640041, 0.012456191703677177, 0.012495182454586029, 0.010741359554231167]

# Transformers(Encoder * 3 + Decoder * 3), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
# Local Self Attention
y3 = [0.034291502088308334, 0.025741055607795715, 0.01740306243300438, 0.013940487988293171, 0.012150107882916927, 0.011353292502462864, 0.011100418865680695]

# Transformers(Encoder * 3 + Decoder * 3), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
# Stride Sparse Self Attention
y4 = [0.03285108879208565, 0.0230310820043087, 0.01766713708639145, 0.014568011276423931, 0.013989264145493507, 0.01138947531580925, 0.011203959584236145]

# draw
plt.plot(x, y1, marker='o', label='Full Self Attention')
plt.plot(x, y2, marker='o', label='Atrous Self Attention')
plt.plot(x, y3, marker='o', label='Local Self Attention')
plt.plot(x, y4, marker='o', label='Stride Sparse Self Attention')

# add title and axis labels
plt.title('Comparison of NMSE of different Sparse Attention')
plt.xlabel('SNR (dB)')
plt.ylabel('NMSE')

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

# Transformers(Encoder * 3 + Decoder * 3), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
# Full Self Attention
y1 = [15.56542539475691, 15.641844298837185, 15.69420245522329, 15.72461722262373, 15.73736293004918, 15.742259803203385, 15.74552633958734]

# Transformers(Encoder * 3 + Decoder * 3), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
# Atrous Self Attention
y2 = [15.55532376144608, 15.644902614717342, 15.67222529243178, 15.72534715048587, 15.738480920042843, 15.7381316534288, 15.753832913509044]

# Transformers(Encoder * 3 + Decoder * 3), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
# Local Self Attention
y3 = [15.541558425324208, 15.618991486838459, 15.69410226947647, 15.725179520479465, 15.741222315527239, 15.748356436102494, 15.750619762581621]

# Transformers(Encoder * 3 + Decoder * 3), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
# Stride Sparse Self Attention
y4 = [15.5546319101134, 15.643446541205929, 15.69172941135746, 15.719552365946374, 15.724742243510288, 15.74803261180948, 15.749693078885972]

# draw
plt.plot(x, y1, marker='o', label='Full Self Attention')
plt.plot(x, y2, marker='o', label='Atrous Self Attention')
plt.plot(x, y3, marker='o', label='Local Self Attention')
plt.plot(x, y4, marker='o', label='Stride Sparse Self Attention')

# add title and axis labels
plt.title('Comparison of sum rate of different Sparse Attention')
plt.xlabel('SNR (dB)')
plt.ylabel('sum rate(bandwith = 10)')

# add legend
plt.legend()

# save the plot
plt.savefig('Comparison of sum rate of different Sparse Attention')

# display the plot
plt.grid(True)
plt.show()
####################################################################################################################

# 比例值
values = [1.0, 0.5, 34/64, 46/64]
labels = ['Full Self Attention', 'Atrous Self Attention', 'Local Self Attention', 'Stride Sparse Self Attention']

# 创建条形图
plt.bar(labels, values)

# 添加标题和标签
plt.title('Comparison of calculation loading ratio of different Sparse Attention and Full Self Attention')
plt.xlabel('Categories')
plt.ylabel('Proportion')

# 显示图形
plt.show()