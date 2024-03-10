import matplotlib.pyplot as plt

# data
x = [-10, -5, 0, 5, 10, 15, 20]
# CNN, epochs = 200, lr = 0.0001, batch_size = 32
y1 = [0.649711661089483, 0.3475041160976275, 0.22225152482196797, 0.14494572869140035, 0.12128514659706721, 0.10142940629835279, 0.08489599243689268]

# Transformers(Encoder * 9), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
y2 = [0.03319474682211876, 0.030052466318011284, 0.016912512481212616, 0.013853426091372967, 0.013164506293833256, 0.011629097163677216, 0.011142524890601635]

# Transformers(Encoder * 3 + Decoder * 3), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
y3 = [0.03295202925801277, 0.024725012481212616, 0.016572220250964165, 0.01351124607026577, 0.011882485821843147, 0.011450755409896374, 0.010888315737247467]

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
y1 = [14.058652952281365, 14.863718856051712, 15.277555758699604, 15.44859638003606, 15.347140577498028, 15.615965469824227, 15.62448257698082]

# Transformers(Encoder * 9), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
y2 = [15.551511616320626, 15.57999690907113, 15.698507919501896, 15.725959078579967, 15.73213395024454, 15.745886624325953, 15.750242102848484]

# Transformers(Encoder * 3 + Decoder * 3), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
y3 = [15.553713778468975, 15.62816334863857, 15.701564150189729, 15.72902636704422, 15.743617962580778, 15.747483214712926, 15.752517105400617]

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
y1 = [0.03295202925801277, 0.024725012481212616, 0.016572220250964165, 0.01351124607026577, 0.011882485821843147, 0.011450755409896374, 0.010888315737247467]

# Transformers(Encoder * 3 + Decoder * 3), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
# Atrous Self Attention
y2 = [0.032459087669849396, 0.021085403859615326, 0.015914564952254295, 0.012831115163862705, 0.011864903382956982, 0.011223804205656052, 0.010580628179013729]

# Transformers(Encoder * 3 + Decoder * 3), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
# Local Self Attention
y3 = [0.03189250826835632, 0.02233617939054966, 0.016399750486016273, 0.013261282816529274, 0.011701206676661968, 0.010796995833516121, 0.010449905879795551]

# Transformers(Encoder * 3 + Decoder * 3), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
# Stride Sparse Self Attention
y4 = [0.032578837126493454, 0.021571574732661247, 0.015432968735694885, 0.012876344844698906, 0.011555371806025505, 0.010694130323827267, 0.010519152507185936]

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
y1 = [15.553713778468975, 15.62816334863857, 15.701564150189729, 15.72902636704422, 15.743617962580778, 15.747483214712926, 15.752517105400617]

# Transformers(Encoder * 3 + Decoder * 3), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
# Atrous Self Attention
y2 = [15.558185465907597, 15.660977481197651, 15.707468871895095, 15.735121293979006, 15.743775339387081, 15.749514615639303, 15.755270250083662]

# Transformers(Encoder * 3 + Decoder * 3), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
# Local Self Attention
y3 = [15.563323394555795, 15.64970900162957, 15.703112870690317, 15.731266684382328, 15.745241032091705, 15.753334309883158, 15.756439769949978]

# Transformers(Encoder * 3 + Decoder * 3), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
# Stride Sparse Self Attention
y4 = [15.557099334636352, 15.656598549726002, 15.711791330535348, 15.734716119263615, 15.746546646253226, 15.754254715761103, 15.755820227182856]

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
values = [1.0, 0.5, 74/256, 158/256]
labels = ['Full Self Attention', 'Atrous Self Attention', 'Local Self Attention', 'Stride Sparse Self Attention']

# 创建条形图
plt.bar(labels, values)

# 添加标题和标签
plt.title('Comparison of calculation loading ratio of different Sparse Attention and Full Self Attention')
plt.xlabel('Categories')
plt.ylabel('Proportion')

# 显示图形
plt.show()