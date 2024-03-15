import matplotlib.pyplot as plt

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
y4 = [0.03621919825673103, 0.0272029098123312, 0.017789877951145172, 0.014526165090501308, 0.012576806358993053, 0.011595480144023895, 0.011442258022725582]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 400, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, Atrous Self Attention, ff layer 用 Conv1D, 使用 BatchNormalization
y5 = [0.05668972060084343, 0.028094256296753883, 0.020573817193508148, 0.028254395350813866, 0.02500385418534279, 0.020342955365777016, 0.0244763046503067]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 400, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, Atrous Self Attention, ff layer 用 Conv1D, reverse attention layer and ff layer
y6 = [0.03578875958919525, 0.02395220287144184, 0.017472850158810616, 0.013907503336668015, 0.01374274492263794, 0.013484571129083633, 0.011466003023087978]

# draw
# plt.plot(x, y1, marker='o', label='CNN')
plt.plot(x, y2, marker='o', label='Full Self Attention')
plt.plot(x, y3, marker='o', label='Atrous Self Attention')
plt.plot(x, y4, marker='o', label='Atrous Self Attention with Convolution layer')
plt.plot(x, y5, marker='o', label='Atrous Self Attention with Convolution layer and BatchNormalization')
plt.plot(x, y6, marker='o', label='Atrous Self Attention with Convolution layer with reverse')


# set y-axis to log scale
plt.yscale('log')

# add title and axis labels
plt.title('Comparison of NMSE of Atrous Self Attention with Convolution layer and Others')
plt.xlabel('SNR (dB)')
plt.ylabel('NMSE (log scale)')

# add legend
plt.legend()

# save the plot
plt.savefig('Comparison of NMSE of Atrous Self Attention with Convolution layer and Others')

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
y4 = [15.523851843084003, 15.605638873813945, 15.690533174195762, 15.71985169518842, 15.73733479108248, 15.7461279053344, 15.747500409925326]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 400, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, Atrous Self Attention, ff layer 用 Conv1D, 使用 BatchNormalization
y5 = [15.336422811727001, 15.597574072960933, 15.665477346013974, 15.596124639155295, 15.625516490153341, 15.667556703768637, 15.630281030850368]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 400, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, Atrous Self Attention, ff layer 用 Conv1D, reverse attention layer and ff layer
y6 = [15.527766755550392, 15.635012816385569, 15.693383651016465, 15.725402559240159, 15.72688045009137, 15.729196030441454, 15.747287674243054]

# draw
# plt.plot(x, y1, marker='o', label='CNN')
plt.plot(x, y2, marker='o', label='Full Self Attention')
plt.plot(x, y3, marker='o', label='Atrous Self Attention')
plt.plot(x, y4, marker='o', label='Atrous Self Attention with Convolution layer')
plt.plot(x, y5, marker='o', label='Atrous Self Attention with Convolution layer and BatchNormalization')
plt.plot(x, y6, marker='o', label='Atrous Self Attention with Convolution layer with reverse')

# set y-axis to log scale
plt.yscale('log')

# add title and axis labels
plt.title('Comparison of sum rate of Atrous Self Attention with Convolution layer and Others')
plt.xlabel('SNR (dB)')
plt.ylabel('sum rate(bandwith = 10) (log scale)')

# add legend
plt.legend()

# save the plot
plt.savefig('Comparison of sum rate of Atrous Self Attention with Convolution layer and Others')

# display the plot
plt.grid(True)
plt.show()