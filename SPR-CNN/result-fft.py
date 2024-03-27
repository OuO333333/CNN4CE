import matplotlib.pyplot as plt

# data
x = [-10, -5, 0, 5, 10, 15, 20]

# CNN, epochs = 200, lr = 0.0001, batch_size = 32
y1 = [0.6419357114932095, 0.3948697806398535, 0.2653386565881674, 0.16621672269013613, 0.12786874896302208, 0.1266478453073014, 0.11196449103168699]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, Atrous Self Attention, ff layer 用 Conv1D
y4 = [0.03621919825673103, 0.0272029098123312, 0.017789877951145172, 0.014526165090501308, 0.012576806358993053, 0.011850962415337563, 0.011727375909686089]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, Atrous Self Attention, ff layer 用 Conv1D, 使用 BatchNormalization
y5 = [0.05668972060084343, 0.028094256296753883, 0.020573817193508148, 0.028254395350813866, 0.02500385418534279, 0.020342955365777016, 0.0244763046503067]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, Atrous Self Attention, ff layer 用 Conv1D, change norm position
y7 = [0.039442263543605804, 0.02432405948638916, 0.021599117666482925, 0.016517218202352524, 0.015613597817718983, 0.014050054363906384, 0.012439227662980556]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, Atrous Self Attention, ff layer 用 Conv1D, 使用 fft in multi-head-attention
y8 = [0.04002300277352333, 0.025759074836969376, 0.028633849695324898, 0.01591109298169613, 0.01619063876569271, 0.014287014491856098, 0.013909406960010529]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, Atrous Self Attention, ff layer 用 Conv1D
# modality-aware transformers

# draw
# plt.plot(x, y1, marker='o', label='CNN')
plt.plot(x, y4, marker='o', label='Atrous Self Attention with Convolution layer')
# plt.plot(x, y5, marker='o', label='Atrous Self Attention with Convolution layer and BatchNormalization')
# plt.plot(x, y7, marker='o', label='Atrous Self Attention with Convolution layer with changing norm position')
plt.plot(x, y8, marker='o', label='Atrous Self Attention with Convolution layer with fft in multi-head-attention')


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

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, Atrous Self Attention, ff layer 用 Conv1D
y4 = [15.523851843084003, 15.605638873813945, 15.690533174195762, 15.71985169518842, 15.73733479108248, 15.74383919925955, 15.744946353832843]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, Atrous Self Attention, ff layer 用 Conv1D, 使用 BatchNormalization
y5 = [15.336422811727001, 15.597574072960933, 15.665477346013974, 15.596124639155295, 15.625516490153341, 15.667556703768637, 15.630281030850368]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, Atrous Self Attention, ff layer 用 Conv1D, change norm position
y7 = [15.494502152019606, 15.631655688035869, 15.65623851411047, 15.701972776773513, 15.710089680065902, 15.724123697543275, 15.738567858253838]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, Atrous Self Attention, ff layer 用 Conv1D, 使用 fft in multi-head-attention
y8 = [15.489207487952974, 15.618692996837884, 15.592689694203141, 15.707417936913393, 15.704906880100955, 15.721997719195706, 15.725385431670285]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, Atrous Self Attention, ff layer 用 Conv1D
# modality-aware transformers

# draw
# plt.plot(x, y1, marker='o', label='CNN')
plt.plot(x, y4, marker='o', label='Atrous Self Attention with Convolution layer')
# plt.plot(x, y5, marker='o', label='Atrous Self Attention with Convolution layer and BatchNormalization')
# plt.plot(x, y7, marker='o', label='Atrous Self Attention with Convolution layer with changing norm position')
plt.plot(x, y8, marker='o', label='Atrous Self Attention with Convolution layer with fft in multi-head-attention')

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