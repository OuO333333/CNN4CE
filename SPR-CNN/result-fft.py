import matplotlib.pyplot as plt

# data
x = [-10, -5, 0, 5, 10, 15, 20]

# CNN, epochs = 200, lr = 0.0001, batch_size = 32
y1 = [0.6419357114932095, 0.3948697806398535, 0.2653386565881674, 0.16621672269013613, 0.12786874896302208, 0.1266478453073014, 0.11196449103168699]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 400, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, Atrous Self Attention, ff layer 用 Conv1D
y2 = [0.03733016178011894, 0.026352835819125175, 0.01988144963979721, 0.015685485675930977, 0.014073392376303673, 0.013301283121109009, 0.01246847864240408]

# Not yet!
# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, Atrous Self Attention, ff layer 用 Conv1D, 使用 fft in multi-head-attention
y3 = [0.04002300277352333, 0.025759074836969376, 0.028633849695324898, 0.01591109298169613, 0.01619063876569271, 0.014287014491856098, 0.013909406960010529]

# Not yet!
# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, Atrous Self Attention, ff layer 用 Conv1D
# Modality-aware Transformer
y4 = [0.04002300277352333, 0.025759074836969376, 0.028633849695324898, 0.01591109298169613, 0.01619063876569271, 0.014287014491856098, 0.013909406960010529]

# draw
# plt.plot(x, y1, marker='o', label='CNN')
plt.plot(x, y2, marker='o', label='Atrous Self Attention with Convolution layer')
plt.plot(x, y3, marker='o', label='Atrous Self Attention with Convolution layer with fft in multi-head-attention')
plt.plot(x, y4, marker='o', label='Modality-aware Transformer')


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

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 400, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, Atrous Self Attention, ff layer 用 Conv1D
y2 = [15.513741995396554, 15.613326189757453, 15.67171283671444, 15.70944412424403, 15.723914389956803, 15.730839697733403, 15.738305735345495]

# Not yet!
# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, Atrous Self Attention, ff layer 用 Conv1D, 使用 fft in multi-head-attention
y3 = [15.489207487952974, 15.618692996837884, 15.592689694203141, 15.707417936913393, 15.704906880100955, 15.721997719195706, 15.725385431670285]

# Not yet!
# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, Atrous Self Attention, ff layer 用 Conv1D
# Modality-aware Transformer
y4 = [15.489207487952974, 15.618692996837884, 15.592689694203141, 15.707417936913393, 15.704906880100955, 15.721997719195706, 15.725385431670285]

# draw
# plt.plot(x, y1, marker='o', label='CNN')
plt.plot(x, y2, marker='o', label='Atrous Self Attention with Convolution layer')
plt.plot(x, y3, marker='o', label='Atrous Self Attention with Convolution layer with fft in multi-head-attention')
plt.plot(x, y4, marker='o', label='Modality-aware Transformer')

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