import matplotlib.pyplot as plt
import math

fontsize = 25

####################################################################################################################

# data
x = [-10, -5, 0, 5, 10, 15, 20]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, atrous_self_attention,
# encoder ff layer 用 Conv1D, decoder ff layer 用 ffn
# modality-aware transformers
y2 = [0.03944879397749901, 0.026776893064379692, 0.01985848695039749, 0.015884367749094963, 0.013839473016560078, 0.013212382793426514, 0.012837452813982964]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, atrous_self_attention, d = 3
# encoder ff layer 用 Conv1D, decoder ff layer 用 ffn
# modality-aware transformers
y5 = [0.03968484327197075, 0.028611933812499046, 0.023897996172308922, 0.02146201953291893, 0.019011519849300385, 0.016211938112974167, 0.01379705686122179]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, random_self_attention
# encoder ff layer 用 Conv1D, decoder ff layer 用 ffn
# modality-aware transformers
y6 = [0.03952145203948021, 0.028553225100040436, 0.02063240297138691, 0.01693073660135269, 0.014729450456798077, 0.014260709285736084, 0.013719967566430569]

# draw
plt.plot(x, y2, marker='o', label='Atrous Self Attention(d=2)')
plt.plot(x, y5, marker='o', label='Atrous Self Attention(d=3)')
plt.plot(x, y6, marker='o', label='Random Self Attention')


# set y-axis to log scale
plt.yscale('log')

# add title and axis labels
plt.title('Comparison of NMSE of Proposed Transformer of different Sparse Attention', fontsize=fontsize)
plt.xlabel('SNR (dB)', fontsize=fontsize)
plt.ylabel('NMSE (log scale)', fontsize=fontsize)

# add legend
plt.legend(fontsize=fontsize)

# save the plot
plt.savefig('Comparison of NMSE of Proposed Transformer of different Sparse Attention')

# display the plot
plt.grid(True)
plt.show()
####################################################################################################################

# data
x = [-10, -5, 0, 5, 10, 15, 20]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, atrous_self_attention,
# encoder ff layer 用 Conv1D, decoder ff layer 用 ffn
# modality-aware transformers
y2 = [15.494442592010083, 15.609491910438882, 15.671919573771865, 15.707657914006779, 15.726012765282944, 15.731636832092104, 15.734998353604407]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, atrous_self_attention, d = 3
# encoder ff layer 用 Conv1D, decoder ff layer 用 ffn
# modality-aware transformers
y5 = [15.49229083432177, 15.592888048108476, 15.635502141705988, 15.65747419928054, 15.679543601852913, 15.704715510298538, 15.726393267161576]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, random_self_attention
# encoder ff layer 用 Conv1D, decoder ff layer 用 ffn
# modality-aware transformers
y6 = [15.494442592010083, 15.609491910438882, 15.671919573771865, 15.707657914006779, 15.726012765282944, 15.731636832092104, 15.734998353604407]

y2 = [value / 10 for value in y2]
y5 = [value / 10 for value in y5]
y6 = [value / 10 for value in y6]

# draw
plt.plot(x, y2, marker='o', label='Atrous Self Attention')
plt.plot(x, y5, marker='o', label='Atrous Self Attention(d=3)')
plt.plot(x, y6, marker='o', label='Random Self Attention')

# set y-axis to log scale
plt.yscale('log')

# add title and axis labels
plt.title('Comparison of sum rate of Proposed Transformer of different Sparse Attention', fontsize=fontsize)
plt.xlabel('SNR (dB)', fontsize=fontsize)
plt.ylabel('sum rate(bandwith = 1) (log scale)', fontsize=fontsize)

# add legend
plt.legend(fontsize=fontsize)

# save the plot
plt.savefig('Comparison of sum rate of Proposed Transformer of different Sparse Attention')

# display the plot
plt.grid(True)
plt.show()