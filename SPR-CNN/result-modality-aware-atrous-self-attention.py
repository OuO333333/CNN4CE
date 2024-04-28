import matplotlib.pyplot as plt
import math

fontsize = 25

####################################################################################################################

# data
x = [-10, -5, 0, 5, 10, 15, 20]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, full attention
# encoder ff layer 用 Conv1D, decoder ff layer 用 ffn
# modality-aware transformers
y1 = [0.039645493030548096, 0.02747759222984314, 0.019949764013290405, 0.016916004940867424, 0.014825291931629181, 0.013869925402104855, 0.01333386730402708]

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
y6 = [0.04192124307155609, 0.04125601425766945, 0.04102673381567001, 0.0407208614051342, 0.04050442576408386, 0.040384791791439056, 0.0396304652094841]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, Local Self Attention, 50%
# encoder ff layer 用 Conv1D, decoder ff layer 用 ffn
# modality-aware transformers
y7 = [0.039498768746852875, 0.02775953710079193, 0.020758099853992462, 0.01766771636903286, 0.015282352454960346, 0.014278898946940899, 0.013560183346271515]

# draw
plt.plot(x, y2, marker='o', label='Atrous Self Attention(50%)')
plt.plot(x, y1, marker='o', label='Full Self Attention')
plt.plot(x, y7, marker='o', label='Local Self Attention(50%)')
plt.plot(x, y5, marker='o', label='Atrous Self Attention(33%)')
plt.plot(x, y6, marker='o', label='Random Self Attention(50%)')




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
# 自己寫的 multi-head self attention, full attention
# encoder ff layer 用 Conv1D, decoder ff layer 用 ffn
# modality-aware transformers
y1 = [15.49264954985823, 15.603153955212745, 15.671097689190924, 15.698389138207103, 15.717167092546497, 15.725739653022522, 15.730547503306035]

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
y6 = [15.471887670465838, 15.477959831493383, 15.48005189623524, 15.482842452373948, 15.484816921589978, 15.485908037287839, 15.49278659806734]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, Local Self Attention, 50%
# encoder ff layer 用 Conv1D, decoder ff layer 用 ffn
# modality-aware transformers
y7 = [15.49398709544224, 15.60060311439872, 15.663817260607027, 15.691631612401814, 15.713064030051235, 15.72207049582714, 15.728517878915074]

y1 = [value / 10 for value in y1]
y2 = [value / 10 for value in y2]
y5 = [value / 10 for value in y5]
y6 = [value / 10 for value in y6]
y7 = [value / 10 for value in y7]



# draw
plt.plot(x, y2, marker='o', label='Atrous Self Attention(50%)')
plt.plot(x, y1, marker='o', label='Full Self Attention')
plt.plot(x, y7, marker='o', label='Local Self Attention(50%)')
plt.plot(x, y5, marker='o', label='Atrous Self Attention(33%)')
plt.plot(x, y6, marker='o', label='Random Self Attention(50%)')


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