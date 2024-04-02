import matplotlib.pyplot as plt
import math

# data
x = [-10, -5, 0, 5, 10, 15, 20]
# CNN, epochs = 200, lr = 0.0001, batch_size = 32
y1 = [0.6419357114932095, 0.3948697806398535, 0.2653386565881674, 0.16621672269013613, 0.12786874896302208, 0.1266478453073014, 0.11196449103168699]

# Transformers(Encoder * 2 + Decoder * 2), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
y2 = [0.03987005725502968, 0.029254980385303497, 0.02019014209508896, 0.018056659027934074, 0.015157107263803482, 0.014463509432971478, 0.013804900459945202]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, full attention
# encoder ff layer 用 Conv1D, decoder ff layer 用 ffn
# modality-aware transformers
y3 = [0.039829012006521225, 0.02747759222984314, 0.019949764013290405, 0.017768146470189095, 0.014825291931629181, 0.013869925402104855, 0.01333386730402708]

# set y-axis to log scale
plt.yscale('log', base=2)

# draw
plt.plot(x, y1, marker='o', label='CNN')
plt.plot(x, y2, marker='o', label='Transformers')
plt.plot(x, y3, marker='o', label='Proposed Transformer')

# add title and axis labels
plt.title('Comparison of NMSE of CNN and Transformers and Proposed Transformer')
plt.xlabel('SNR (dB)')
plt.ylabel('NMSE (log scale)')

# add legend
plt.legend()

# save the plot
plt.savefig('Comparison of NMSE of CNN and Transformers and Proposed Transformer')

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
y2 = [15.490602115597438, 15.58706522935487, 15.66893307641061, 15.688134016492416, 15.714188500585596, 15.720413966130947, 15.726322933376025]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, full attention
# encoder ff layer 用 Conv1D, decoder ff layer 用 ffn
# modality-aware transformers
y3 = [15.49097643884792, 15.603153955212745, 15.671097689190924, 15.690728563181299, 15.717167092546497, 15.725739653022522, 15.730547503306035]

# set y-axis to log scale
plt.yscale('log', base=10)

# draw
plt.plot(x, y1, marker='o', label='CNN')
plt.plot(x, y2, marker='o', label='Transformers')
plt.plot(x, y3, marker='o', label='Proposed Transformer')

# add title and axis labels
plt.title('Comparison of sum rate of CNN and Transformers and Proposed Transformer')
plt.xlabel('SNR (dB)')
plt.ylabel('sum rate(bandwith = 10) (log scale)')

# add legend
plt.legend()

# save the plot
plt.savefig('Comparison of sum rate of CNN and Transformers and Proposed Transformer')

# display the plot
plt.grid(True)
plt.show()
####################################################################################################################

# data

# Transformers(Encoder * 2 + Decoder * 2), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
y1 = [0.03987005725502968, 0.029254980385303497, 0.02019014209508896, 0.018056659027934074, 0.015157107263803482, 0.014463509432971478, 0.013804900459945202]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, full attention
# encoder ff layer 用 Conv1D, decoder ff layer 用 ffn
# modality-aware transformers
y2 = [0.039829012006521225, 0.02747759222984314, 0.019949764013290405, 0.017768146470189095, 0.014825291931629181, 0.013869925402104855, 0.01333386730402708]

# draw
plt.plot(x, y1, marker='o', label='Transformers')
plt.plot(x, y2, marker='o', label='Proposed Transformer')

# set y-axis to log scale
plt.yscale('log', base=2)

# add title and axis labels
plt.title('Comparison of NMSE of Transformers and Proposed Transformer')
plt.xlabel('SNR (dB)')
plt.ylabel('NMSE (log scale)')

# add legend
plt.legend()

# save the plot
plt.savefig('Comparison of NMSE of Transformers and Proposed Transformer')

# display the plot
plt.grid(True)
plt.show()
####################################################################################################################
# data
x = [-10, -5, 0, 5, 10, 15, 20]

# Transformers(Encoder * 2 + Decoder * 2), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
y1 = [15.490602115597438, 15.58706522935487, 15.66893307641061, 15.688134016492416, 15.714188500585596, 15.720413966130947, 15.726322933376025]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, full attention
# encoder ff layer 用 Conv1D, decoder ff layer 用 ffn
# modality-aware transformers
y2 = [15.49097643884792, 15.603153955212745, 15.671097689190924, 15.690728563181299, 15.717167092546497, 15.725739653022522, 15.730547503306035]

# draw
plt.plot(x, y1, marker='o', label='Transformers')
plt.plot(x, y2, marker='o', label='Proposed Transformer')

# set y-axis to log scale
plt.yscale('log', base=2)

# add title and axis labels
plt.title('Comparison of sum rate of Transformers and Proposed Transformer')
plt.xlabel('SNR (dB)')
plt.ylabel('sum rate(bandwith = 10) (log scale)')

# add legend
plt.legend()

# save the plot
plt.savefig('Comparison of sum rate of Transformers and Proposed Transformer')

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

# save the plot
plt.savefig('Comparison of calculation loading ratio of different Sparse Attention and Full Self Attention')

# 显示图形
plt.show()
####################################################################################################################

# data
x = [-10, -5, 0, 5, 10, 15, 20]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, full attention
# encoder ff layer 用 Conv1D, decoder ff layer 用 ffn
# modality-aware transformers
y1 = [0.039829012006521225, 0.02747759222984314, 0.019949764013290405, 0.017768146470189095, 0.014825291931629181, 0.013869925402104855, 0.01333386730402708]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, atrous_self_attention,
# encoder ff layer 用 Conv1D, decoder ff layer 用 ffn
# modality-aware transformers
y2 = [0.03944879397749901, 0.026776893064379692, 0.01985848695039749, 0.015884367749094963, 0.013839473016560078, 0.013212382793426514, 0.012837452813982964]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, Local Self Attention
# encoder ff layer 用 Conv1D, decoder ff layer 用 ffn
# modality-aware transformers
y3 = [0.03952145203948021, 0.028553225100040436, 0.02063240297138691, 0.01693073660135269, 0.014729450456798077, 0.014260709285736084, 0.013719967566430569]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, Stride Sparse Self Attention
# encoder ff layer 用 Conv1D, decoder ff layer 用 ffn
# modality-aware transformers
y4 = [0.03975171595811844, 0.02894938737154007, 0.020740732550621033, 0.016013476997613907, 0.014851024374365807, 0.013610122725367546, 0.013560788705945015]

# draw
plt.plot(x, y1, marker='o', label='Full Self Attention')
plt.plot(x, y2, marker='o', label='Atrous Self Attention')
plt.plot(x, y3, marker='o', label='Local Self Attention')
plt.plot(x, y4, marker='o', label='Stride Sparse Self Attention')

# set y-axis to log scale
plt.yscale('log')

# add title and axis labels
plt.title('Comparison of NMSE of Proposed Transformer of different Sparse Attention')
plt.xlabel('SNR (dB)')
plt.ylabel('sum rate(bandwith = 10) (log scale)')

# add legend
plt.legend()

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
y1 = [15.49097643884792, 15.603153955212745, 15.671097689190924, 15.690728563181299, 15.717167092546497, 15.725739653022522, 15.730547503306035]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, atrous_self_attention,
# encoder ff layer 用 Conv1D, decoder ff layer 用 ffn
# modality-aware transformers
y2 = [15.494442592010083, 15.609491910438882, 15.671919573771865, 15.707657914006779, 15.726012765282944, 15.731636832092104, 15.734998353604407]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, Local Self Attention
# encoder ff layer 用 Conv1D, decoder ff layer 用 ffn
# modality-aware transformers
y3 = [15.493780284204943, 15.593419636118835, 15.664949597637182, 15.698256769090925, 15.718027282607842, 15.722233706836965, 15.727084762060883]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, Stride Sparse Self Attention
# encoder ff layer 用 Conv1D, decoder ff layer 用 ffn
# modality-aware transformers
y4 = [15.491681067760004, 15.589832575098683, 15.663973687062441, 15.706498242577116, 15.716936093266758, 15.728069976733451, 15.728512427221728]

# draw
plt.plot(x, y1, marker='o', label='Full Self Attention')
plt.plot(x, y2, marker='o', label='Atrous Self Attention')
plt.plot(x, y3, marker='o', label='Local Self Attention')
plt.plot(x, y4, marker='o', label='Stride Sparse Self Attention')

# set y-axis to log scale
plt.yscale('log')

# add title and axis labels
plt.title('Comparison of sum rate of Proposed Transformer of different Sparse Attention')
plt.xlabel('SNR (dB)')
plt.ylabel('sum rate(bandwith = 10) (log scale)')

# add legend
plt.legend()

# save the plot
plt.savefig('Comparison of sum rate of Proposed Transformer of different Sparse Attention')

# display the plot
plt.grid(True)
plt.show()