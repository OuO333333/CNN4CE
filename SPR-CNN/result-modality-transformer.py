import matplotlib.pyplot as plt
import math

fontsize = 25

# data
x = [-10, -5, 0, 5, 10, 15, 20]
# CNN, epochs = 200, lr = 0.0001, batch_size = 32
y1 = [0.6419357114932095, 0.3948697806398535, 0.2653386565881674, 0.17596123561662536, 0.13396011468502073, 0.11912793777809844, 0.11196449103168699]

# Transformers(Encoder * 2 + Decoder * 2), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
y2 = [0.03987005725502968, 0.029254980385303497, 0.02159520797431469, 0.018056659027934074, 0.016204895451664925, 0.01474946178495884, 0.014378390274941921]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, full attention
# encoder ff layer 用 Conv1D, decoder ff layer 用 ffn
# modality-aware transformers
y3 = [0.039645493030548096, 0.02747759222984314, 0.019949764013290405, 0.016916004940867424, 0.014825291931629181, 0.013869925402104855, 0.01333386730402708]

# DAECNNATT
y4 = [0.5412381862428629, 0.33254079647163465, 0.2115293457419186, 0.1564142778360277, 0.12438449687660279, 0.10890364229279535, 0.10238436697803815]

# SPARSEMATTDAE
y5 = [0.5238966308505519, 0.32426746977979914, 0.20119746725457452, 0.13996944454347623, 0.11280704789072017, 0.09475028820637019, 0.08761300950468075]

# set y-axis to log scale
plt.yscale('log', base=2)

# draw
plt.plot(x, y1, marker='o', label='CNN')
plt.plot(x, y2, marker='o', label='Transformers')
plt.plot(x, y3, marker='o', label='Proposed Transformer')
plt.plot(x, y4, marker='o', label='DAECNNATT')
plt.plot(x, y5, marker='o', label='SPARSEMATTDAE')

# add title and axis labels
plt.title('Comparison of NMSE of CNN and Transformers and Proposed Transformer', fontsize=fontsize)
plt.xlabel('SNR (dB)', fontsize=fontsize)
plt.ylabel('NMSE (log scale)', fontsize=fontsize)

# add legend
plt.legend(fontsize=fontsize)

# save the plot
plt.savefig('Comparison of NMSE of CNN and Transformers and Proposed Transformer')

# display the plot
plt.grid(True)
plt.show()
####################################################################################################################
# data
x = [-10, -5, 0, 5, 10, 15, 20]
# CNN, epochs = 200, lr = 0.0001, batch_size = 32
y1 = [12.376030106733767, 13.813554657995969, 14.51362168247246, 14.97759882683261, 15.190586800457142, 15.265055865213462, 15.300884747531022]

# Transformers(Encoder * 2 + Decoder * 2), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
y2 = [15.490602115597438, 15.58706522935487, 15.656273772156585, 15.688134016492416, 15.70477877832414, 15.71784768980056, 15.721177762372365]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, full attention
# encoder ff layer 用 Conv1D, decoder ff layer 用 ffn
# modality-aware transformers
y3 = [15.49264954985823, 15.603153955212745, 15.671097689190924, 15.698389138207103, 15.717167092546497, 15.725739653022522, 15.730547503306035]

# DAECNNATT
y4 = [12.97931951377382, 14.154661877931208, 14.794740768464427, 15.077112879082799, 15.238707802839244, 15.316166877397217, 15.348662296652947]

# SPARSEMATTDAE
y5 = [13.080715119036215, 14.199338913358442, 14.848096915580719, 15.160305529554883, 15.296675207596744, 15.386621664883695, 15.422020653796265]

# set y-axis to log scale
plt.yscale('log', base=10)

# draw
plt.plot(x, y1, marker='o', label='CNN')
plt.plot(x, y2, marker='o', label='Transformers')
plt.plot(x, y3, marker='o', label='Proposed Transformer')
plt.plot(x, y4, marker='o', label='DAECNNATT')
plt.plot(x, y5, marker='o', label='SPARSEMATTDAE')

# add title and axis labels
plt.title('Comparison of sum rate of CNN and Transformers and Proposed Transformer', fontsize=fontsize)
plt.xlabel('SNR (dB)', fontsize=fontsize)
plt.ylabel('sum rate(bandwith = 10) (log scale)', fontsize=fontsize)

# add legend
plt.legend(fontsize=fontsize)

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
y1 = [0.03987005725502968, 0.029254980385303497, 0.02159520797431469, 0.018056659027934074, 0.016204895451664925, 0.01474946178495884, 0.014378390274941921]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, full attention
# encoder ff layer 用 Conv1D, decoder ff layer 用 ffn
# modality-aware transformers
y2 = [0.039645493030548096, 0.02747759222984314, 0.019949764013290405, 0.016916004940867424, 0.014825291931629181, 0.013869925402104855, 0.01333386730402708]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, full attention
# encoder ff layer 用 ffn, decoder ff layer 用 ffn
# modality-aware transformers
y3 = [0.039773717522621155, 0.028461197391152382, 0.020952610298991203, 0.017375558614730835, 0.01585640199482441, 0.014552067965269089, 0.014185205101966858]

# draw
plt.plot(x, y1, marker='o', label='Transformers(time domain only)')
plt.plot(x, y2, marker='o', label='Proposed Transformer')
plt.plot(x, y3, marker='o', label='Proposed Transformer(time domain + frequency domain)')

# set y-axis to log scale
plt.yscale('log', base=2)

# add title and axis labels
plt.title('Comparison of NMSE of Proposed Transformer and Others', fontsize=fontsize)
plt.xlabel('SNR (dB)', fontsize=fontsize)
plt.ylabel('NMSE (log scale)', fontsize=fontsize)

# add legend
plt.legend(fontsize=fontsize)

# save the plot
plt.savefig('Comparison of NMSE of Proposed Transformer and Others')

# display the plot
plt.grid(True)
plt.show()
####################################################################################################################
# data
x = [-10, -5, 0, 5, 10, 15, 20]

# Transformers(Encoder * 2 + Decoder * 2), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
y1 = [15.490602115597438, 15.58706522935487, 15.656273772156585, 15.688134016492416, 15.70477877832414, 15.71784768980056, 15.721177762372365]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, full attention
# encoder ff layer 用 Conv1D, decoder ff layer 用 ffn
# modality-aware transformers
y2 = [15.49264954985823, 15.603153955212745, 15.671097689190924, 15.698389138207103, 15.717167092546497, 15.725739653022522, 15.730547503306035]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, full attention
# encoder ff layer 用 ffn, decoder ff layer 用 ffn
# modality-aware transformers
y3 = [15.49148045194048, 15.594252673970859, 15.662064721667459, 15.694258319307453, 15.707909134999468, 15.719619257954932, 15.722911156969142]

# draw
plt.plot(x, y1, marker='o', label='Transformers(time domain only)')
plt.plot(x, y2, marker='o', label='Proposed Transformer')
plt.plot(x, y3, marker='o', label='Proposed Transformer(time domain + frequency domain)')


# set y-axis to log scale
plt.yscale('log', base=2)

# add title and axis labels
plt.title('Comparison of sum rate of Proposed Transformer and Others', fontsize=fontsize)
plt.xlabel('SNR (dB)', fontsize=fontsize)
plt.ylabel('sum rate(bandwith = 10) (log scale)', fontsize=fontsize)

# add legend
plt.legend(fontsize=fontsize)

# save the plot
plt.savefig('Comparison of sum rate of Proposed Transformer and Others')

# display the plot
plt.grid(True)
plt.show()
####################################################################################################################
# 比例值
values = [1.0, 0.5, 125/1024, 574/1024]
labels = ['Full Self Attention', 'Atrous Self Attention', 'Local Self Attention', 'Stride Sparse Self Attention']

# 创建条形图
plt.bar(labels, values)

plt.xticks(fontsize=20)  # 调整x轴标签字体大小

# 添加标题和标签
plt.title('Comparison of calculation loading ratio of different Sparse Attention and Full Self Attention', fontsize=fontsize)
plt.xlabel('Categories', fontsize=fontsize)
plt.ylabel('Proportion', fontsize=fontsize)

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
y1 = [0.039645493030548096, 0.02747759222984314, 0.019949764013290405, 0.016916004940867424, 0.014825291931629181, 0.013869925402104855, 0.01333386730402708]

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
plt.title('Comparison of sum rate of Proposed Transformer of different Sparse Attention', fontsize=fontsize)
plt.xlabel('SNR (dB)', fontsize=fontsize)
plt.ylabel('sum rate(bandwith = 10) (log scale)', fontsize=fontsize)

# add legend
plt.legend(fontsize=fontsize)

# save the plot
plt.savefig('Comparison of sum rate of Proposed Transformer of different Sparse Attention')

# display the plot
plt.grid(True)
plt.show()