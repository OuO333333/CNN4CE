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
y3 = [0.039645493030548096, 0.02747759222984314, 0.019949764013290405, 0.016258176416158676, 0.014586728066205978, 0.014074832201004028, 0.01333386730402708]

# DAECNNATT
y4 = [0.5412381862428629, 0.33254079647163465, 0.2115293457419186, 0.1564142778360277, 0.12438449687660279, 0.10890364229279535, 0.10238436697803815]

# SPARSEMATTDAE
y5 = [0.5238966308505519, 0.32426746977979914, 0.20119746725457452, 0.13996944454347623, 0.11280704789072017, 0.09475028820637019, 0.08761300950468075]

# FEDformer
y6 = [0.04003031924366951, 0.032262492924928665, 0.023152129724621773, 0.017859801650047302, 0.01619112305343151, 0.014731141738593578, 0.014243846759200096]

# set y-axis to log scale
plt.yscale('log', base=2)

# draw
plt.plot(x, y1, marker='o', label='CNN')
plt.plot(x, y4, marker='o', label='Attention-aided Autoencoder')
plt.plot(x, y5, marker='o', label='Sparse Autoencoder')
plt.plot(x, y2, marker='o', label='Transformers')
plt.plot(x, y6, marker='o', label='FEDformer')
plt.plot(x, y3, marker='o', label='Proposed Transformer')


# add title and axis labels
plt.title('Comparison of NMSE of CNN and Transformers and Proposed Transformer', fontsize=fontsize)
plt.xlabel('SNR (dB)', fontsize=fontsize)
plt.ylabel('NMSE (log scale)', fontsize=fontsize)

# add legend
plt.legend(fontsize=20)

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
y3 = [15.49264954985823, 15.603153955212745, 15.671097689190924, 15.70430017748037, 15.719308225768199, 15.723901398921354, 15.730547503306035]

# DAECNNATT
y4 = [12.97931951377382, 14.154661877931208, 14.794740768464427, 15.077112879082799, 15.238707802839244, 15.316166877397217, 15.348662296652947]

# SPARSEMATTDAE
y5 = [13.080715119036215, 14.199338913358442, 14.848096915580719, 15.160305529554883, 15.296675207596744, 15.386621664883695, 15.422020653796265]

# FEDformer
y6 = [15.489140933602904, 15.55980019276288, 15.642233280343834, 15.689904380496367, 15.70490247599874, 15.718012085872813, 15.722385011103668]

# set y-axis to log scale
plt.yscale('log', base=10)

# draw
plt.plot(x, y1, marker='o', label='CNN')
plt.plot(x, y4, marker='o', label='Attention-aided Autoencoder')
plt.plot(x, y5, marker='o', label='Sparse Autoencoder')
plt.plot(x, y2, marker='o', label='Transformers')
plt.plot(x, y6, marker='o', label='FEDformer')
plt.plot(x, y3, marker='o', label='Proposed Transformer')

y1 = [value / 10 for value in y1]
y2 = [value / 10 for value in y2]
y3 = [value / 10 for value in y3]
y4 = [value / 10 for value in y4]
y5 = [value / 10 for value in y5]
y6 = [value / 10 for value in y6]

# add title and axis labels
plt.title('Comparison of sum rate of CNN and Transformers and Proposed Transformer', fontsize=fontsize)
plt.xlabel('SNR (dB)', fontsize=fontsize)
plt.ylabel('sum rate(bandwith = 1) (log scale)', fontsize=fontsize)

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
y2 = [0.039645493030548096, 0.02747759222984314, 0.019949764013290405, 0.016258176416158676, 0.014586728066205978, 0.014074832201004028, 0.01333386730402708]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, full attention
# encoder ff layer 用 ffn, decoder ff layer 用 ffn
# modality-aware transformers
y3 = [0.039773717522621155, 0.028461197391152382, 0.020952610298991203, 0.017375558614730835, 0.01585640199482441, 0.014552067965269089, 0.014185205101966858]

# FEDformer
y6 = [0.04003031924366951, 0.032262492924928665, 0.023152129724621773, 0.017859801650047302, 0.01619112305343151, 0.014731141738593578, 0.014243846759200096]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, full attention, no cross attention
# encoder ff layer 用 ffn, decoder ff layer 用 ffn
# modality-aware transformers
y7 = [0.0398753322660923, 0.030808862298727036, 0.022814031690359116, 0.018377229571342468, 0.01656636968255043, 0.014835546724498272, 0.01476007979363203]

# draw
plt.plot(x, y1, marker='o', label='Transformers(Time domain only)')
plt.plot(x, y7, marker='o', label='Proposed Transformer(No cross attention, no spatial)')
plt.plot(x, y3, marker='o', label='Proposed Transformer(No spatial)')
plt.plot(x, y2, marker='o', label='Proposed Transformer')
# plt.plot(x, y6, marker='o', label='FEDformer')

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
y2 = [15.49264954985823, 15.603153955212745, 15.671097689190924, 15.70430017748037, 15.719308225768199, 15.723901398921354, 15.730547503306035]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, full attention
# encoder ff layer 用 ffn, decoder ff layer 用 ffn
# modality-aware transformers
y3 = [15.49148045194048, 15.594252673970859, 15.662064721667459, 15.694258319307453, 15.707909134999468, 15.719619257954932, 15.722911156969142]

# FEDformer
y6 = [15.489140933602904, 15.55980019276288, 15.642233280343834, 15.689904380496367, 15.70490247599874, 15.718012085872813, 15.722385011103668]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, full attention, no cross attention
# encoder ff layer 用 ffn, decoder ff layer 用 ffn
# modality-aware transformers
y7 = [15.490554053541645, 15.572984746486938, 15.645283507862723, 15.685250583857632, 15.701531128715258, 15.717075024595555, 15.717752387349684]

y1 = [value / 10 for value in y1]
y2 = [value / 10 for value in y2]
y3 = [value / 10 for value in y3]
y6 = [value / 10 for value in y6]
y7 = [value / 10 for value in y7]


# draw
plt.plot(x, y1, marker='o', label='Transformers(Time domain only)')
plt.plot(x, y7, marker='o', label='Proposed Transformer(No cross attention, no spatial)')
plt.plot(x, y3, marker='o', label='Proposed Transformer(No spatial)')
plt.plot(x, y2, marker='o', label='Proposed Transformer')
# plt.plot(x, y6, marker='o', label='FEDformer')

# set y-axis to log scale
plt.yscale('log')

# add title and axis labels
plt.title('Comparison of sum rate of Proposed Transformer and Others', fontsize=fontsize)
plt.xlabel('SNR (dB)', fontsize=fontsize)
plt.ylabel('sum rate(bandwith = 1) (log scale)', fontsize=fontsize)

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
y1 = [0.039645493030548096, 0.02747759222984314, 0.019949764013290405, 0.016258176416158676, 0.014586728066205978, 0.014074832201004028, 0.01333386730402708]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, atrous_self_attention,
# encoder ff layer 用 Conv1D, decoder ff layer 用 ffn
# modality-aware transformers
y2 = [0.03944879397749901, 0.026776893064379692, 0.01985848695039749, 0.015884367749094963, 0.013839473016560078, 0.013212382793426514, 0.012837452813982964]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, Local Self Attention
# encoder ff layer 用 Conv1D, decoder ff layer 用 ffn
# modality-aware transformers
y3 = [0.04016707092523575, 0.0282458383589983, 0.020967191085219383, 0.01693073660135269, 0.015475446358323097, 0.014496663585305214, 0.013719967566430569]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, Stride Sparse Self Attention
# encoder ff layer 用 Conv1D, decoder ff layer 用 ffn
# modality-aware transformers
y4 = [0.03975171595811844, 0.027584269642829895, 0.020740732550621033, 0.01648659072816372, 0.014851024374365807, 0.014254416339099407, 0.013560788705945015]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, atrous_self_attention, d = 3
# encoder ff layer 用 Conv1D, decoder ff layer 用 ffn
# modality-aware transformers
y5 = [0.03968484327197075, 0.028611933812499046, 0.022718237712979317, 0.021889904513955116, 0.019011519849300385, 0.016211938112974167, 0.01379705686122179]

# draw
plt.plot(x, y1, marker='o', label='Full Self Attention')
plt.plot(x, y2, marker='o', label='Atrous Self Attention(d=2)')
plt.plot(x, y3, marker='o', label='Local Self Attention')
plt.plot(x, y4, marker='o', label='Stride Sparse Self Attention')
# plt.plot(x, y5, marker='o', label='Atrous Self Attention(d=3)')


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
y1 = [15.49264954985823, 15.603153955212745, 15.671097689190924, 15.70430017748037, 15.719308225768199, 15.723901398921354, 15.730547503306035]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, atrous_self_attention,
# encoder ff layer 用 Conv1D, decoder ff layer 用 ffn
# modality-aware transformers
y2 = [15.494442592010083, 15.609491910438882, 15.671919573771865, 15.707657914006779, 15.726012765282944, 15.731636832092104, 15.734998353604407]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, Local Self Attention
# encoder ff layer 用 Conv1D, decoder ff layer 用 ffn
# modality-aware transformers
y3 = [15.487893916991833, 15.59620204858171, 15.661933478931305, 15.698256769090925, 15.71133025985971, 15.720116450547517, 15.727084762060883]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, Stride Sparse Self Attention
# encoder ff layer 用 Conv1D, decoder ff layer 用 ffn
# modality-aware transformers
y4 = [15.491681067760004, 15.602188908767813, 15.663973687062441, 15.702248021962497, 15.716936093266758, 15.722290191751824, 15.728512427221728]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, atrous_self_attention, d = 3
# encoder ff layer 用 Conv1D, decoder ff layer 用 ffn
# modality-aware transformers
y5 = [15.49229083432177, 15.592888048108476, 15.646147580348586, 15.653617242610963, 15.679543601852913, 15.704715510298538, 15.726393267161576]


y1 = [value / 10 for value in y1]
y2 = [value / 10 for value in y2]
y3 = [value / 10 for value in y3]
y4 = [value / 10 for value in y4]
y5 = [value / 10 for value in y5]

# draw
plt.plot(x, y1, marker='o', label='Full Self Attention')
plt.plot(x, y2, marker='o', label='Atrous Self Attention')
plt.plot(x, y3, marker='o', label='Local Self Attention')
plt.plot(x, y4, marker='o', label='Stride Sparse Self Attention')
# plt.plot(x, y5, marker='o', label='Atrous Self Attention(d=3)')

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