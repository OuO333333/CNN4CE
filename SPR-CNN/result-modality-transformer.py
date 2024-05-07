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
y2 = [0.040231116116046906, 0.030061252415180206, 0.02286352589726448, 0.018579063937067986, 0.016204895451664925, 0.015272018499672413, 0.014378390274941921]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, full attention
# encoder ff layer 用 Conv1D, decoder ff layer 用 ffn
# modality-aware transformers
y3 = [0.039645493030548096, 0.02848387509584427, 0.02177749201655388, 0.017665809020400047, 0.015258733183145523, 0.01451767235994339, 0.013874311000108719]

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
plt.plot(x, y4, marker='o', label='Attention-aided Auto-Encoder')
plt.plot(x, y5, marker='o', label='Sparse Auto-Encoder')
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
y2 = [15.487309683281385, 15.579760882190199, 15.64483701235907, 15.68343480874185, 15.70477877832414, 15.713156822752882, 15.721177762372365]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, full attention
# encoder ff layer 用 Conv1D, decoder ff layer 用 ffn
# modality-aware transformers
y3 = [15.49264954985823, 15.594047444695512, 15.65463062254004, 15.691648785427166, 15.713276056208183, 15.719927910290641, 15.725700256494086]

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
plt.plot(x, y4, marker='o', label='Attention-aided Auto-Encoder')
plt.plot(x, y5, marker='o', label='Sparse Auto-Encoder')
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
# 比例值
values = [24.937631607055664, 34.374982595443726, 50.404070138931274, 233.8608410358429, 413.7626326084137, 265.59384655952454]
labels = ['CNN', 'Attention-aided Auto-Encoder', 'Sparse Auto-Encoder', 'Transformers', 'FEDformer', 'Proposed Transformer']

# 创建条形图
plt.bar(labels, values)

plt.xticks(fontsize=12)  # 调整x轴标签字体大小

# 添加标题和标签
plt.title('Comparison of Training Time', fontsize=fontsize)
plt.xlabel('Categories', fontsize=fontsize)
plt.ylabel('Time(s)', fontsize=fontsize)

# save the plot
plt.savefig('Comparison of Training Time')

# 显示图形
plt.show()
####################################################################################################################

# data

# Transformers(Encoder * 2 + Decoder * 2), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
y1 = [0.040231116116046906, 0.030061252415180206, 0.02286352589726448, 0.018579063937067986, 0.016204895451664925, 0.015272018499672413, 0.014378390274941921]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, full attention
# encoder ff layer 用 Conv1D, decoder ff layer 用 ffn
# modality-aware transformers
y2 = [0.039645493030548096, 0.02848387509584427, 0.02177749201655388, 0.017665809020400047, 0.015258733183145523, 0.01451767235994339, 0.013874311000108719]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, full attention
# encoder ff layer 用 ffn, decoder ff layer 用 ffn
# modality-aware transformers
y3 = [0.04003896936774254, 0.02920401655137539, 0.021999454125761986, 0.017966583371162415, 0.01563282497227192, 0.01466186810284853, 0.014185205101966858]

# FEDformer
y6 = [0.04003031924366951, 0.032262492924928665, 0.023152129724621773, 0.017859801650047302, 0.01619112305343151, 0.014731141738593578, 0.014243846759200096]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, full attention, no cross attention
# encoder ff layer 用 ffn, decoder ff layer 用 ffn
# modality-aware transformers
y7 = [0.040216702967882156, 0.030808862298727036, 0.02333066239953041, 0.019309449940919876, 0.01656636968255043, 0.01590658724308014, 0.01476007979363203]

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
y1 = [15.487309683281385, 15.579760882190199, 15.64483701235907, 15.68343480874185, 15.70477877832414, 15.713156822752882, 15.721177762372365]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, full attention
# encoder ff layer 用 Conv1D, decoder ff layer 用 ffn
# modality-aware transformers
y2 = [15.49264954985823, 15.594047444695512, 15.65463062254004, 15.691648785427166, 15.713276056208183, 15.719927910290641, 15.725700256494086]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, full attention
# encoder ff layer 用 ffn, decoder ff layer 用 ffn
# modality-aware transformers
y3 = [15.48906192608235, 15.587526682731204, 15.652629488346413, 15.688944079251307, 15.709916997359823, 15.718633820350675, 15.722911156969142]

# FEDformer
y6 = [15.489140933602904, 15.55980019276288, 15.642233280343834, 15.689904380496367, 15.70490247599874, 15.718012085872813, 15.722385011103668]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, full attention, no cross attention
# encoder ff layer 用 ffn, decoder ff layer 用 ffn
# modality-aware transformers
y7 = [15.487441121434859, 15.572984746486938, 15.6406224579517, 15.676862212688121, 15.701531128715258, 15.707458382273852, 15.717752387349684]

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
values = [1.0, 0.5, 7/16, 4/16]
labels = ['Full Self Attention', 'Atrous Self Attention', 'Local Self Attention', 'Atrous Local Self Attention']

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
y1 = [0.039645493030548096, 0.02848387509584427, 0.02177749201655388, 0.017665809020400047, 0.015258733183145523, 0.01451767235994339, 0.013874311000108719]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, atrous_self_attention,
# encoder ff layer 用 Conv1D, decoder ff layer 用 ffn
# modality-aware transformers
y2 = [0.038145072758197784, 0.027047526091337204, 0.02087853103876114, 0.016779784113168716, 0.014481263235211372, 0.01399294938892126, 0.01356890331953764]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, Local Self Attention
# encoder ff layer 用 Conv1D, decoder ff layer 用 ffn
# modality-aware transformers
y3 = [0.039379578083753586, 0.028154458850622177, 0.021071387454867363, 0.017013927921652794, 0.014845786616206169, 0.014248121529817581, 0.0137781398370862]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, atrous_and_local_self_attention, d = 2, windows = 6
# encoder ff layer 用 Conv1D, decoder ff layer 用 ffn
# modality-aware transformers
y4 = [0.03719576820731163, 0.02675441838800907, 0.02032649889588356, 0.016275744885206223, 0.014112374745309353, 0.01363054383546114, 0.013208819553256035]

# draw
plt.plot(x, y1, marker='o', label='Full Self Attention')
plt.plot(x, y2, marker='o', label='Atrous Self Attention')
plt.plot(x, y3, marker='o', label='Local Self Attention')
plt.plot(x, y4, marker='o', label='Atrous Local Self Attention')


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
y1 = [15.49264954985823, 15.594047444695512, 15.65463062254004, 15.691648785427166, 15.713276056208183, 15.719927910290641, 15.725700256494086]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, atrous_self_attention,
# encoder ff layer 用 Conv1D, decoder ff layer 用 ffn
# modality-aware transformers
y2 = [15.506321704157296, 15.60704432309743, 15.662732238044741, 15.6996133959397, 15.720254642621406, 15.724635949414886, 15.728439669153394]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, Local Self Attention
# encoder ff layer 用 Conv1D, decoder ff layer 用 ffn
# modality-aware transformers
y3 = [15.495073466994242, 15.597029174869308, 15.660994609097447, 15.697509075343214, 15.716983071441868, 15.722346618774578, 15.726562977136382]

# Transformers(Encoder * 2 + Decoder * 2, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
# 自己寫的 multi-head self attention, atrous_and_local_self_attention, d = 2, windows = 6
# encoder ff layer 用 Conv1D, decoder ff layer 用 ffn
# modality-aware transformers
y4 = [15.514965219088664, 15.609695093405547, 15.667704968574576, 15.704142340486547, 15.723564602694648, 15.727886811139943, 15.7316688188335]


y1 = [value / 10 for value in y1]
y2 = [value / 10 for value in y2]
y3 = [value / 10 for value in y3]
y4 = [value / 10 for value in y4]

# draw
plt.plot(x, y1, marker='o', label='Full Self Attention')
plt.plot(x, y2, marker='o', label='Atrous Self Attention')
plt.plot(x, y3, marker='o', label='Local Self Attention')
plt.plot(x, y4, marker='o', label='Atrous Local Self Attention')


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
####################################################################################################################

# 比例值
values = [0.49264954985823, 0.016927558928728104, 0.014378390274941921, 0.019694823771715164, 0.028786765411496162]
labels = ['8', '16', '32', '64', '128']

# 创建条形图
plt.plot(labels, values)

plt.xticks(fontsize=20)  # 调整x轴标签字体大小

# 添加标题和标签
plt.title('Comparison of NMSE of different Element Numbers(Nr = 16)', fontsize=fontsize)
plt.xlabel('Element Numbers', fontsize=fontsize)
plt.ylabel('NMSE', fontsize=fontsize)

# save the plot
plt.savefig('Comparison of NMSE of different Element Numbers(Nr = 16)')

# 显示图形
plt.show()