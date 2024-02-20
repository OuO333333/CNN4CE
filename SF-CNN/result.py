import matplotlib.pyplot as plt

# data
x = [-10, -5, 0, 5, 10, 15, 20]
# CNN, epochs = 200, lr = 0.0001, batch_size = 32
y1 = [0.8902655945595724, 0.5751199399355023, 0.36848606844546833, 0.29410193315144156, 0.2553683132455284, 0.23459051626799857, 0.2303461244047422]

# Transformers(Encoder, new version, (32 * 16 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
y2 = [1.005581103661059, 1.023077848010152, 1.0091059088906147, 1.0158015911275906, 0.13586349671622336, 0.0907570120291898, 0.052233937396309695]

# Transformers(Encoder, new version, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
y3 = [1.0034997822283107, 1.0022078917963781, 0.42036374677900773, 0.24588232616118508, 0.13356168133422286, 0.08693595202165977, 0.055574757864449695]

# Transformers(Encoder, new version, (4 * 16 * 32)), epochs = 200, lr = 0.0001, batch_size = 32
y4 = [1.0056178096656145, 1.0051167132633074, 1.0047594643512026, 1.0135836603921828, 1.0049115998872755, 0.14973634341748948, 0.10742040065425469]
# draw
plt.plot(x, y1, marker='o', label='CNN')
plt.plot(x, y2, marker='o', label='Transformers(Encoder), (Nt * Nr * channel)')
plt.plot(x, y3, marker='o', label='Transformers(Encoder), (Nr * Nt * channel)')
plt.plot(x, y4, marker='o', label='Transformers(Encoder), (channel * Nr * Nt)')

# add title and axis labels
plt.title('Comparison of CNN and Transformers(Encoder)')
plt.xlabel('SNR (dB)')
plt.ylabel('NMSE')

# add legend
plt.legend()

# save the plot
plt.savefig('Comparison of CNN and Transformers(Encoder)')

# display the plot
plt.grid(True)
plt.show()

# others data

####################################################################################################################

# data
x = [-10, -5, 0, 5, 10, 15, 20]
# CNN, epochs = 200, lr = 0.0001, batch_size = 32
y1 = [0.8902655945595724, 0.5751199399355023, 0.36848606844546833, 0.29410193315144156, 0.2553683132455284, 0.23459051626799857, 0.2303461244047422]

# Transformers(Encoder, new version, with positional encoder, (32 * 16 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
y2 = [0.030933775007724762, 0.021904414519667625, 0.016832156106829643, 0.013988425023853779, 0.012418906204402447, 0.011840230785310268, 0.011266968213021755]

# Transformers(Encoder, new version, with positional encoder, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
y3 = [0.030377166345715523, 0.020921865478157997, 0.01644887961447239, 0.013676438480615616, 0.012209874577820301, 0.01129317656159401, 0.011004100553691387]

# Transformers(Encoder, new version, with positional encoder, (4 * 16 * 32)), epochs = 200, lr = 0.0001, batch_size = 32
y4 = [0.035654228180646896, 0.025894038379192352, 0.01961899921298027, 0.01625351794064045, 0.0141898263245821, 0.013078039512038231, 0.012580392882227898]
# draw
plt.plot(x, y1, marker='o', label='CNN')
plt.plot(x, y2, marker='o', label='Transformers(Encoder), (Nt * Nr * channel)')
plt.plot(x, y3, marker='o', label='Transformers(Encoder), (Nr * Nt * channel)')
plt.plot(x, y4, marker='o', label='Transformers(Encoder), (channel * Nr * Nt)')

# add title and axis labels
plt.title('Comparison of CNN and Transformers(Encoder with positional encoding)')
plt.xlabel('SNR (dB)')
plt.ylabel('NMSE')

# add legend
plt.legend()

# save the plot
plt.savefig('Comparison of CNN and Transformers(Encoder with positional encoding)')

# display the plot
plt.grid(True)
plt.show()

####################################################################################################################

# data
x = [-10, -5, 0, 5, 10, 15, 20]

# draw

plt.plot(x, y2, marker='o', label='Transformers(Encoder), (Nt * Nr * channel)')
plt.plot(x, y3, marker='o', label='Transformers(Encoder), (Nr * Nt * channel)')
plt.plot(x, y4, marker='o', label='Transformers(Encoder), (channel * Nr * Nt)')

# add title and axis labels
plt.title('Transformers(Encoder with positional encoding)')
plt.xlabel('SNR (dB)')
plt.ylabel('NMSE')

# add legend
plt.legend()

# save the plot
plt.savefig('Transformers(Encoder with positional encoding)')

# display the plot
plt.grid(True)
plt.show()
