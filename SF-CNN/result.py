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
