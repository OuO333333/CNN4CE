####################################################################################################################

# data
x = [-10, -5, 0, 5, 10, 15, 20]
# CNN, epochs = 200, lr = 0.0001, batch_size = 32
y1 = [15.8902655945595724, 15.5751199399355023, 15.36848606844546833, 15.29410193315144156, 15.2553683132455284, 15.23459051626799857, 15.2303461244047422]

# Transformers(Encoder, (32 * 16 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
y2 = [15.030933775007724762, 15.021904414519667625, 15.016832156106829643, 15.013988425023853779, 15.012418906204402447, 15.011840230785310268, 15.011266968213021755]

# Transformers(Encoder, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
y3 = [15.030377166345715523, 15.020921865478157997, 15.01644887961447239, 15.013676438480615616, 15.012209874577820301, 15.01129317656159401, 15.011004100553691387]

# Transformers(Encoder, (4 * 16 * 32)), epochs = 200, lr = 0.0001, batch_size = 32
y4 = [15.035654228180646896, 15.025894038379192352, 15.01961899921298027, 15.01625351794064045, 15.0141898263245821, 15.013078039512038231, 15.012580392882227898]

# Transformers(Encoder, originnal shape), epochs = 200, lr = 0.0001, batch_size = 32
y5 = [15.11000585556030273, 15.05076548457145691, 15.04574974998831749, 15.02721697837114334, 15.02782992459833622, 15.028809938579797745, 15.022801632061600685]

# draw
plt.plot(x, y1, marker='o', label='CNN')
plt.plot(x, y2, marker='o', label='Transformers(Encoder), (Nt * Nr * channel)')
plt.plot(x, y3, marker='o', label='Transformers(Encoder), (Nr * Nt * channel)')
plt.plot(x, y4, marker='o', label='Transformers(Encoder), (channel * Nr * Nt)')
plt.plot(x, y5, marker='o', label='Transformers(Encoder), original shape')

# add title and axis labels
plt.title('Comparison of CNN and Transformers(Encoder) with different reshape method')
plt.xlabel('SNR (dB)')
plt.ylabel('Sum rate(bandwith = 10)')

# add legend
plt.legend()

# save the plot
plt.savefig('Comparison of CNN and Transformers(Encoder) with different reshape method')

# display the plot
plt.grid(True)
plt.show()
####################################################################################################################

# data
x = [-10, -5, 0, 5, 10, 15, 20]

# Transformers(Encoder * 4 + Decoder * 4, (32 * 16 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
y6 = [15.575649165235863, 15.657152143870377, 15.701397482164204, 15.7260508584098, 15.73853903358405, 15.744997238517547, 15.749720161849886]

# Transformers(Encoder * 4 + Decoder * 4, (16 * 32 * 4)), epochs = 200, lr = 0.0001, batch_size = 32
y7 = [15.580097663387434, 15.662422590888331, 15.70616330179968, 15.72884259760811, 15.74271209404264, 15.749388603678916, 15.750669380795564]

# Transformers(Encoder * 4 + Decoder * 4, (4 * 16 * 32)), epochs = 200, lr = 0.0001, batch_size = 32
y8 = [15.506681251466436, 15.607668816520237, 15.019532108679413795, 15.016106395050883293, 15.01433786004781723, 15.01295443531125784, 15.012212691828608513]

# draw
plt.plot(x, y4, marker='o', label='Transformers(Encoder), (channel * Nr * Nt)', color='green')
plt.plot(x, y8, marker='o', label='Transformers(Encoder + Decoder), (channel * Nr * Nt)', color=(127/255, 255/255, 0))
plt.plot(x, y2, marker='o', label='Transformers(Encoder), (Nt * Nr * channel)', color='blue')
plt.plot(x, y6, marker='o', label='Transformers(Encoder + Decoder), (Nt * Nr * channel)', color=(102/255, 178/255, 255/255))
plt.plot(x, y3, marker='o', label='Transformers(Encoder), (Nr * Nt * channel)', color='red')
plt.plot(x, y7, marker='o', label='Transformers(Encoder + Decoder), (Nr * Nt * channel)', color=(255/255, 102/255, 102/255))
# add title and axis labels
plt.title('Comparison of Transformers(Encoder + Decoder)  and Transformers(Encoder)')
plt.xlabel('SNR (dB)')
plt.ylabel('Sum rate(bandwith = 10)')

# add legend
plt.legend()

# save the plot
plt.savefig('Comparison of Transformers(Encoder + Decoder)  and Transformers(Encoder)')

# display the plot
plt.grid(True)
plt.show()
