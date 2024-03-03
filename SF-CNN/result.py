import matplotlib.pyplot as plt

# data
x = [-10, -5, 0, 5, 10, 15, 20]
# CNN, epochs = 200, lr = 0.0001, batch_size = 32
y1 = [0.8902655945595724, 0.5751199399355023, 0.36848606844546833, 0.29410193315144156, 0.2553683132455284, 0.23459051626799857, 0.2303461244047422]

# Transformers(Encoder * 9), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
y2 = [0.03453879430890083, 0.026762498542666435, 0.021355926990509033, 0.01854751817882061, 0.016703907400369644, 0.015434942208230495, 0.013476668857038021]

# Transformers(Encoder * 4 + Decoder * 4), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention, Decoder 有 mask
y3 = [0.0341397225856781, 0.025831393897533417, 0.0199327003210783, 0.01462156418710947, 0.014521920122206211, 0.012782293371856213, 0.012219314463436604]

# [-10: O, -5: X, 0: O, 5: O, 10:O , 15: O, 20: O]
# draw
plt.plot(x, y1, marker='o', label='CNN')
plt.plot(x, y2, marker='o', label='Transformers(Encoder)')
plt.plot(x, y3, marker='o', label='Transformers(Encoder + Decoder)')

# add title and axis labels
plt.title('Comparison NMSE of CNN and Transformers(Encoder) and Transformers(Encoder + Decoder)')
plt.xlabel('SNR (dB)')
plt.ylabel('NMSE')

# add legend
plt.legend()

# save the plot
plt.savefig('Comparison NMSE of CNN and Transformers(Encoder) and Transformers(Encoder + Decoder)')

# display the plot
plt.grid(True)
plt.show()
####################################################################################################################
# data
x = [-10, -5, 0, 5, 10, 15, 20]
# CNN, epochs = 200, lr = 0.0001, batch_size = 32
y1 = [12.940970626378812, 13.972075582297652, 14.508584726671309, 14.825802753405233, 14.9719951899972, 15.106508174427415, 15.093083672359931]

# Transformers(Encoder * 9), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention
y2 = [15.539312784548422, 15.609762987385716, 15.658542552063942, 15.68381586680459, 15.700382733433553, 15.711774756083067, 15.729337311794527]

# Transformers(Encoder * 4 + Decoder * 4), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
# 自己寫的 multi-head attention, Decoder 有 mask
y3 = [15.542936558705353, 15.618175592967326, 15.671355988000355, 15.719071981813382, 15.719965706932149, 15.735559637133356, 15.740602518871455]

# [-10: O, -5: X, 0: O, 5: O, 10:O , 15: O, 20: O]
# draw
plt.plot(x, y1, marker='o', label='CNN')
plt.plot(x, y2, marker='o', label='Transformers(Encoder)')
plt.plot(x, y3, marker='o', label='Transformers(Encoder + Decoder)')

# add title and axis labels
plt.title('Comparison sum rate of CNN and Transformers(Encoder) and Transformers(Encoder + Decoder)')
plt.xlabel('SNR (dB)')
plt.ylabel('sum rate(bandwith = 10)')

# add legend
plt.legend()

# save the plot
plt.savefig('Comparison sum rate of CNN and Transformers(Encoder) and Transformers(Encoder + Decoder)')

# display the plot
plt.grid(True)
plt.show()