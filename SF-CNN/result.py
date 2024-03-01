import matplotlib.pyplot as plt

# data
x = [-10, -5, 0, 5, 10, 15, 20]
# CNN, epochs = 200, lr = 0.0001, batch_size = 32
y1 = [0.8902655945595724, 0.5751199399355023, 0.36848606844546833, 0.29410193315144156, 0.2553683132455284, 0.23459051626799857, 0.2303461244047422]

# Transformers(Encoder), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
y2 = [0.030168386176228523, 0.020958859473466873, 0.01619298756122589, 0.013632509857416153, 0.012119762599468231, 0.011332360096275806, 0.01102001965045929]

# Transformers(Encoder + Decoder), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
y3 = [0.030041592195630074, 0.020925084128975868, 0.016060134395956993, 0.01353185623884201, 0.01198374293744564, 0.011237981729209423, 0.010901781730353832]

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

# Transformers(Encoder), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
y2 = [15.578949326993863, 15.66211852964721, 15.704970515352377, 15.727940458993261, 15.741494041206668, 15.748543839189978, 15.751339345320154]

# Transformers(Encoder + Decoder), epochs = 200, lr = 0.0001, batch_size = 32
# reshape type = (Nr, Nt, channel)
y3 = [15.580097663387434, 15.662422590888331, 15.70616330179968, 15.72884259760811, 15.74271209404264, 15.749388603678916, 15.75239740560448]

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
