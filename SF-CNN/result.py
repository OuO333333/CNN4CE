import matplotlib.pyplot as plt

# data
x = [-10, -5, 0, 5, 10, 15, 20]
# CNN, epochs = 200, lr = 0.0001, batch_size = 32
y1 = [0.810673524905295, 0.4919532945371677, 0.2643258475639457, 0.16703036852743622, 0.11317730937141791, 0.09042117733683336, 0.07431391089906239]
# Transformers(Encoder), epochs = 200, lr = 0.0001, batch_size = 32
y2 = [0.9228426684052887, 0.6831239491929525, 0.2941315855551049, 0.14006600876597874, 0.1021696856387804, 0.06976020783386723, 0.04713466330402714]

# draw
plt.plot(x, y1, marker='o', label='CNN')
plt.plot(x, y2, marker='o', label='Transformers(Encoder)')


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
# CNN, epochs = 200, lr = 0.0001, batch_size = 128
y3 = [0.8902655945595724, 0.5751199399355023, 0.36848606844546833, 0.29410193315144156, 0.2553683132455284, 0.23459051626799857, 0.2303461244047422]
# Transformers(Encoder), epochs = 40, lr = 0.0001, batch_size = 32
y4 = [0.9770831849262505, 0.7012182835272307, 0.49189052603046274, 0.39420415368532674, 0.23643466031546911, 0.19068080361620274, 0.14560928882392193]
