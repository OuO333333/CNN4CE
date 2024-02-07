# CNN4CE
This code is modify from https://github.com/phdong21/CNN4CE.  
https://github.com/phdong21/CNN4CE/blob/main/SF-CNN/SF_CNN_2fre_train.py  
SF_CNN_2fre_train.py 是要改模型的地方, 125 - 156 行。
原版的是用 CNN 做矩陣的估計,  
前面的 src, trg 怎麽生出來的可以不用知道,  
知道這個 CNN 的 src 是 H_train_noisy, trg 是 H_train 就好, 這兩個都是 16 * 32 * 4 的矩陣, 各有 1000 個。  
loss function是用 MSE,  
最後測試模型時是將 H_test_noisy 帶入模型估計出 ~H_test_noisy,  
~H_test_noisy 與 H_test 即可計算出 NMSE。
