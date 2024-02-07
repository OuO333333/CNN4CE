# CNN4CE
This code is modify from https://github.com/phdong21/CNN4CE.  
https://github.com/phdong21/CNN4CE/blob/main/SF-CNN/SF_CNN_2fre_train.py  
SF_CNN_2fre_train.py 是要改模型的地方, 125 - 156 行。  
原版的是用 CNN 做矩陣的估計,  
前面的 src, trg 怎麽生出來的可以不用知道,  
知道這個 CNN 的 src 是 H_train_noisy, trg 是 H_train 就好, 這兩個都是 16 * 32 * 4 的矩陣, 各有 1000 個。  
矩陣的特徵是數字不大, 大約介於 -2 ~ 2, 有小數點,  
loss function是用 MSE,  
最後測試模型時是將 H_test_noisy 帶入模型估計出 ~H_test_noisy,  
~H_test_noisy 與 H_test 即可計算出 NMSE。  

要把 CNN 的部份(125 - 156 行)替換成 transformer,  
src, trg 必須為 16 * 32 * 4 的矩陣,  
loss function 必須使用 MSE,  
要自己刻 transformer, 可以用網路上別人刻好的來移植,  
transformer 需容易修改其架構。  
transformer 的結果必須比 CNN 好, 以比較 NMSE 為準。
