### 1.模型性能

​	We got 91.83135705 scores on the prediction results for 1day

​	We got 56.52173913 scores on the prediction results for 20day

​	We got 59.55204216 scores on the prediction results for 60day

### 2.代码环境

​	OS：Windows10

​	Env： Cuda10.1   cuDNN7.6   GTX1080Ti 

​	Driver： NVIDIA Driver Version 441.22

​	Lib： tensorflow-gpu-2.0.0 + keras2.2.4-tf + pandas0.25.3 + sklearn0.21.3 + numpy1.17.4

### 3.模型描述 

​	Based on the historical futures trading data, this model makes a regression forecast of the future futures closing price, so as to calculate the future rise and fall of futures based on the difference of the closing price.

​	The model is divided into the following three parts:

1. **Preprocessing section**

   In our program, seven features of 'open. Price','close. Price', 'Volume', 'SPX', 'SX5E', 'UKX' and 'VIX' were selected in light of a simple analysis. Then we combined the training data and the test data, which aims to provide the 'close. Price' feature of the test data on the first day for the training data and facilitate subsequent processing. And our procedure ensures that there will be no data leak when feeding the data into the model.

   According to our evaluation, we choose to use the data only before 01-04-2011 which will not reduce the performance of our model.

   Then we used MinMaxScaler to normalize the test data, and took the historical information of the first 20 days of each day as the training data of the day to predict the future. The 20 days were the better parameters obtained by referring to the trading rules in the Way of the turtle.

   Finally, the training data should maintain the shape of (n, 20,7) into the model.

2. **Training section**

   We use LSTM as the main part of the model, in which 128 units are set up in the LSTM layer and the BN layer is added in the middle of the LSTM. Finally, a Dense layer is used to generate a predicted value of the future closing price every day. The data shape is (n,1).

   We chose Adam for the optimizer and MSE for the loss function.

3. **Testing section**

   In order to avoid the phenomenon of data leak, during the test, we first add data and historical data merge one by one, then input it into LSTM for prediction, and output the predicted results one by one to add them to the historical data.

### 4.测试所用时间

​	Prediction for 1day costs 230 seconds on average.

​	Prediction for 20day costs 232 seconds on average.

​	Prediction for 60day costs 597 seconds on average.

### 5.联系方式

​	851022445@qq.com

​	1498793919@qq.com

​	stevezhang1999@126.com

​	