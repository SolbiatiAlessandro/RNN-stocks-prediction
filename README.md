# RNN-stock-prediction

Another attempt to use Deep-Leaning in the financial markets, the project is documented
here -> https://solbiatialessandro.github.io/RNN-stocks-prediction/ 

- basic -

If you wanna run the <b>regression model</b> (implemented with LSTM) just run regression.py

If you wanna take a look at the code you can run the <b>framework</b> with a Jupyter Notebook

- advanced -

If you wanna predict stock prices [see docs] I uploaded a model (.hdf5) working at 63% accuracy with GOOGL, you just need to upload it in KERAS and launch it:

  model.load_weights()
  pred = model.predict())
