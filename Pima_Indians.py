from keras.models import Sequential
from keras.layers import Dense
import pandas as pd

dataset = pd.read_csv("pima_indians_diabetes.csv")
# split into input (X) and output (Y) variables
X = dataset.drop('Outcome', axis=1)
Y = dataset['Outcome'].values

# create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, epochs=150, batch_size=10)
# evaluate the model
scores = model.evaluate(X, Y)


print ("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print ("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))



