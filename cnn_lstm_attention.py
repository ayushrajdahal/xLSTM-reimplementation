## Data read and header inclusion
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import random
from keras.layers import Layer
from keras import backend as K

from keras.models import Sequential
from keras.layers import Dense,Dropout, Flatten,LSTM,RepeatVector,TimeDistributed,Conv1D,MaxPooling1D,GlobalMaxPooling1D,  Activation, Input, Attention
from keras.layers import Permute, Lambda, RepeatVector, Multiply
from keras.callbacks import EarlyStopping
from statsmodels.tsa.seasonal import seasonal_decompose


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout, TimeDistributed
from tensorflow.keras.layers import Input, Concatenate, Permute, Reshape, Multiply, Lambda
from tensorflow.keras import Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

# ! pip install livelossplot
from keras.optimizers import Adam
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)
os.environ['TF_DETERMINISTIC_OPS'] = '1'


# Verify GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


############################# data import#########################

# Data import
data = pd.read_excel('Solar station site 1 (Nominal capacity-50MW) (1).xlsx') # site 1
# data = pd.read_excel('Solar station site 2 (Nominal capacity-130MW).xlsx')  # site 2
# data = pd.read_excel('Solar station site 3 (Nominal capacity-30MW).xlsx')  #
# data = pd.read_excel('Solar station site 4 (Nominal capacity-130MW).xlsx') 
# data = pd.read_excel('Solar station site 5 (Nominal capacity-110MW).xlsx') 
# data = pd.read_excel('Solar station site 6 (Nominal capacity-35MW).xlsx') 
# data = pd.read_excel('Solar station site 7 (Nominal capacity-30MW).xlsx') 
# data = pd.read_excel('Solar station site 8 (Nominal capacity-30MW).xlsx') 


data.head()
print(data.dtypes)

# if data.dtypes.any() == np.dtype('<M8[ns]'):
#     data = data.apply(lambda x: x.astype(float))

# Convert time column to datetime and set it as index
data['Time(year-month-day h:m:s)'] = pd.to_datetime(data['Time(year-month-day h:m:s)'])
data.set_index('Time(year-month-day h:m:s)', inplace=True)

# Strip leading/trailing spaces from column names
data.columns = data.columns.str.strip()

# Handle NaN values
data.fillna(method='ffill', inplace=True)

# Normalize the features
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Convert the scaled data back to a dataframe
data_scaled = pd.DataFrame(data_scaled, columns=data.columns, index=data.index)

# Display the first few rows of the normalized data
data_scaled.head()





# #  cleaning
# data.replace(['--', '<NULL>'], 'NaN', inplace=True)

# # Convert 'Time(year-month-day h:m:s)' to datetime and set as index (if not already)
# data['Time(year-month-day h:m:s)'] = pd.to_datetime(data['Time(year-month-day h:m:s)'])
# data.set_index('Time(year-month-day h:m:s)', inplace=True)

# # Fill missing values
# data.fillna(method='ffill', inplace=True)

################################data 


def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data[i-n_steps:i, :-1])  # All features except the last (target) column
        y.append(data[i, -1])  # Target column (Power output)
    return np.array(X), np.array(y)

# Number of timesteps in the input sequence
n_steps = 24  # For example, 24*15min = 6 hours of historical data to predict the next value

# Prepare input/output sequences
X, y = create_sequences(data_scaled.values, n_steps)

# Splitting data into training and testing sets (let's use 80-20 split)
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

X_train.shape, y_train.shape, X_test.shape, y_test.shape


########################################## MOdel detaisl ######################

# Building the model with CNN, LSTM, and Attention
def CNNLSTMATTENTION(input_shape):
    # Input layer
     input_layer = Input(shape=input_shape)

   #  # CNN layer
   #  conv1 = Conv1D(filters=64, kernel_size=2, activation='relu')(input_layer)
   #  pool1 = MaxPooling1D(pool_size=2)(conv1)
   #  drop1 = Dropout(0.2)(pool1)

   #  # LSTM layer
   #  lstm1 = LSTM(50)(drop1)
   # # lstm1 = LSTM(50, return_sequences=True)(drop1)
   #  drop2 = Dropout(0.2)(lstm1)

   #  # Attention mechanism
   #  attention_probs = Dense(50, activation='softmax', name='attention_vec')(drop2)
   #  attention_mul = Multiply()([drop2, attention_probs])

   #  # Flattening the output to feed into output layer
   #  flat = Flatten()(attention_mul)

   #  # Output layer
   #  output = Dense(1)(flat)

   #  model = Model(inputs=input_layer, outputs=output)
   #  model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape'])
      
     conv1 = Conv1D(filters=64, kernel_size=2, activation='relu')(input_layer)
     pool1 = MaxPooling1D(pool_size=2)(conv1)
     drop1 = Dropout(0.2)(pool1)
     lstm1 = LSTM(50, return_sequences=True)(drop1)
     drop2 = Dropout(0.2)(lstm1)
     attention_probs = Dense(50, activation='softmax', name='attention_vec')(drop2)
     attention_mul = Multiply()([drop2, attention_probs])
     flat = Flatten()(attention_mul)
     output = Dense(1)(flat)
     model= Model(inputs=input_layer, outputs=output)
     model.compile(optimizer='adam', loss='mse')

     return model

# Instantiate and compile the model
model_1 = CNNLSTMATTENTION((n_steps, X_train.shape[2]))
model_1 .summary()

############################################# training#############


history = model_1.fit(X_train, y_train, epochs=25, batch_size=100, validation_split=0.2, verbose=1)


######################## plot accuracy ####################

# Plotting training and validation loss
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Test')
plt.title('Training vs Testing Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# ##########################


predictions_1 = model_1.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions_1))
mae = mean_absolute_error(y_test, predictions_1)
mape = np.mean(np.abs((y_test - predictions_1) / y_test)) * 100
r2 = r2_score(y_test, predictions_1)

epsilon = 1e-8  # Small constant
mape = np.mean(np.abs((y_test - predictions_1) / (y_test + epsilon))) * 100
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, predictions_1))}")
print(f"MAE: {mean_absolute_error(y_test, predictions_1)}")
print(f"MAPE: {mape}")
print(f"R2 Score: {r2_score(y_test, predictions_1)}")


# # print(f"RMSE: {rmse}")
# # print(f"MAE: {mae}")
# # print(f"MAPE: {mape}")
# # print(f"R2 Score: {r2}")

# ####################################################

# import matplotlib.pyplot as plt

# # Assuming 'predictions' and 'y_test' are available
# plt.figure(figsize=(10, 6), dpi=300)  # Set the figure size for better readability
# plt.plot(y_test[:500], label='Actual Solar Power', linewidth=2)  # Plot actual values
# plt.plot(predictions_1[:500], label='Predicted Solar Power', linewidth=2)  # Plot predicted values
# plt.title('Comparison of Actual and Predicted Solar Power')  # Add a title
# plt.xlabel('Time (15 Seconds)',fontsize=12)  # Label the x-axis
# plt.ylabel('Solar Power (MW)',fontsize=12)  # Label the y-axis
# plt.legend()  # Add a legend to clarify which line is which
# #plt.grid(True)  # Add grid for better readability of the plot
# plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

# # Save the plot as a PNG file
# plt.savefig('Actual_vs_Predicted_Solar_Power.png',)

# # Show the plot
# plt.show()


# ########################################## fearure abs





# ####################################################



# # Assuming model, X_test are already defined
# original_output = model_1.predict(X_test[:14000])
# feature_impacts = []

# for i in range(X_test.shape[2]):  # Iterate over each feature
#     temp = X_test[:14000].copy()
#     temp[:, :, i] = 0  # Set one feature at a time to zero
#     modified_output = model_1.predict(temp)
#     impact = np.mean(np.abs(original_output - modified_output), axis=0)  # Measure the mean absolute change
#     feature_impacts.append(impact)

# # Convert impacts to a numpy array for easier handling
# feature_impacts = np.array(feature_impacts)


# ###################################



# ###################################
# # Names of features, adjust these as per your dataset
# feature_names = ['Total solar irradiance (W/m2)', 'Direct normal irradiance (W/m2)', 'Global horizontal irradiance (W/m2)', 'Air temperature  (°C)', 'Atmosphere (hpa)']

# # If feature_impacts is a 2D array with a single column, convert it to a 1D array
# if feature_impacts.ndim > 1:
#     feature_impacts = feature_impacts.flatten()  # This will convert a 2D array to 1D

# # Ensure feature_names is a list of strings
# if not isinstance(feature_names, list):
#     feature_names = list(feature_names)


# # Check shapes and content
# print("Feature impacts:", feature_impacts)
# print("Feature names:", feature_names)

# colors = ['blue', 'orange', 'green', 'red', 'purple']

# # Creating a bar plot to visualize the impacts
# plt.figure(figsize=(10, 6,), dpi=300)
# plt.bar(feature_names, feature_impacts, color=colors)
# #plt.xlabel('Features', fontsize=12,)
# plt.ylabel('Impact on Model Output', fontsize=12,fontweight='bold')
# #plt.title('Impact of Each Feature on CNN-LSTM-Attention Prediction Model',fontweight='bold')
# plt.xticks(rotation=35,fontsize=12,fontweight='bold')  
# # Rotate feature names for better readability
# plt.yticks(fontsize=12,fontweight='bold')  
# plt.tight_layout()
# ax = plt.gca()

# # Increase the width of all sides of the axis lines
# ax.spines['bottom'].set_linewidth(2)  # Increase width of the bottom (x) axis
# ax.spines['left'].set_linewidth(2)    # Increase width of the left (y) axis
# ax.spines['top'].set_linewidth(2)     # Increase width of the top axis
# ax.spines['right'].set_linewidth(2)   # Increase width of the right axis
# # Save the plot as a PNG file
# plt.savefig('Impact of Each Feature on CNN-LSTM-Attention Prediction Model.png',)

# plt.show()




