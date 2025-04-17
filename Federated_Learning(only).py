#ELLE-LEE MCGILLY BLOCKCHAIN-BASED FEDERATED LEARNING FOR SECURING IOT DEVICES HONOURS PROJECT

import tensorflow as tf
import keras
from keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from pip._internal.req.req_file import preprocess
from numpy.random import normal


''' LOAD AND PREPROCESS MNIST DATASET '''
" --- the MNIST dataset consists of 28x28 greyscale images of handwritten digits (0-9)  --- "
" --- x_train and x_test contain the images while y_train and y_test contain the corresponding labels --- "
" --- normalisation (x_train, x_test = x_train / 255.0) ensures that the pixel values of the images are scaled to a range between 0 and 1 / --- "
" --- / from their original range of 0-255, this helps with faster and more stable convergence during training --- "
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  #### Normalize data ####

''' SPLIT THE DATASET INTO 10 CLIENTS '''
" --- since FL is a distributed approach each client should have a portion of the training data --- "
" --- the training data will be split into 10 equal parts - one for each client --- "
" --- Non-IID scenario to keep it realistic making each client have different data --- "
store_client_data = {}  ### the store_client_data dictionary stores the data for each client ###
number_of_clients = 10
data_per_client = len(x_train) // number_of_clients ### dividing by 10 ###

for i in range(number_of_clients):
    store_client_data[i] = {
        'x': x_train[i * data_per_client: (i + 1) * data_per_client],
        ### contains the images for the client. ###
        'y': y_train[i * data_per_client: (i + 1) * data_per_client]
        ### contains the labels for the client. ###
    }


''' CREATING THE MODEL '''
" --- defining the model architecture with the Keras Sequential API to stack layers in a linear manner --- "
" https://www.tensorflow.org/guide/keras/sequential_model "

def make_model():
    cnn_model=models.Sequential([
        layers.Input(shape=(28,28)),### Specify the input shape using the Input layer ###
        layers.Flatten(),  ### Converts each 28x28 image into a 1D vector of 784 elements ###
        layers.Dense(128, activation='relu'),
        ### A fully connected dense layer with 128 neurons and ReLU activation. ReLU allows th emodel to learn nonlinear relationships ###
        layers.Dropout(0.2),
        ### A dropout layer that randomly drops 20% of the neurons during training to prevent overfitting ###
        layers.Dense(10, activation='softmax'),
        ### The output layer with 10 neurons (1 for each digit) and softmax activation ###
        ### softmax converts the outputs into probability distribution over the 10 classes ###

    ])

    cnn_model.compile(optimizer='adam',  ### gradient-based optimiser ###
                  loss='sparse_categorical_crossentropy', ### A loss function suitable for classification tasks where labels are integers ###
                  metrics=['accuracy'])  ### to track how well the model preforms on the test data ###
    return cnn_model

''' MALICIOUS CLIENT SIMULATION '''
" simulating malicious behaviour by adding random noise to the model weights to the clients with IDs 3 & 7 "
" https://medium.com/adding-noise-to-network-weights-in-tensorflow/adding-noise-to-network-weights-in-tensorflow-fddc82e851cb "

def mal_clients(cnn_model,client_id):
    if client_id in [3, 7]:
        weights = cnn_model.get_weights()
        for i in range(len(weights)):
            noise = normal(loc=0.0, scale=1.0, size=weights[i].shape)
            weights[i] += noise
        cnn_model.set_weights(weights)
        print(f"Malicious client", {client_id}," added noise." )


''' FL TRAINING '''

def fl_round(store_client_data):
    global_model = make_model()
    client_model_weights = [] # Initialize an empty list to store model weights from clients

    for client_id, data in store_client_data.items():
        create_client_model = make_model()
        create_client_model.set_weights(global_model.get_weights())
        create_client_model.fit(data['x'], data['y'], epochs=10, verbose=0) ### simulate training on the client's data ### also setting a higher epoch from 5>10 to make the test data more realisitc
        mal_clients(create_client_model, client_id)                        ### introduce malicious behavior towards clients ###
        client_model_weights.append(create_client_model.get_weights())     ### save the client model weights for aggregation

    ### Aggregate the model weights ###
    avg_weights = []
    for layer_weights in zip(*client_model_weights): ### loop through each layer of the model weights ###
        avg_layer_weights = np.mean(np.array(layer_weights), axis=0)
        avg_weights.append(avg_layer_weights)

    global_model.set_weights(avg_weights) ### <-- updates the global model with the avg'd weights

    return global_model


''' MODEL EVALUATION '''
" --- Evaluates the final model on the test set (x_test, y_test) after the FL rounds --- "
" --- model.evaluate() computes the loss and accuracy of the model on the test data --- "

def eval_model(eval_cnn_model, x_test, y_test):
    eval_loss, eval_acc = eval_cnn_model.evaluate(x_test, y_test, verbose=0)
    return eval_acc

''' RUNNING THE FL SIMULATION '''
" Make it so the model goes through 5 rounds of FL learning "
" The global model is updated with the aggregated weights from all clients and after each round the global model's accuracy is evaluated on the test set and logged "
" Print accuracy after final round "
rounds = 5
accuracy_log = []

for round_num in range(rounds):
    print(f"Round {round_num + 1}")
    global_model = fl_round(store_client_data)

    " --- Evaluate global model after training --- "
    accuracy = eval_model(global_model, x_test, y_test)
    accuracy_log.append(accuracy)

    print(f"Accuracy after round {round_num + 1}: {accuracy * 100:.2f}%")

" --- Print accuracy after final round --- "
print("\nFinal Model Accuracy after all rounds:", accuracy_log[-1] * 100)




# --- Plot accuracy after each FL round ---
plt.figure(figsize=(8, 5))
plt.plot(range(1, rounds + 1), [acc * 100 for acc in accuracy_log], marker='o', color='blue')
plt.title("Federated Learning Accuracy per round")
plt.xlabel("Round")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.grid(True)
plt.tight_layout()
plt.show()