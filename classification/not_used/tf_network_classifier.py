# import numpy
# import pandas
# from pathlib import Path
# from typing import List, Tuple
# from pandas.core.frame import DataFrame
# from sklearn.utils import compute_class_weight
# from tensorflow import config
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.initializers import HeNormal
# from sklearn.model_selection._split import train_test_split
# 
# 
# config.set_visible_devices([], 'GPU')
# WEIGHTS_LOCATION = Path(__file__).parent.parent.joinpath('data', 'network_weights.h5')
# 
# 
# class TFNetwork(object):
# 
#     def __init__(self):
#         self.get_data()
# 
#     def initialize(self):
#         data, targets, labels, x_test, y_test = self.get_data()
#         self.network = self.build_model(len(data.columns.values), len(targets.columns.values))
#         class_weights = compute_class_weight(class_weight='balanced',
#                                              classes=targets.columns.values,
#                                              y=labels)
#         class_weights = {num: value for num, value in enumerate(class_weights)}
# 
#         print('Do you want to load a model from a previous run? Enter y/n')
#         answer = input().lower()
# 
#         if answer == 'y':
#             self.network.load_weights(WEIGHTS_LOCATION)
#             print('Model loaded.')
#         else:
#             history = self.network.fit(
#                 x=data,
#                 y=targets,
#                 epochs=100,
#                 verbose=2,
#                 validation_data=(x_test, y_test),
#                 batch_size=32,
#                 class_weight=class_weights,
#                 shuffle=True)
# 
#             print('Done training. Do you want to save the model? y/n')
#             answer = input().lower()
#             if answer == 'y':
#                 self.network.save_weights(WEIGHTS_LOCATION)
#                 print(f'Network weights saved to {WEIGHTS_LOCATION}.')
# 
#         self.test(x_test, y_test)
# 
#     def get_data(self) -> Tuple[DataFrame, DataFrame, List]:
#         """
#         Retrieve the data from the csv file, and split the targets from the rest of the data.
#         Some NaN values are set to 0, while the average score is set to the average of that column.
# 
#          :return:
#         - the data without the targets
#         - dataframe with only the targets
#         - list with the targets as indices instead of one-hot encoded.
#         """
#         path = str(Path(__file__).parent.parent.joinpath("data", "data-normalization-output.csv"))
#         df = pandas.read_csv(path)
# 
#         df.drop(df.columns[0], axis=1, inplace=True)
#         avg_avg_score = df['average_score'].mean()
#         df['average_score'] = df['average_score'].replace(numpy.nan, avg_avg_score)
#         df.fillna(0, inplace=True)
# 
#         targets = df[['final_result__Distinction', 'final_result__Fail', 'final_result__Pass',
#                       'final_result__Withdrawn']]
#         df = df.drop(['final_result__Distinction', 'final_result__Fail', 'final_result__Pass',
#                       'final_result__Withdrawn'], axis=1)
# 
#         targets.columns = [x.split("_")[-1] for x in targets.columns.values]
#         target_list = targets.idxmax(1)
# 
#         # used to get extra test data; left out for now
#         x_train, x_test, y_train, y_test = train_test_split(df, targets, test_size=0.1,
#                                                             random_state=32, shuffle=True,
#                                                             stratify=targets)
# 
#         return x_train, y_train, target_list, x_test, y_test
# 
#     @staticmethod
#     def build_model(input_size: int, output_size: int) -> Sequential:
#         """
#         Builds a neural network.
# 
#         :param input_size: number of parameters in the input layer
#         :param output_size: number of categories to be predicted
#         :return: a compiled neural network
#         """
#         input_size = [input_size]
#         model = Sequential()
#         model.add(Dense(128, activation='relu', input_shape=input_size,
#                         kernel_initializer=HeNormal(),
#                         bias_initializer='zeros'))
#         model.add(Dense(output_size, activation='softmax',
#                         kernel_initializer=HeNormal(),
#                         bias_initializer='zeros'))
# 
#         model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#         return model
# 
#     def test(self, data: DataFrame, labels: DataFrame):
#         """
#         Double check test function. Prints the accuracy.
# 
#         :param data: data without the target labels
#         :param labels: dataframe with only target labels
#         """
#         output = list(self.network.predict(data, verbose=True))
#         output_indices = [numpy.argmax(x) for x in output]
#         corrects = [label[output_indices[i]] != 0 for i, label in enumerate(labels.values)]
#         # incorrects = [(list(label), list(output[i])) for i, label in enumerate(labels.values)
#         #               if label[output_indices[i]] == 0]
#         print("Accuracy 4 categories: ", corrects.count(True) / len(corrects))
#         pf_labels = []
#         for label in labels.values:
#             pfl = [0, 0, 0, 0]
#             if label[0] != 0 or label[2] != 0:
#                 pfl[0] = 1
#                 pfl[2] = 1
#             if label[1] != 0 or label[3] != 0:
#                 pfl[1] = 1
#                 pfl[3] = 1
#             pf_labels.append(pfl)
#         kinda_corrects = [label[output_indices[i]] != 0 for i, label in enumerate(pf_labels)]
#         print("Accuracy for pass/fail: ", kinda_corrects.count(True) / len(kinda_corrects))
