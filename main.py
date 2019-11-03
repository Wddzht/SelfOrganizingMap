import numpy as np
from som import SOM
import data
import normalize

if __name__ == "__main__":
    # generate some random data with 36 features
    data_class = data.DataClass([float] * 4 + [str])
    data_class.read(r"E:\_Python\SelfOrganizingMap\data\iris.data", False, split_tag=',')
    data_class.parse()
    normalize.min_max_normalize(data_class, [0, 1, 2, 3])
    _data = np.array(data_class.data)
    data = np.delete(_data, -1, axis=1)
    data = np.array(data, dtype='float_')

    np.mean(data)
    np.std(data)

    som = SOM(15, 15)  # initialize the SOM
    som.fit(data, 10000, save_e=True, interval=500)  # fit the SOM for 10000 epochs, save the error every 500 steps
    som.plot_error_history(
        filename=r'E:\_Python\SelfOrganizingMap\images\som_error.iris.png')  # plot the training error history

    targets = _data[:, -1]
    targets_index = []
    target_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    for item in targets:
        if item == target_names[0]:
            targets_index.append(0)
        elif item == target_names[1]:
            targets_index.append(1)
        elif item == target_names[2]:
            targets_index.append(2)
        else:
            raise ValueError(targets)
    # now visualize the learned representation with the class labels

    som.plot_point_map(data, targets_index, target_names,
                       filename=r'E:\_Python\SelfOrganizingMap\images\som.iris.png')  # todo:have three classes
    som.plot_class_density(data, targets, t='Iris-setosa', name='setosa',
                           filename=r'E:\_Python\SelfOrganizingMap\images\class_setosa.png')
    som.plot_class_density(data, targets, t='Iris-versicolor', name='versicolor',
                           filename=r'E:\_Python\SelfOrganizingMap\images\class_versicolor.png')
    som.plot_class_density(data, targets, t='Iris-virginica', name='virginica',
                           filename=r'E:\_Python\SelfOrganizingMap\images\class_virginica.png')
    som.plot_distance_map(
        filename=r'E:\_Python\SelfOrganizingMap\images\distance_map.iris.png')  # plot the distance map after training

    som.plot_distance_point_map(data, targets_index, target_names,
                                filename=r'E:\_Python\SelfOrganizingMap\images\distance_point.iris.png')  # plot the distance map after training
