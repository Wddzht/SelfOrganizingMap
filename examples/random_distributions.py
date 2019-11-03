import numpy as np
from som import SOM

if __name__ == "__main__":
    # generate some random data with 36 features
    data1 = np.random.normal(loc=-.25, scale=0.5, size=(500, 36))
    data2 = np.random.normal(loc=.25, scale=0.5, size=(500, 36))
    data = np.vstack((data1, data2))

    som = SOM(10, 10)  # initialize the SOM
    som.fit(data, 1000, save_e=True, interval=100)  # fit the SOM for 10000 epochs, save the error every 100 steps
    som.plot_error_history(filename='E:\_Python\SelfOrganizingMap\images\som_error.png')  # plot the training error history

    targets = np.array(500 * [0] + 500 * [1])  # create some dummy target values

    # now visualize the learned representation with the class labels
    som.plot_point_map(data, targets, ['Class 0', 'Class 1'], filename=r'E:\_Python\SelfOrganizingMap\images\som.t.png')
    som.plot_class_density(data, targets, t=0, name='Class 0', filename=r'E:\_Python\SelfOrganizingMap\images\class_0.t.png')
    som.plot_distance_map(filename='E:\_Python\SelfOrganizingMap\images\distance_map.t.png')  # plot the distance map after training
