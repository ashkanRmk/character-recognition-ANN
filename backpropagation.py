from random import random
from math import exp


def read_train_file(file="OCR_train.txt"):
    training_data_list = []
    train_file = open(file, "r")
    for line in train_file:
        line = list(line.replace(" ", ""))
        line = [int(x) * 2 - 1 for x in line if x != "\n"]
        training_data_list.extend([line[:]])
    return training_data_list


def active_func_z(z_in):          # activation function using bipolar sigmoid function
    return (1 - exp(-z_in)) / (1 + exp(-z_in))


def make_binary(n):
    pos = n.index(1) + 1
    res = list(format(pos, 'b').zfill(3))
    res = [int(x) for x in res]
    return res


def set_v(num):
    v = []
    for x in range(num):
        v.extend([[random() for _ in range(64)]])  # initialize weights and biases
        # v.extend([[0] * 64])
    return v


def set_w(num):
    w = []
    for x in range(num):
        w.extend([[random() for _ in range(21)]])  # initialize weights and biases
        # w.extend([[0] * 21])
    return w

cal_eroor = lambda error, total: (error / total) * 100

v = set_v(21)
w = set_w(7)


epoch = 0                                          #counter of epoch
training_data = read_train_file()
hidden_unit = 21
z_in = [hidden_unit]
y_in = [7]

"""TRAINING PHASE OF NN"""
""" 1st PHASE: FEED FORWARD """

# while max(we) > epsilon or ch:                               #check stopping condition
for o in range(9):
    epoch += 1
    ch = False
    for j in training_data:
        x = j[:63]              # set each input unit
        b = j[63]
        expected = j[-7:]
        z_in[0] = 1

        for i in range(hidden_unit+1):                          # each output unit
            for V, s in zip(v, x):
                z_in[i+1] += V * s                     # calculate z_in(j)      j = 1, ..., p
            z_in[i+1] = active_func_z(z_in[i+1])

        for i in range(7):
            for W, z in zip(w, z_in):
                y_in[i] += W * z
            y_in[i] = active_func_z(y_in[i])


        #     for pos in range(63):
        #         temp = weight[pos]
        #         weight[pos] += (alpha * (t - y_in)) * x[pos]      #update weights(i, j)   i = 1, ..., 63
        #         dw[pos] = weight[pos] - temp
        #         print(dw[pos])
        #     temp = weight[63]
        #     weight[63] += alpha * (t - y_in)              # update bias(j)
        #     dw[63] = weight[63] - temp
        #     we.append(max(dw))
# print(str(epoch))
#
"""WEIGHTS AND BIASES SAVING PHASE OF NN"""
weight_file = open("Adeline_weights.txt‬‬", "w")
weight_file.write("Epochs: " + str(epoch) + "th" + "\n" + "\n")
for V in v:
    weight_file.write(str(V) + "\n" + "\n")
weight_file.close()

print("\nThe Neural Network has been trained in " + str(epoch) + "th epochs.")
print("Weights and Biases saved in: ‫Adeline_weights.txt")

"""USE PHASE OF ADELINE NN"""
output = []
_error = 0
_total = 0
results = open("‫‪results_adeline.txt‬‬", "w")
if input("\nDo you want to use your Adeline NN? (y/n)") == 'y':
    test_file = read_train_file("OCR_test.txt")
    for elem in test_file:
        sample = elem[:63]
        target = elem[-7:]
        b=elem[63]

        output.clear()
        _total += 1
        for weight in w:
            result = b
            for w, s in zip(weight, sample):
                result += w * s
            output.append(active_func_z(result))
        if target != output:
            _error += 1
        print("Expected: " + str(target))
        results.write("Expected: " + str(target))
        print("Result:   " + str(output) + "\n------------\n")
        results.write("\nResult:   " + str(output) + "\n------------\n")

print("\n\nPercent of Error in NN: " + str(cal_eroor(_error, _total)))
print("\nNumber of Cells in NN: " + str(len(w)))
results.write("\n\nPercent of Error in NN: " + str(cal_eroor(_error, _total)))
results.write("\nNumber of Cells in NN: " + str(len(w)))
results.close()
