from random import *

def read_train_file(file="OCR_train.txt"):
    training_data_list = []
    train_file = open(file, "r")
    for line in train_file:
        line = list(line.replace(" ", ""))
        line = [int(x) * 2 - 1 for x in line if x != "\n"]
        training_data_list.extend([line[:]])
    return training_data_list

def active_func(y_in, teta=0):                    #activation function
    if y_in > teta:
        return 1
    elif teta >= y_in >= -teta:
        return 0
    elif y_in < -teta:
        return -1

def make_binary(n):
    pos = n.index(1) + 1
    res = list(format(pos, 'b').zfill(3))
    res = [int(x) for x in res]
    return res

def make_forth_cells(n):
    pos = n.index(1) + 1
    if pos == 1:
        list = [1,0,0,0]
    elif pos == 2:
        list = [0,1,0,0]
    elif pos == 3:
        list = [0,0,1,0]
    elif pos == 4:
        list = [0,0,0,1]
    elif pos == 5:
        list = [1,1,0,0]
    elif pos == 6:
        list = [0,0,1,1]
    elif pos == 7:
        list = [0,1,1,0]
    return list

def set_weight(num):
    weights = []
    for x in range(num):
        weights.extend([[random() for _ in range(64)]])  # initialize weights and biases
        # weights.extend([[0] * 64])
    return weights


cal_eroor = lambda error, total: (error / total) * 100


weights = set_weight(7)
"""3 CELLS"""
# weights = set_weight(3)
"""4 CELLS"""
# weights = set_weight(4)


errors = [True]                                     #contain errors of each training pair
epoch = 0                                           #counter of epochs
training_data = read_train_file()

"""TRAINING PHASE OF NN"""
while True in errors:                               #check stopping condition
    errors.clear()
    epoch += 1
    for j in training_data:
        x = j[:64]                 #set each input unit
        expected = j[-7:]

        """3 CELLS"""
        # expected = make_binary(expected)
        """4 CELLS"""
        # expected = make_forth_cells(expected)

        # print(expected)
        for weight, t in zip(weights, expected):    #each output unit
            result = 0                              # y_in in each training pair
            for w, s in zip(weight, x):
                result += w * s                     #calculate y_in(j)      j = 1, ..., 7

            if active_func(result) != t:
                for pos in range(63):
                    weight[pos] += t * x[pos]       #update weights(i, j)   i = 1, ..., 63
                weight[63] += t                     #update bias(j)
                errors.append(True)
            else:
                errors.append(False)                #weights unchanged!
    print(str(epoch))

"""WEIGHTS AND BIASES SAVING PHASE OF NN"""
weight_file = open("‫‪perceptron_weights.txt‬‬", "w")
weight_file.write("Epochs: " + str(epoch) + "th" + "\n" + "\n")
for w in weights:
    weight_file.write(str(w) + "\n" + "\n")
weight_file.close()

print("\nThe Neural Network has been trained in " + str(epoch) + "th epochs.")
print("Weights and Biases saved in: ‫‪perceptron_weights.txt")


"""USE PHASE OF PERCEPTRON NN"""
output = []
_error = 0
_total = 0
results = open("‫‪results.txt‬‬", "w")
if input("\nDo you want to use your Perceptron NN? (y/n)") == 'y':
    test_file = read_train_file("OCR_test.txt")
    for elem in test_file:
        sample = elem[:64]
        target = elem[-7:]

        """3 CELLS"""
        # target = make_binary(target)
        """4 CELLS"""
        # target = make_forth_cells(target)

        output.clear()
        _total += 1
        for weight in weights:
            result = 0
            for w, s in zip(weight, sample):
                result += w * s
            output.append(active_func(result))
        if target != output:
            _error += 1
        print("Expected: " + str(target))
        results.write("Expected: " + str(target))
        print("Result:   " + str(output) + "\n------------\n")
        results.write("\nResult:   " + str(output) + "\n------------\n")

print("\n\nPercent of Error in NN: " + str(cal_eroor(_error, _total)))
print("\nNumber of Cells in NN: " + str(len(weights)))
results.write("\n\nPercent of Error in NN: " + str(cal_eroor(_error, _total)))
results.write("\nNumber of Cells in NN: " + str(len(weights)))
results.close()
