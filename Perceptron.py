from numpy import random

def read_train_file(file="OCR_train.txt"):
    training_data_list = []
    train_file = open(file, "r")
    for line in train_file:
        line = list(line.replace(" ", ""))
        line = [x for x in line if x != "\n"]
        training_data_list.extend([line[:]])
    return training_data_list

def active_func(y_in, teta=0.2):                    #activation function
    if y_in > teta:
        return 1
    elif teta >= y_in >= -teta:
        return 0
    elif y_in < -teta:
        return -1

make_int_array = lambda my_list: [int(x) * 2 - 1 for x in my_list]      #make bipolar

cal_eroor = lambda error, total: (error / total) * 100

weights = []
for x in range(7):
    weights.extend([random.rand(64)])               #initialize weights and biases

errors = [True]                                     #contain errors of each training pair
epoch = 0                                           #counter of epochs
training_data = read_train_file()

"""TRAINING PHASE OF NN"""
while True in errors:                               #check stopping condition
    errors.clear()
    epoch += 1
    for j in training_data:
        x = make_int_array(j[:64])                  #set each input unit
        expected = make_int_array(j[-7:])

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
        sample = make_int_array(elem[:64])
        target = make_int_array(elem[-7:])
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
results.write("\n\nPercent of Error in NN: " + str(cal_eroor(_error, _total)))
results.close()
