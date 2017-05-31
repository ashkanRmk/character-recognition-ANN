#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#include <fstream>
#include <conio.h>
#include <stdlib.h>
#include <time.h>

using namespace std;

struct data
{
	vector<int> training_data_list;
	vector<int> target_data;
}data_set[21];

struct w
{
	vector<float> weights_cell;
}weights[7];

int activation_func(float y_in)
{
	float teta = 0.2;
	if (y_in > teta)
		return 1;
	else if (teta >= y_in >= -teta)
		return 0;
	else if (y_in < -teta)
		return -1;
}

float cal_error(int error, int total)
{
	return (error / total) * 100;
}


int main()
{
	fstream inputFile;
	inputFile.open("OCR_train.txt", ios::in | ios::beg);
	if (!inputFile)
	{
		cerr << "some thing wrong during opening file!" << endl;
		_getche();
		exit(1);
	}

	//Read training file 
	int tmp;
	for (int k = 0; k < 21; k++)
	{
		for (int i = 0; i < 64; i++)
		{
			inputFile >> tmp;
			data_set[k].training_data_list.push_back(tmp * 2 - 1);
		}
		for (int j = 0; j < 7; j++)
		{
			inputFile >> tmp;
			data_set[k].target_data.push_back(tmp * 2 - 1);
		}
	}
	inputFile.close();

	//Set weights
	for (int i = 0; i < 7; i++)
	{
		for (int j = 0; j < 64; j++)
		{
			srand(time(NULL));
			weights[i].weights_cell.push_back((float)rand() / RAND_MAX);
		}
	}

	int epoch = 0;
	vector<bool> errors;


	//Learning Phase
	do {
		errors.clear();
		epoch++;
		for (int i = 0; i < 21; i++)
		{
			for (int j = 0; j < 7; j++)
			{
				float result = 0;
				for (int k = 0; k < 64; k++)
					result += weights[j].weights_cell[k] * data_set[i].training_data_list[k];

				if (activation_func(result) != data_set[i].target_data[j])
				{
					for (int m = 0; m < 63; m++)
						weights[j].weights_cell[m] += data_set[i].target_data[j] * data_set[i].training_data_list[m];
					weights[j].weights_cell[63] += data_set[i].target_data[j];
					errors.push_back(true);
				}
				else
				{
					errors.push_back(false);
				}
			}
			cout << epoch << endl;
		}
	} while (find(errors.begin(), errors.end(), true) != errors.end());


	//Save weights to file
	fstream outputFile;
	outputFile.open("‫‪perceptron_weights.txt", ios::out);
	if (!outputFile)
	{
		cerr << "File can not open." << endl;
		exit(1);
	}
	outputFile << "Epochs: " << epoch << endl;
	for (int i = 0; i < 7; i++)
	{
		for (int j = 0; j < 64; j++)
		{
			outputFile << weights[i].weights_cell[j] << " ";
		}
		outputFile << endl << endl;
	}
	outputFile.close();


	//Using Phase
	vector<int> output;
	int err = 0;
	int total = 0;

	fstream inputFile1;
	inputFile1.open("OCR_test.txt", ios::in | ios::beg);
	if (!inputFile)
	{
		cerr << "some thing wrong during opening file!" << endl;
		_getche();
		exit(1);
	}

	//Read test file 
	for (int k = 0; k < 21; k++)
	{
		for (int i = 0; i < 64; i++)
		{
			inputFile1 >> tmp;
			data_set[k].training_data_list.push_back(tmp * 2 - 1);
		}
		for (int j = 0; j < 7; j++)
		{
			inputFile1 >> tmp;
			data_set[k].target_data.push_back(tmp * 2 - 1);
		}
	}
	inputFile1.close();

	fstream outputFile2;
	outputFile2.open("‫‪results.txt", ios::out);
	if (!outputFile2)
	{
		cerr << "File can not open." << endl;
		exit(1);
	}

	for (int i = 0; i < 21; i++)
	{
		output.clear();
		total++;
		for (int j = 0; j < 7; j++)
		{
			float result = 0;
			for (int k = 0; k < 64; k++)
				result += weights[j].weights_cell[k] * data_set[i].training_data_list[k];
			output.push_back(activation_func(result));
		}

		if (data_set[i].target_data != output)
			err++;

		for (int w = 0; w < 7; i++)
		{
			outputFile2 << "Expected: " << data_set[i].target_data[w] << " ";
			cout << "Expected: " << data_set[i].target_data[w] << " ";
		}
		outputFile2 << endl;
		cout << endl;
		for (int w = 0; w < 7; i++)
		{
			outputFile2 << "Result: " << output[w] << " ";
			cout << "Result: " << output[w] << " ";
		}
		outputFile2 << endl;
		cout << endl;
	}

	outputFile2 << "Percent of Error: " << cal_error(err, total) << endl;
	cout << "Percent of Error: " << cal_error(err, total) << endl;
	outputFile2.close();

	_getche();
	return 0;
}
