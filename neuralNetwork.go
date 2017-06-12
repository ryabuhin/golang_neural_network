// NeuralNetwork for letter recognition | Valentine Riabukhin

// * -w1- \
// * -w2-- \   <<-  correct weights (backpropagation)   <<-
// * -w3------ * (output | summator -> func_activation) ->> result
// * -w4-- /
// * -w5- /

package main

import (
	"fmt"
	"math"
)

var tableOfLetters = [][][]float64{
	{
		{0.0, 1.0, 0.0},
		{1.0, 0.0, 1.0},
		{1.0, 1.0, 1.0},
		{1.0, 0.0, 1.0},
		{1.0, 0.0, 1.0},
		{65.0},
	},
	{
		{1.0, 1.0, 1.0},
		{1.0, 0.0, 1.0},
		{1.0, 1.0, 0.0},
		{1.0, 0.0, 1.0},
		{1.0, 1.0, 1.0},
		{66.0},
	},
	{
		{1.0, 1.0, 1.0},
		{1.0, 0.0, 0.0},
		{1.0, 0.0, 0.0},
		{1.0, 0.0, 0.0},
		{1.0, 1.0, 1.0},
		{67.0},
	},
}

var tableOfLettersValidate = [][][]float64{
	{
		{0.0, 1.0, 1.0},
		{0.0, 0.0, 1.0},
		{0.0, 1.0, 0.0},
		{0.0, 0.0, 1.0},
		{0.0, 1.0, 1.0},
	},
	{
		{1.0, 1.0, 1.0},
		{1.0, 0.0, 0.0},
		{1.0, 0.0, 0.0},
		{1.0, 0.0, 0.0},
		{1.0, 1.0, 1.0},
	},
	{
		{0.0, 1.0, 0.0},
		{1.0, 0.0, 1.0},
		{1.0, 1.0, 1.0},
		{1.0, 0.0, 1.0},
		{1.0, 0.0, 1.0},
	},
}

type neuralNetwork struct {
	enters, weights    []float64
	out, learningSpeed float64
}

func (neuralNet *neuralNetwork) summator() {
	neuralNet.out = 0
	for i, enter := range neuralNet.enters {
		neuralNet.out += enter * neuralNet.weights[i]
	}
	neuralNet.out = toFixed(1/(1+math.Exp(-neuralNet.out))*3+65, 0)
}

func (neuralNet *neuralNetwork) correctWeights(error float64) {
	for i, weight := range neuralNet.weights {
		neuralNet.weights[i] = neuralNet.learningSpeed*error*neuralNet.enters[i] + weight
	}
}

func (neuralNet *neuralNetwork) train() {
	epochError := 1.0
	for epoch := 0; epoch < 20 || epochError != 0; epoch++ {
		epochError = 0
		trainArray := []float64{}
		for i := 0; i < len(tableOfLetters); i++ {
			trainArray = trainArray[:0]
			for j := 0; j < len(tableOfLetters[i])-1; j++ {
				trainArray = append(trainArray, tableOfLetters[i][j]...)
			}
			neuralNet.enters = trainArray
			neuralNet.summator()
			error := tableOfLetters[i][5][0] - neuralNet.out
			neuralNet.correctWeights(error)
			epochError += math.Abs(float64(error))
		}
	}
}

func main() {
	fmt.Println("Initialization ...")
	nNet := neuralNetwork{
		learningSpeed: 0.001,
		weights:       []float64{0.01, 0.2, 0.03, 0.1, 0.05, 0.033, 0.22, 0.001, 0.09, 0.14, 0.214, 0.05, 0.033, 0.22, 0.2},
	}
	fmt.Println("Start learning machine")
	nNet.train()
	fmt.Println("Stop learning machines\n")
	trainArray := []float64{}
	for i := 0; i < len(tableOfLettersValidate); i++ {
		trainArray = trainArray[:0]
		for j := 0; j < len(tableOfLettersValidate[i]); j++ {
			trainArray = append(trainArray, tableOfLettersValidate[i][j]...)
		}
		nNet.enters = trainArray
		nNet.summator()
		fmt.Println("Out result: ", string(int(nNet.out)), "\t for: ", nNet.enters)
	}
}

func round(num float64) int {
	return int(num + math.Copysign(0.5, num))
}

func toFixed(num float64, precision int) float64 {
	output := math.Pow(10, float64(precision))
	return float64(round(num*output)) / output
}
