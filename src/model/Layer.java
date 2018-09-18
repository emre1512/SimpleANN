package model;

import java.util.ArrayList;
import java.util.List;

import math.IActivation;

public abstract class Layer {

	private List<Neuron> neurons;
	private IActivation activationFunction;
	
	public Layer(int neuronCount, IActivation activationFunc){
		this.neurons = new ArrayList<>();
		this.activationFunction = activationFunc;
	}

	public List<Neuron> getNeurons() {
		return neurons;
	}

	public void addNeuron(Neuron neuron) {
		this.neurons.add(neuron);
	}		
			
}
