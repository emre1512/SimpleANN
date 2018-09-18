package model;

import math.IActivation;
import math.IError;

public class OutputLayer extends Layer{

	private IError errorFunction;
	
	public OutputLayer(int neuronCount, IActivation activationFunc, IError errorFunction) {
		super(neuronCount, activationFunc);
		this.errorFunction = errorFunction;
	}

}
