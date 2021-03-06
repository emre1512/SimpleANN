package model;

import java.util.Random;

import math.IActivation;

public class Neuron {

	private float[] weights;
	private float[] updatedWeights;
    private float input;
    private float bias;
    private float neuronOutput;
    private float activationOutput;
    private IActivation activationFunction;
    private float delta;
    public float desiredOutput;
    public float error;
    public float derivativeOfError;
    private static final Random random = new Random();
    
    public Neuron(IActivation activationFunction, int weightCount, float bias) {
        this.weights = new float[weightCount];
        this.updatedWeights = new float[weightCount];
        this.activationFunction = activationFunction;
        this.delta = 0;
        this.error = 0;
        this.derivativeOfError = 0;
        this.desiredOutput = 0;
        this.bias = bias;
         
        initializeWeights();
    }
    
    public float getWeight(int index) {
        return weights[index];
    }
    
    public float getWeightCount(){
    	return weights.length;
    }
    
    public void setWeight(int index, float weight) {
        weights[index] = weight;     
    }
    
    public float getUpdatedWeight(int index) {
        return updatedWeights[index];
    }
    
    public void setUpdatedWeight(int index, float weight) {
    	updatedWeights[index] = weight;     
    }
 
    public float getInput() {
        return input;
    }
 
    public void setInput(float totalInput) {
        this.input = totalInput;
    }
    
    public float getNeuronOutput() {
		return neuronOutput;
	}

	public void setNeuronOutput(float neuronOutput) {
		this.neuronOutput = neuronOutput;
	}

	public float getActivationOutput() {
		return activationOutput;
	}

	public void setActivationOutput(float totalInput) {
		this.activationOutput = activationFunction.activate(totalInput);
	}

	public float getDesiredOutput() {
		return desiredOutput;
	}

	public void setDesiredOutput(float desiredOutput) {
		this.desiredOutput = desiredOutput;
	}

	public float getDelta() {
        return delta;
    }
	
	public void setDelta(float delta) {
        this.delta = delta;
    }
		
//	public void activate(float totalInput) {
//        activationOutput = activationFunction.activate(totalInput);
//    }
	
//	public void calculateDelta(float derivativeOfError) {
//        //delta = derivativeOfError * activationFunction.derivative(neuronOutput);
//        delta = derivativeOfError * neuronOutput * (1 - neuronOutput);
//    }
	     
    public float getBias() {
		return bias;
	}

	public void setBias(float bias) {
		this.bias = bias;
	}

	private void initializeWeights() {    	  
    	for (int i = 0; i < weights.length; i++) {  		
//    		float value = (float)(random.nextInt(100)) / 100.0f;      
    		float value = (float)(random.nextInt((100) + 1)) / 100.0f;
            this.weights[i] = value;
        }   	
    }
    
	
	
//    private void initializeBias() {
//    	float value = (float)(random.nextInt(100)) / 100.0f;        
//        this.bias = value;
//    }
	
}
