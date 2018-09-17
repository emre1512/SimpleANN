package model;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {

	private List<Neuron> inputLayer;
    private List<Neuron> hiddenLayer;
    private List<Neuron> outputLayer;
    private float desiredError;
    private float globalError;
    private float nu;
    private IError errorFunction;
    
    public NeuralNetwork(float desiredError, float nu, IError errorFunction) {
        inputLayer = new ArrayList<>();
        hiddenLayer = new ArrayList<>();
        outputLayer = new ArrayList<>();
        this.globalError = 0;
        this.desiredError = desiredError;
        this.nu = nu;
        this.errorFunction = errorFunction;
    }
    
    public void addInputNeuron(Neuron neuron) {
        inputLayer.add(neuron);
    }
     
    public void addHiddenNeuron(Neuron neuron) {
        hiddenLayer.add(neuron);
    }
     
    public void addOutNeuron(Neuron neuron) {
        outputLayer.add(neuron);
    }
    
    public Neuron getOutput(int index) {
        return outputLayer.get(index);
    } 
    
    public void train() {
        feedForward();
        calculateError();
        backPropagation();
//        updateWeights();
    }
    
    public void test() {
        feedForward();
    }
    
    protected void feedForward() {
    	for (Neuron nH : hiddenLayer) {
            float totalInput = 0;
            for (Neuron nI : inputLayer) {
                int i = inputLayer.indexOf(nI);
                totalInput += nI.getActivationOutput() * nH.getWeight(i);
            }
            
            totalInput += nH.getBias();
            
            nH.setNeuronOutput(totalInput);
            nH.setActivationOutput(totalInput);
        }
         
        for (Neuron nO : outputLayer) {
            float totalInput = 0;
            for (Neuron nH : hiddenLayer) {
                int i = hiddenLayer.indexOf(nH);
                totalInput += nH.getActivationOutput() * nO.getWeight(i);
            }
            
            totalInput += nO.getBias();

            nO.setNeuronOutput(totalInput);
            nO.setActivationOutput(totalInput);
        }
    }
    
    protected void calculateError() {
        
    	resetGlobalError();
        
    	for (Neuron nO : outputLayer) {

//            nO.error = errorFunction.error(nO.getActivationOutput() - nO.desiredOutput);
//            globalError += nO.error;
    		
    		// Error is considered as e = ai - yi
        	nO.error = nO.getActivationOutput() - nO.desiredOutput;
            globalError += errorFunction.error(nO.error);

        }
    	
		// Global error is calculated as G = sqrt(sum((ai - yi)^2))
        globalError = (float) Math.sqrt(globalError);
    }
    
    protected void backPropagation() {
    	
    	// BP for output layer weights and biases
        for (Neuron nO : outputLayer) {
            float derivativeOfError = errorFunction.derivative(nO.getActivationOutput(), nO.desiredOutput);
            
            // Calculate deltaO
            //float deltaO = derivativeOfError * nO.getNeuronOutput() * (1 - nO.getNeuronOutput());
            float deltaO = derivativeOfError * nO.getActivationOutput() * (1.0f - nO.getActivationOutput());
            //float deltaO = nO.error * nO.getActivationOutput() * (1.0f - nO.getActivationOutput());

            nO.setDelta(deltaO);
            
            for (Neuron nH : hiddenLayer) {
                int i = hiddenLayer.indexOf(nH);
        		float weightDiff = nu * nO.getDelta() * nH.getActivationOutput();
                float biasDiff = nu * nO.getDelta();
                //nO.setUpdatedWeight(i, nO.getWeight(i) - weightDiff);
                nO.setWeight(i, nO.getWeight(i) - weightDiff);
                nO.setBias(nO.getBias() - biasDiff);
            }
        }
        
        
        // BP for hidden layer weights
        for (Neuron nH : hiddenLayer) {
            int i = hiddenLayer.indexOf(nH);       		

        	float deltaH = 0;
        	
        	// Calculate deltaH
        	for(Neuron nO : outputLayer){     		
        		//deltaH += nO.getDelta() * nO.getWeight(i) * nH.getNeuronOutput() * (1 - nH.getNeuronOutput()); 
        		deltaH += nO.getDelta() * nO.getWeight(i) * nH.getActivationOutput() * (1.0f - nH.getActivationOutput()); 
        	}
        	
        	nH.setDelta(deltaH);
        	
            for (Neuron nI : inputLayer) {
                int p = inputLayer.indexOf(nI);
                float weightDiff = nu * nH.getDelta() * nI.getActivationOutput();
                float biasDiff = nu * nH.getDelta();
                //nH.setUpdatedWeight(p, nH.getWeight(p) - weightDiff);
                nH.setWeight(p, nH.getWeight(p) - weightDiff);
                nH.setBias(nH.getBias() - biasDiff);
            }
        }
    }
     
    private void updateWeights(){
    	 for (int i = 0; i < outputLayer.size(); i++){
    		 Neuron n = outputLayer.get(i);
    		 
    		 for(int j = 0; j < n.getWeightCount(); j++){
        		 n.setWeight(j, n.getUpdatedWeight(j));
    		 }    	
    	 }
    	 
    	 for (int i = 0; i < hiddenLayer.size(); i++){
    		 Neuron n = hiddenLayer.get(i);
    		 
    		 for(int j = 0; j < n.getWeightCount(); j++){
        		 n.setWeight(j, n.getUpdatedWeight(j));
    		 }  		 
    	 }
    }
	
    public float getGlobalError() {
        return this.globalError;
    }
    
    public void resetGlobalError() {
        this.globalError = 0;
    }
     
    public boolean hasLearnt() {

        return (globalError < desiredError);
    }
    
}
