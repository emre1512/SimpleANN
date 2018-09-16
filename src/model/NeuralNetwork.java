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
        updateWeights();
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
            
            //nH.setInput(totalInput);
            //nH.activate(totalInput);
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
            
            //nO.setInput(totalInput);
            //nO.activate(totalInput);
            nO.setNeuronOutput(totalInput);
            nO.setActivationOutput(totalInput);
            
        }
    }
    
    protected void calculateError() {
        //error = 0;
        for (Neuron nO : outputLayer) {
        	
        	// No need for this line actually
            nO.error = errorFunction.error(nO.getActivationOutput(), nO.desiredOutput);
            
            // Global error
            globalError += nO.error;
        }
    }
    
    protected void backPropagation() {
    	
    	// BP for output layer weights
        for (Neuron nO : outputLayer) {
            float derivativeOfError = errorFunction.derivative(nO.getActivationOutput(), nO.desiredOutput);
            //nO.derivativeOfError = derivativeOfError;
            //nO.calculateDelta(derivativeOfError);
            
            // Calculate deltaO
            float deltaO = derivativeOfError * nO.getNeuronOutput() * (1 - nO.getNeuronOutput());
            nO.setDelta(deltaO);
            
            for (Neuron nH : hiddenLayer) {
                int i = hiddenLayer.indexOf(nH);
                //nH.calculateDelta(lambda, nO.getDelta() * nO.getWeight(i));
                float diff = nu * nO.getDelta() * nH.getActivationOutput();
                //nO.setWeight(i, nO.getWeight(i) - diff);
                nO.setUpdatedWeight(i, nO.getWeight(i) - diff);
            }
        }
         
        // BP for hidden layer weights
        for (Neuron nH : hiddenLayer) {
            int i = hiddenLayer.indexOf(nH);       		

        	float deltaH = 0;
        	
        	// Calculate deltaH
        	for(Neuron nO : outputLayer){     		
        		deltaH += nO.getDelta() * nO.getWeight(i) * nH.getNeuronOutput() * (1 - nH.getNeuronOutput());  		
        	}
        	
        	nH.setDelta(deltaH);
        	
            for (Neuron nI : inputLayer) {
                int p = inputLayer.indexOf(nI);
                float diff = nu * nH.getDelta() * nI.getActivationOutput();
                //nH.setWeight(i, nH.getWeight(i) - diff);
                nH.setUpdatedWeight(p, nH.getWeight(p) - diff);
            }
        }
    }
     
    private void updateWeights(){
    	 for (int i = 0; i < outputLayer.size(); i++){
    		 Neuron n = outputLayer.get(i);
    		 n.setWeight(i, n.getUpdatedWeight(i));
    	 }
    	 
    	 for (int i = 0; i < hiddenLayer.size(); i++){
    		 Neuron n = hiddenLayer.get(i);
    		 n.setWeight(i, n.getUpdatedWeight(i));
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
