package main;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;

import model.IError;
import model.NeuralNetwork;
import model.Neuron;
import model.SigmoidActivation;
import model.SquaredError;

public class Main {
	
	public static void main(String[] args){
				
		NeuralNetwork nn = new NeuralNetwork(5E-3f, 0.1f, new SquaredError());
 
		// Input layer
        Neuron input1 = new Neuron(new SigmoidActivation(), 0, 0);
        Neuron input2 = new Neuron(new SigmoidActivation(), 0, 0);

        // Hidden layer
        Neuron hidden1 = new Neuron(new SigmoidActivation(), 2, -1f);
        Neuron hidden2 = new Neuron(new SigmoidActivation(), 2, -1f);

        // Output layer
        Neuron output1 = new Neuron(new SigmoidActivation(), 2, -1f);
		
        nn.addInputNeuron(input1);
        nn.addInputNeuron(input2);
 
        nn.addHiddenNeuron(hidden1);
        nn.addHiddenNeuron(hidden2);
 
        nn.addOutNeuron(output1);
        
        // XOR problem dataset
        List<float[]> dataset = new ArrayList<>();
        dataset.add(new float[]{0, 0, 0});
        dataset.add(new float[]{0, 1, 1});
        dataset.add(new float[]{1, 1, 0});
        dataset.add(new float[]{1, 0, 1});
        
        int iteration = 0;
        
        do {

        	iteration++;
        	
        	nn.resetGlobalError();
        	
        	for(int i = 0; i < dataset.size(); i++){
                input1.setActivationOutput(dataset.get(i)[0]);
                input2.setActivationOutput(dataset.get(i)[1]);
                output1.setDesiredOutput(dataset.get(i)[2]);
                
                nn.train();
        	}        
        	
            if (iteration % 50000 == 0) {
                System.out.println("-------------------------------");
                System.out.println("Current iteration:" + iteration);
                System.out.println("Current error:" + nn.getGlobalError());
                System.out.println("-------------------------------");
            }
        	
		} while (!nn.hasLearnt() && iteration < 1);
        
        
//        // Test
//        
//        System.out.println("Training has been completed.");
//        System.out.println("Total iteration: " + iteration + ", accepted error: " + nn.getGlobalError());
//        System.out.println("Test cases are in progress...");
//         
//        for(int i = 0; i < dataset.size(); i++){
//                    	
//        	input1.setNeuronOutput(dataset.get(i)[0]);
//            input2.setNeuronOutput(dataset.get(i)[1]);
//            output1.setDesiredOutput(dataset.get(i)[2]);
//            
//            nn.test();
//            
//            System.out.println(dataset.get(i)[0] + " XOR " + dataset.get(i)[0] + " = " + quantizeResult(nn.getOutput(0).getActivationOutput()));
//
//    	}
               
	}

	public static String quantizeResult(float result) {
        return Integer.toString(Math.round(result)) + " (" + Float.toString(result) + ")";
    }
	
}
