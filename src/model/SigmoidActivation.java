package model;

public class SigmoidActivation implements IActivation {

	@Override
	public float activate(float input) {
		return (float) (1. / (1. + Math.exp(-input)));
	}

	@Override
	public float derivative(float input) {
		return input * (1 - input);
	}

}
