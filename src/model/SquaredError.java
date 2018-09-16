package model;

public class SquaredError implements IError {

	@Override
	public float error(float result, float desired) {
		return (float) Math.pow(result - desired, 2);
	}

	@Override
	public float derivative(float result, float desired) {
		return 2 * (result - desired);
	}
	
}
