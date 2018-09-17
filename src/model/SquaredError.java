package model;

public class SquaredError implements IError {

	@Override
	public float error(float value) {
//		System.out.println(Math.pow(result - desired, 2));
		return (float) Math.pow(value, 2);
	}

//	@Override
//	public float derivative(float result, float desired) {
//		return 2 * (result - desired);
//	}
	
	@Override
	public float derivative(float result, float desired) {
		return (result - desired);
	}
	
}
