package zFactor;

public class Test {

	public static void main(String[] args) {
		double Ppr, Tpr, z;
		
		Ppr = 1;
		Tpr = 1;
		
		z = CalculateZFactor.ANN10(Ppr, Tpr);
		
		System.out.println(z);
	}
}
