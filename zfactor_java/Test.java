package zFactor;

public class Test {

	public static void main(String[] args) {
		double zANN10, zANN5, zDAK;
		
		for (double Tpr = 1.0; Tpr <= 3.0; Tpr += 0.1)
		{
			System.out.println("\n================================@ Tpr = " + Tpr + " =========================================");
			for (double Ppr = 0; Ppr <= 30; Ppr += 1.0)
			{
				zANN10 = CalculateZFactor.ANN10(Ppr, Tpr);
				zANN5  = CalculateZFactor.ANN5(Ppr, Tpr);
				zDAK = CalculateZFactor.DAK(Ppr, Tpr);
				
				System.out.println("Ppr = " + Ppr + "\t ==>" + "\tz ANN10 = " + zANN10 + "\tz ANN5 = " + zANN5 + "\tz DAK = " + zDAK);
			}
		}
	}
}
