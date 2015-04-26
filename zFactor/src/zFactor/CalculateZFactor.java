package zFactor;

public final class CalculateZFactor {
	//Minimum and Maximum values used in the neural network to normalize the input and output values.
	private final static double Ppr_min = 0;
	private final static double Ppr_max = 30;
	private final static double Tpr_min = 1;
	private final static double Tpr_max = 3;
	private final static double Z_min = 0.25194;
	private final static double Z_max = 2.66;

	/**
	 * The is the structure of the Network 2-5-5-1
	 */
	//Weights and Biases for the 1st layer of neurons
	private double[][] wb1_5 =	{{-1.5949,	7.9284,		7.2925},
								{-1.7917,	1.2117,		2.221},
								{5.3547,	-4.5424,	-0.9846},
								{4.6209,	2.2228,		8.9966},
								{-2.3577,	-0.1499,	-1.5063}};


	//Weights and Biases for the 2nd layer of neurons
	private double[][] wb2_5 =	{{2.3617,	-4.0858,	1.2062,		-1.1518,	-1.2915,	2.0626},
								{10.0141,	9.8649,		-11.4445,	-123.0698,	7.5898,	    95.1393},
								{10.4103,	14.1358,	-10.9061,	-125.5468,	6.3448,	    93.8916},
								{-1.7794,	14.0742,	-1.4195,	12.0894,	-15.4537,	-9.9439},
								{-0.5988,	-0.4354,	-0.336, 	9.9429,		-0.4029,	-8.3371}};


	//Weights and Biases for the 3rd layer of neurons
	private double[] wb3_5 =	{1.4979,	-37.466,	37.7958,	-7.7463,	6.9079,		2.8462 };
	//-------------END OF NETWORK 2-5-5-1 STRUCTURE------------------------------------------------------------------------

	/**
	 * The is the structure of the Network 2-10-10-1
	 */
	//Weights and Biases for the 1st layer of neurons
	private static double[][] wb1_10 =	{{2.2458,	-2.2493,	-3.7801},
								{3.4663,	8.1167,		-14.9512},
								{5.0509,	-1.8244,	3.5017},
								{6.1185,	-0.2045,	0.3179},
								{1.3366,	4.9303,		2.2153},
								{-2.8652,	1.1679,		1.0218},
								{-6.5716,	-0.8414,	-8.1646},
								{-6.1061,	12.7945,	7.2201},
								{13.0884,	7.5387,		19.2231},
								{70.7187,	7.6138,		74.6949}};


	//Weights and Biases for the 2nd layer of neurons
	private static double[][] wb2_10 =		{{4.674,	1.4481,		-1.5131,	0.0461,		-0.1427,	2.5454,		-6.7991,	-0.5948,	-1.6361,	0.5801,		-3.0336},
									{-6.7171,	-0.7737,	-5.6596,	2.975,	    14.6248,	2.7266,	    5.5043,	    -13.2659,	-0.7158,	3.076,	    15.9058},
									{7.0753,	-3.0128,	-1.1779,	-6.445,	    -1.1517,	7.3248,	    24.7022,	-0.373,	    4.2665,	    -7.8302,	-3.1938},
									{2.5847,	-12.1313,	21.3347,	1.2881,	    -0.2724,	-1.0393,	-19.1914,	-0.263,	    -3.2677,	-12.4085,	-10.2058},
									{-19.8404,	4.8606,	    0.3891,	    -4.5608,	-0.9258,	-7.3852,	18.6507,	0.0403,	    -6.3956,	-0.9853,	13.5862},
									{16.7482,	-3.8389,	-1.2688,	1.9843,	    -0.1401,	-8.9383,	-30.8856,	-1.5505,	-4.7172,	10.5566,	8.2966},
									{2.4256,	2.1989,	    18.8572,	-14.5366,	11.64,	    -19.3502,	26.6786,	-8.9867,	-13.9055,	5.195,	    9.7723},
									{-16.388,	12.1992,	-2.2401,	-4.0366,	-0.368,	    -6.9203,	-17.8283,	-0.0244,	9.3962,	    -1.7107,	-1.0572},
									{14.6257,	7.5518,	    12.6715,	-12.7354,	10.6586,	-43.1601,	1.3387,	    -16.3876,	8.5277,	    45.9331,	-6.6981},
									{-6.9243,	0.6229,	    1.6542,	    -0.6833,	1.3122,	    -5.588,	    -23.4508,	0.5679,	    1.7561,	    -3.1352,	5.8675}};


	//Weights and Biases for the 3rd layer of neurons
	private static double[] wb3_10 =		{-30.1311, 2.0902,		-3.5296,	18.1108,	-2.528,		-0.7228,	0.0186,		5.3507,		-0.1476,	-5.0827,	3.9767 };
	//-------------END OF NETWORK 2-10-10-1 STRUCTURE------------------------------------------------------------------------
	
	private double[][] n1_5 = new double[5][2];      //input and output of the 1st layer in 2-5-5-1 network.	[,0] ==> inputs, [,1] ==> outputs
	private double[][] n2_5 = new double[5][2];      //input and output of the 2nd layer in 2-5-5-1 network.	[,0] ==> inputs, [,1] ==> outputs
	private static double[][] n1_10 = new double[10][2];    //input and output of the 1st layer in 2-10-10-1 network.	[,0] ==> inputs, [,1] ==> outputs
	private static double[][] n2_10 = new double[10][2];    //input and output of the 2nd layer in 2-10-10-1 network.	[,0] ==> inputs, [,1] ==> outputs

/*	private double p_pr;	//Pseudo Critical Pressure
	private double t_pr;	//Pseudo Critical Temperature
	private double zAnn5;	//z factor of Artificial Neural Network
	private double zAnn10;	//z factor of Artificial Neural Network
	private double zDAK;	//z factor of DAK
	private double zGuess = 0;	//guess value of DAK
*/
	private final double tol = 0.001;     //tolerance of DAK
	private final int maxIter = 20;    //Max number of iterations for DAK

	private static double Ppr_n;   //normalized Ppr - is used in ANN
	private static double Tpr_n;   //normalized Tpr - is used in ANN
	private static double z_n;     //normalized z   - is used in ANN
	//-------------------------------------------------------------------------------------------------------------------------
	
	
	public static double ANN10(double Ppr, double Tpr)
	{		
        Ppr_n = 2.0 / (Ppr_max - Ppr_min) * (Ppr - Ppr_min) - 1.0;
        Tpr_n = 2.0 / (Tpr_max - Tpr_min) * (Tpr - Tpr_min) - 1.0;  
		
        for (int i = 0; i < 10; i++)
        {
            n1_10[i][0] = Ppr_n * wb1_10[i][0] + Tpr_n * wb1_10[i][1] + wb1_10[i][2];
            n1_10[i][1] = logSig(n1_10[i][0]);
        }

        for (int i = 0; i < 10; i++)
        {
            n2_10[i][0] = n1_10[0][1] * wb2_10[i][0] + n1_10[1][1] * wb2_10[i][1] + n1_10[2][1] * wb2_10[i][2] + n1_10[3][1] * wb2_10[i][3] + n1_10[4][1] * wb2_10[i][4] + n1_10[5][1] * wb2_10[i][5] + n1_10[6][1] * wb2_10[i][6] + n1_10[7][1] * wb2_10[i][7] + n1_10[8][1] * wb2_10[i][8] + n1_10[9][1] * wb2_10[i][9] + wb2_10[i][10];
            n2_10[i][1] = logSig(n2_10[i][0]);
        }

        z_n = n2_10[0][1] * wb3_10[0] + n2_10[1][1] * wb3_10[1] + n2_10[2][1] * wb3_10[2] + n2_10[3][1] * wb3_10[3] + n2_10[4][1] * wb3_10[4] + n2_10[5][1] * wb3_10[5] + n2_10[6][1] * wb3_10[6] + n2_10[7][1] * wb3_10[7] + n2_10[8][1] * wb3_10[8] + n2_10[9][1] * wb3_10[9] + wb3_10[10];

        double zAnn10 = (z_n + 1) * (Z_max - Z_min) / 2 + Z_min;  //reverse normalization of normalized z factor.
        
        return zAnn10;
	}


	private static double logSig(double x) {
		return 1 / (1 + Math.exp(-1 * x));
	}
}
