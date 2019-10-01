import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.util.ArrayList;

import org.encog.engine.network.activation.ActivationTANH;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.PersistBasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.PersistTrainingContinuation;
import org.encog.neural.networks.training.propagation.TrainingContinuation;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

public class Driver {
	//using BTC data as test
	public static final String csv = "Data.csv";
	public static final ArrayList<Double> change = new ArrayList<Double>();

	public static double sd = -1;
	public static double mean = -1;

	public static BasicNetwork network;
	public static TrainingContinuation trainer;

	public static String testNetworkLocation = "Weights";
	public static String trainerLocation = "TrainingContinuation";

	public static final int setSize = 1650; //1650
	public static final int batchSize = 165; //165
	public static final int epochSize = 1000000; //100k
	
	public static double [][] input;
	public static double [][] output;

	public static void main(String[] args) {

		System.out.println("Working directory = " + System.getProperty("user.dir"));
		System.out.println(args.length);
		//args[0] = location, args[1] == boolean create new network
		if(args.length == 2 && Integer.parseInt(args[1]) == 0)
		{
			System.out.println("Loading Network");
			testNetworkLocation = args[0];
		}
		else if(args.length == 2 && Integer.parseInt(args[1]) == 1)
		{
			testNetworkLocation = args[0];
			System.out.println("Generating network at " + testNetworkLocation);
			network = generateNetwork();
		}
		else
		{
			System.out.println("Loading default network");
			loadNetwork(testNetworkLocation, trainerLocation);
			//network = generateNetwork();
		}


		//System.out.println(network.dumpWeightsVerbose());

		loadData(change);

		double[][][] trainingSet = generateTrainingSet(setSize);

		input = trainingSet[0];
		output = trainingSet[1];

		System.out.println("Training set generated");
		MLDataSet dataset = new BasicMLDataSet(input, output);


		final ResilientPropagation train = new ResilientPropagation(network, dataset);

		train.setBatchSize(batchSize);
		long iter = 0;
		double epochCompletion = 0;
		
		do {
			train.iteration();
			iter++;
			epochCompletion = (iter/(setSize/batchSize));

			if(iter%(setSize/batchSize) == 0)
			{
				System.out.println("Epoch #" + epochCompletion + " Iteration #" + iter + " Error: " + train.getError());
			}
			else if(epochCompletion%1000 == 0)
			{
				TrainingContinuation continuation = train.pause();
				saveNetwork(network, testNetworkLocation, continuation, trainerLocation);
				train.resume(continuation);
			}
		} while(epochSize >= iter/(setSize/batchSize)); 
			//while(train.getError() > .92);//
		train.finishTraining();

		System.out.println("Finished training");
		

		System.out.println("Sub-Set Accuracy: " + getLocalAccuracy());
		System.out.println("Full Set Accuracy: " + getActualAccuracy());
		
		/*
		double test[] = {-.76596,.64539,1,.184397,.099291,.198582,.602837};
		double testOut[] = {0};
		
		network.compute(test,testOut);
		System.out.println(testOut[0]);
		*/
		saveNetwork(network, testNetworkLocation, trainer, trainerLocation);
		
		testNetwork(network);
		testNetworkOverall(network);

		System.exit(0);
	}

	public static void loadData(ArrayList<Double> change)
	{
		BufferedReader reader;
		try {
			reader = new BufferedReader(new FileReader(new File(csv)));
			try {
				String [] str = reader.readLine().split(",");
				sd = Double.parseDouble(str[3]);
				mean = Double.parseDouble(str[4]);
				while(reader.ready())
				{
					str = reader.readLine().split(",");
					change.add(Double.parseDouble((str[1])));
				}
				reader.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}

	public static void printData(ArrayList<Double> change)
	{
		System.out.println("Change");
		for(int i = 0; i < change.size(); i++)
		{
			System.out.println(change.get(i));
		}
	}

	public static BasicNetwork generateNetwork()
	{
		BasicNetwork network = new BasicNetwork();
		network.addLayer(new BasicLayer(new ActivationTANH(),true,7));

		network.addLayer(new BasicLayer(new ActivationTANH(),true,25));
		network.addLayer(new BasicLayer(new ActivationTANH(),true,25));
		
		network.addLayer(new BasicLayer(new ActivationTANH(),true,1));
		network.getStructure().finalizeStructure();
		network.reset();
		return network;
	}


	public static double[][][] generateTrainingSet(int size)
	{
		double[][] input = new double[size][7];
		double[][] output = new double[size][1];

		int max = size;//change.size() - 8;

		for(int i = 0; i < size; i++)
		{
			double local = -100;
			int r = (int) (Math.random()*max);
			for(int j = 0; j < 7; j++)
			{
				local = Math.max(local, Math.abs(change.get(r+j)));
			}
			local = Math.abs(local);
			for(int j = 0; j < 7; j++)
			{
				input[i][j] = change.get(r + j)/local;
			}
			if(change.get(r + 7) > 0)
				output[i][0] = 1;
			else if(change.get(r+7) < 0)
				output[i][0] = -1;
			else output[i][0] = 0;
		}

		double[][][] temp = new double[2][][];
		temp[0] = input;
		temp[1] = output;
		return temp;
	}

	public static double getLocalAccuracy()
	{
		double [] inputNodes = new double[7];
		double[] outputNodes = new double[1];

		int numCorrect = 0;
		for(int i = 0; i < input.length; i++)
		{
			for(int j = 0; j < 7; j++)
			{
				inputNodes[j] = input[i][j];
			}

			network.compute(inputNodes, outputNodes);
			
			if(outputNodes[0] > 0 && output[i][0] > 0)
				numCorrect++;
			else if(outputNodes[0] < 0 && output[i][0] < 0)
				numCorrect++;
			
		}
		return (double)(numCorrect)/(input.length);
	}
	
	public static double getActualAccuracy()
	{
		double [] inputNodes = new double[7];
		double[] output = new double[1];

		int numCorrect = 0;
		for(int i = 0; i < change.size() - 8; i++)
		{
			double local = 0;
			for(int j = i; j < i + 7; j++)
			{
				local = Math.max(local, change.get(j));
			}
			local = Math.abs(local);

			for(int j = i; j < i + 7; j++)
			{
				inputNodes[j-i] = change.get(j)/local;
			}
			

			network.compute(inputNodes, output);
			
			if(output[0] > 0 && change.get(i + 7) > 0)
				numCorrect++;
			else if(output[0] < 0 && change.get(i + 7) < 0)
				numCorrect++;
			
		}
		
		return (double)(numCorrect)/(change.size()-8);
	}
	
	public static void testNetwork(BasicNetwork network)
	{
		System.out.println("Testing Network On Trained Set");
		
		double [] inputNodes = new double[7];
		double[] outputNodes = new double[1];

		int day = 0;
		while(day >= 0 && day < input.length)
		{
			System.out.println("Insert day n < " + input.length + ", -1 to exit: ");

			BufferedReader user = new BufferedReader(new InputStreamReader(System.in));
			try {
				day = Integer.parseInt(user.readLine());
				if(day < 0)
				{
					user.close();
					return;
				}
			} catch (NumberFormatException e) {
				e.printStackTrace();
				return;
			} catch (IOException e) {
				e.printStackTrace();
				return;
			}

			double local = 0;
			for(int j = 0; j < 7; j++)
			{
				local = Math.max(local, input[day][j]);
			}
			local = Math.abs(local);

			System.out.println("Change");
			for(int j = 0; j < 7; j++)
			{
				System.out.println(input[day][j]);
				inputNodes[j] = input[day][j]/local;
			}

			network.compute(inputNodes, outputNodes);
			System.out.println("Bullish/Bearish Confidence: " + outputNodes[0]); //inverse sigmoid Math.log(output[0]/(1-output[0]))

			System.out.println("Expected Value: " + (output[day][0]));
		}

	}

	public static void testNetworkOverall(BasicNetwork network)
	{
		System.out.println("Testing Network");
		double [] inputNodes = new double[7];
		double[] output = new double[1];

		int day = 0;
		while(day >= 0 && day < change.size() - 8)
		{
			System.out.println("Insert day n < " + (change.size() - 8) + ", -1 to exit: ");

			BufferedReader input = new BufferedReader(new InputStreamReader(System.in));
			try {
				day = Integer.parseInt(input.readLine());
				if(day < 0)
				{
					input.close();
					return;
				}
			} catch (NumberFormatException e) {
				e.printStackTrace();
				return;
			} catch (IOException e) {
				e.printStackTrace();
				return;
			}

			double local = 0;
			for(int j = day; j < day + 7; j++)
			{
				local = Math.max(local, change.get(j));
			}
			local = Math.abs(local);

			System.out.println("Change");
			for(int j = day; j < day + 7; j++)
			{
				System.out.println(change.get(j));
				inputNodes[j-day] = change.get(j)/local;
			}

			network.compute(inputNodes, output);
			System.out.println("Bullish/Bearish Confidence: " + output[0]); //inverse sigmoid Math.log(output[0]/(1-output[0]))

			System.out.println("Actual Percent Change: " + (change.get(day + 7)));
		}

	}

	public static double denormalize(double zscore)
	{
		return zscore * sd + mean;
	}

	public static void saveNetwork(BasicNetwork network, String networkLoc, TrainingContinuation trainer, String trainLoc)
	{
		//System.out.println("Saved");
		PersistTrainingContinuation saveTrainer = new PersistTrainingContinuation();
		PersistBasicNetwork persister = new PersistBasicNetwork();
		OutputStream writer;
		try {
			writer = new FileOutputStream(networkLoc);
			persister.save(writer, network);
			
			writer = new FileOutputStream(trainLoc);
			saveTrainer.save(writer, trainer);
		} catch (FileNotFoundException e) {
			System.out.println("Failed to save network!");
			e.printStackTrace();
		}
	}


	public static void loadNetwork(String networkLoc, String trainLoc)
	{
		PersistBasicNetwork persister = new PersistBasicNetwork();
		PersistTrainingContinuation saveTrainer = new PersistTrainingContinuation();
		InputStream reader;
		try {
			reader = new FileInputStream(networkLoc);
			network = (BasicNetwork) persister.read(reader);
			
			reader = new FileInputStream(trainLoc);
			trainer = (TrainingContinuation) saveTrainer.read(reader);
		} catch (FileNotFoundException e) {
			System.out.println("Failed to load!");
			e.printStackTrace();
		}
	}

}
