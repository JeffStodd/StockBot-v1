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

	public static BasicNetwork network;
	public static TrainingContinuation trainer;

	public static String testNetworkLocation = "Weights";
	public static String trainerLocation = "TrainingContinuation";

	public static final int setSize = 1650; //1650
	public static final int batchSize = 55; //165
	public static final int epochSize = 1000; //100k
	
	public static double [][] input;
	public static double [][] output;

	public static void main(String[] args) {

		System.out.println("Working directory = " + System.getProperty("user.dir"));
		//args[0] = location, args[1] == boolean create new network
		//user inputs network directory, trainer directory, and create flag = false
		if(args.length == 3 && Integer.parseInt(args[2]) == 0)
		{
			System.out.println("Loading Network");
			testNetworkLocation = args[0];
			trainerLocation = args[1];
			loadNetwork(testNetworkLocation, trainerLocation);
		}
		//user inputs network directory, trainer directory and create flag = true, generates network
		else if(args.length == 3 && Integer.parseInt(args[2]) == 1)
		{
			testNetworkLocation = args[0];
			trainerLocation = args[1];
			System.out.println("Generating network at " + testNetworkLocation);
			network = generateNetwork();
		}
		//user inputs nothing, network and training session is loaded from default locations (Project directory)
		else
		{
			System.out.println("Loading default network");
			loadNetwork(testNetworkLocation, trainerLocation);
			//network = generateNetwork();
		}


		//System.out.println(network.dumpWeightsVerbose());

		//reads data from csv file
		loadData(change);

		double[][][] trainingSet = generateTrainingSet(setSize);

		input = trainingSet[0];
		output = trainingSet[1];
		
		System.out.println("Training set generated");
		MLDataSet dataset = new BasicMLDataSet(input, output);

		final ResilientPropagation train = new ResilientPropagation(network, dataset);
		train.setBatchSize(batchSize);
		//if previous training session 
		if(args.length != 3)
		{
			train.resume(trainer);
			System.out.println("Loaded Training Session");
		}
		
		long iter = 0;
		double epochCompleted = 0;
		
		//while epoch max >= current epoch and accuracy of full dataset is < 90%
		while((epochSize >= iter/(setSize/batchSize) & getActualAccuracy() < 0.90)) //while(train.getError() > .92);//
		{
			train.iteration();
			iter++;
			epochCompleted = (iter/(setSize/batchSize));

			//if one epoch is complete
			if(iter%(setSize/batchSize) == 0)
			{
				System.out.println("Epoch #" + epochCompleted + " Acc: " + getActualAccuracy() + " Loss: " + train.getError());
			}
			//if 1000 epoches are completed, save network and trainer data
			else if(epochCompleted%1000 == 0)
			{
				trainer = train.pause();
				saveNetwork(network, testNetworkLocation, trainer, trainerLocation);
			}
		}
		train.finishTraining();

		System.out.println("\nFinished training\n");
		

		System.out.println("Sub-Set Accuracy: " + getLocalAccuracy());
		System.out.println("Full Set Accuracy: " + getActualAccuracy());
		
		/*
		//test scenario
		double test[] = {
				0.286486486,
				-1,
				-0.016216216,
				-0.275675676,
				-0.837837838,
				0.297297297,
				.778378
		};
		double testOut[] = {0};
		
		network.compute(test,testOut);
		System.out.println(testOut[0]);
		*/
		
		saveNetwork(network, testNetworkLocation, trainer, trainerLocation);
	
		testNetwork(network); //test network on trained dataset
		testNetworkOverall(network); //test network on complete dataset

		System.exit(0);
	}

	public static void loadData(ArrayList<Double> change)
	{
		BufferedReader reader;
		try {
			reader = new BufferedReader(new FileReader(new File(csv)));
			try {
				String [] str = reader.readLine().split(",");
				//sd = Double.parseDouble(str[3]);
				//mean = Double.parseDouble(str[4]);
				
				//loading percent change from second column into change arraylist
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

	//prints data in percent change arraylist
	public static void printData(ArrayList<Double> change)
	{
		System.out.println("Change");
		for(int i = 0; i < change.size(); i++)
		{
			System.out.println(change.get(i));
		}
	}

	//generate network
	//7 days, 7 input nodes
	//1 output, confidence level of bearish/bullish
	public static BasicNetwork generateNetwork()
	{
		BasicNetwork network = new BasicNetwork();
		network.addLayer(new BasicLayer(new ActivationTANH(),true,7));

		//hidden layers
		//activation tanh for values -1 through 1
		network.addLayer(new BasicLayer(new ActivationTANH(),true,25));
		network.addLayer(new BasicLayer(new ActivationTANH(),true,25));
		network.addLayer(new BasicLayer(new ActivationTANH(),true,25));
		network.addLayer(new BasicLayer(new ActivationTANH(),true,25));
		
		network.addLayer(new BasicLayer(new ActivationTANH(),true,1));
		network.getStructure().finalizeStructure();
		network.reset();
		return network;
	}

	//normalizes and shuffles data
	public static double[][][] generateTrainingSet(int size)
	{
		double[][] input = new double[size][7];
		double[][] output = new double[size][1];

		int max = size;//change.size() - 8;

		for(int i = 0; i < size; i++)
		{
			double local = -100;
			int r = (int) (Math.random()*max); //random start index
			
			//get absolute max of percent change over 7 days
			for(int j = 0; j < 7; j++)
			{
				local = Math.max(local, Math.abs(change.get(r+j)));
			}
			local = Math.abs(local);
			
			//scale data over 7 days based off the local absolute max
			for(int j = 0; j < 7; j++)
			{
				input[i][j] = change.get(r + j)/local;
			}
			
			//set expected percent change
			//1 = bullish, 0 = no change, -1 = bearish
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

	//gets network accuracy on trained dataset
	public static double getLocalAccuracy()
	{
		double [] outputNodes = new double[1];

		int numCorrect = 0;
		for(int i = 0; i < input.length; i++)
		{

			network.compute(input[i], outputNodes);
			
			if(outputNodes[0] > 0 && output[i][0] > 0)
				numCorrect++;
			else if(outputNodes[0] < 0 && output[i][0] < 0)
				numCorrect++;
			else if(outputNodes[0] == 0 && output[i][0] == 0)
				numCorrect++;
			
		}
		//return percent correct
		return (double)(numCorrect)/(input.length);
	}
	
	//gets network accuracy on entire dataset loaded from csv
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
				local = Math.max(local, Math.abs(change.get(j))); //get local absolute max
			}

			for(int j = i; j < i + 7; j++)
			{
				inputNodes[j-i] = change.get(j)/local; //scale csv data by local absolute max and insert into inputNodes
			}
			
			//compute prediction
			network.compute(inputNodes, output);
			
			if(output[0] > 0 && change.get(i + 7) > 0)
				numCorrect++;
			else if(output[0] < 0 && change.get(i + 7) < 0)
				numCorrect++;
			else if(output[0] == 0 && change.get(i + 7) == 0)
				numCorrect++;
			
		}
		//return percent correct
		return (double)(numCorrect)/(change.size()-8);
	}
	
	//tests network on trained dataset via user input
	public static void testNetwork(BasicNetwork network)
	{
		System.out.println("\nTesting Network On Trained Set");
		
		double[] outputNodes = new double[1];

		int day = 0;
		//if day is out of bounds
		while(day >= 0 && day < input.length)
		{
			System.out.println("\nInsert day n < " + input.length + ", -1 to exit: ");

			BufferedReader user = new BufferedReader(new InputStreamReader(System.in));
			try {
				day = Integer.parseInt(user.readLine());
				if(day < 0) //if day is negative, return
				{
					return;
				}
			} catch (NumberFormatException e) {
				e.printStackTrace();
				return;
			} catch (IOException e) {
				e.printStackTrace();
				return;
			}

			System.out.println("Input Nodes:");
			for(int j = 0; j < 7; j++)
			{
				System.out.println(j + ": " + input[day][j]); //prints data of day + 7
			}
			
			//compute prediction
			network.compute(input[day], outputNodes);
			System.out.println("Bullish/Bearish Confidence: " + outputNodes[0]);

			System.out.println("Expected Value: " + (output[day][0]));
		}

	}

	//tests network on entire dataset loaded from csv via user input
	public static void testNetworkOverall(BasicNetwork network)
	{
		System.out.println("\nTesting Network on entire csv dataset");
		double [] inputNodes = new double[7];
		double[] output = new double[1];

		int day = 0;
		//if day is negative or day + 7 is out of bounds
		while(day >= 0 && day < change.size() - 8)
		{
			System.out.println("\nInsert day n < " + (change.size() - 8) + ", -1 to exit: ");

			BufferedReader input = new BufferedReader(new InputStreamReader(System.in));
			try {
				day = Integer.parseInt(input.readLine());
				if(day < 0) //return if negative
				{
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
				local = Math.max(local, Math.abs(change.get(j))); //get local absolute max
			}
			local = Math.abs(local);

			System.out.println("Change");
			for(int j = day; j < day + 7; j++)
			{
				System.out.println(change.get(j));
				inputNodes[j-day] = change.get(j)/local; //scale data by local absolute max
			}

			network.compute(inputNodes, output);
			System.out.println("Bullish/Bearish Confidence: " + output[0]); //inverse sigmoid Math.log(output[0]/(1-output[0]))

			System.out.println("Actual Percent Change: " + (change.get(day + 7)));
		}

	}

	/*
	public static double denormalize(double zscore)
	{
		return zscore * sd + mean;
	}
	*/

	//saves network and trainer to specified locations
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

	//loads network and trainer from specified locations
	public static void loadNetwork(String networkLoc, String trainLoc)
	{
		PersistBasicNetwork persister = new PersistBasicNetwork();
		PersistTrainingContinuation loadTrainer = new PersistTrainingContinuation();
		InputStream reader;
		try {
			reader = new FileInputStream(networkLoc);
			network = (BasicNetwork) persister.read(reader);
			
			reader = new FileInputStream(trainLoc);
			trainer = (TrainingContinuation) loadTrainer.read(reader);
		} catch (FileNotFoundException e) {
			System.out.println("Failed to load!");
			e.printStackTrace();
		}
	}

}
