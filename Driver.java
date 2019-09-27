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
import java.util.Arrays;

import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.engine.network.activation.ActivationTANH;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.PersistBasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

public class Driver {
	//using BTC data as test
	public static final String csv = "Data.csv";
	public static final ArrayList<Double> change = new ArrayList<Double>();
	
	private static double sd = -1;
	private static double mean = -1;

	private static BasicNetwork network;

	public static String testNetworkLocation = "C:/Users/jeffr/Desktop/NeuralNetwork";

	public static void main(String[] args) {

		System.out.println("Working Directory = " + System.getProperty("user.dir"));
		//args[0] = location, args[1] == boolean create new network
		if(args.length == 2 && Integer.parseInt(args[1]) == 0)
		{
			System.out.println("Loading Network");
			testNetworkLocation = args[0];
		}
		else if(args.length == 2 && Integer.parseInt(args[1]) == 1)
		{
			testNetworkLocation = args[0];
			System.out.println("Generating Network at " + testNetworkLocation);
			network = generateNetwork();
		}
		else
		{
			System.out.println("Generating New Network");
			network = loadNetwork(testNetworkLocation);
		}


		//System.out.println(network.dumpWeightsVerbose());

		loadData(change);

		double[][][] trainingSet = generateTrainingSet(1650);
		
		double [][] input = trainingSet[0];
		double [][] output = trainingSet[1];

		System.out.println("Training set generated");
		MLDataSet trainer = new BasicMLDataSet(input, output);

		final ResilientPropagation train = new ResilientPropagation(network, trainer);
		//final QuickPropagation train = new QuickPropagation(test, trainer);
		do {
			train.iteration();
			if(Double.isNaN(train.getError()))
			{
				network.reset();
				System.out.println("Resetting Bugged Network...");
			}
			else break;
		} while(true);
		
		int epoch = 0;
		do {
			train.iteration();
			System.out.println("Epoch #" + epoch + " Error:" + train.getError());
			epoch++;
			if(epoch % 2500 == 0)
				saveNetwork(network, testNetworkLocation);
		} while(train.getError() > 0.04 && epoch <= 5000);
		saveNetwork(network, testNetworkLocation);
		train.finishTraining();

		System.out.println("Finished training");
		testNetwork(network);
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
		network.addLayer(new BasicLayer(null,true,7));
		network.addLayer(new BasicLayer(new ActivationTANH(),true,15));
		network.addLayer(new BasicLayer(new ActivationTANH(),true,15));
		network.addLayer(new BasicLayer(new ActivationTANH(),true,15));
		network.addLayer(new BasicLayer(new ActivationTANH(),true,15));
		network.addLayer(new BasicLayer(new ActivationTANH(),true,15));
		network.addLayer(new BasicLayer(new ActivationTANH(),true,15));
		network.addLayer(new BasicLayer(new ActivationTANH(),true,15));
		network.addLayer(new BasicLayer(new ActivationTANH(),true,1));
		network.getStructure().finalizeStructure();
		network.reset();
		return network;
	}
	
	
	public static double[][][] generateTrainingSet(int size)
	{
		double[][] input = new double[size][7];
		double[][] output = new double[size][1];

		int max = change.size() - 8;

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
	
	

	public static void testNetwork(BasicNetwork network)
	{
		System.out.println("Testing Network");
		double [] inputNodes = new double[14];
		double[] output = new double[2];

		while(true)
		{
			System.out.println("Insert day n < 321: ");
			int day = 0;
			BufferedReader input = new BufferedReader(new InputStreamReader(System.in));
			try {
				day = Integer.parseInt(input.readLine());
				if(day < 0)
					break;
			} catch (NumberFormatException e) {
				break;
			} catch (IOException e) {
				break;
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
			System.out.println("Bullish/Bearish Prediction: " + output[0]); //inverse sigmoid Math.log(output[0]/(1-output[0]))

			System.out.println("Expected: " + (change.get(day + 7)));
		}

	}
	
	public static double denormalize(double zscore)
	{
		return zscore * sd + mean;
	}
	
	public static void saveNetwork(BasicNetwork network, String file)
	{
		System.out.println("Saved");
		PersistBasicNetwork persister = new PersistBasicNetwork();
		OutputStream writer;
		try {
			writer = new FileOutputStream(file);
			persister.save(writer, network);
		} catch (FileNotFoundException e) {
			System.out.println("Failed to save network!");
			e.printStackTrace();
		}
	}


	public static BasicNetwork loadNetwork(String file)
	{
		PersistBasicNetwork persister = new PersistBasicNetwork();
		InputStream reader;
		try {
			reader = new FileInputStream(file);
			BasicNetwork network = (BasicNetwork) persister.read(reader);
			return network;
		} catch (FileNotFoundException e) {
			System.out.println("Failed to load network!");
			e.printStackTrace();
			return null;
		}
	}

}
