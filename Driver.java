import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.train.strategy.Strategy;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.quick.QuickPropagation;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.neural.networks.training.strategy.SmartMomentum;

public class Driver {
	//testing with BTC data
	public static final String csv = "StockPredictor/Data.csv";
	public static final ArrayList<Double> change = new ArrayList<Double>();
	public static final ArrayList<Double> volume = new ArrayList<Double>();

	public static void main(String[] args) {
		loadData(change, volume);

		BasicNetwork test = generateNetwork();
		
		//batch size = 9000
		double[][][] trainingSet = generateTrainingSet(9000);
		double [][] input = trainingSet[0];
		double [][] output = trainingSet[1];
		
		System.out.println("Training set generated");
		MLDataSet trainer = new BasicMLDataSet(input, output);

		final ResilientPropagation train = new ResilientPropagation(test, trainer);
		//final QuickPropagation train = new QuickPropagation(test, trainer);
		
		int epoch = 1;

		do {
			train.iteration();
			System.out.println("Epoch #" + epoch + " Error:" + train.getError());
			epoch++;
			if(epoch % 5000 == 0) //stop at 5000 iterations
				break; //save network
		} while(train.getError() > 0.01);
		train.finishTraining();
		
		System.out.println("Finished training");
		
		testNetwork(test);

	}

	public static void loadData(ArrayList<Double> change, ArrayList<Double> volume)
	{
		BufferedReader reader;
		try {
			reader = new BufferedReader(new FileReader(new File(csv)));
			try {
				reader.readLine(); //ignore titles
				while(reader.ready())
				{
					String[] str = reader.readLine().split(","); //split price into [0] and volume into [1]
					change.add(Double.parseDouble((str[0])));
					volume.add(Double.parseDouble((str[1])));
				}
				reader.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}

	public static void printData(ArrayList<Double> change, ArrayList<Double> volume)
	{
		//dumps data into console
		System.out.println("Change : Volume");
		for(int i = 0; i < change.size(); i++)
		{
			System.out.println(change.get(i) + " : " + volume.get(i));
		}
	}



	public static BasicNetwork generateNetwork()
	{
		//activation function = sigmoid
		//first 7 input nodes are price change
		//last 7 input nodes are volume change
		//3 hidden layers
		//output layer with predicted price and volume change
		BasicNetwork network = new BasicNetwork();
		network.addLayer(new BasicLayer(new ActivationSigmoid(), false, 14));
		network.addLayer(new BasicLayer(new ActivationSigmoid(),true,31));
		network.addLayer(new BasicLayer(new ActivationSigmoid(),true,31));
		network.addLayer(new BasicLayer(new ActivationSigmoid(),true,31));
		network.addLayer(new BasicLayer(null,true,2));
		network.getStructure().finalizeStructure();
		network.reset();
		return network;
	}
	
	public static double[][][] generateTrainingSet(int size)
	{
		double[][] input = new double[size][14];
		double[][] output = new double[size][2];
		
		//necessary to prevent out of bounds so index isn't chosen between n-8 and n
		int max = change.size() - 8;
		
		//i < batch size
		for(int i = 0; i < size; i++)
		{
			//setting first 7 nodes
			int r = (int) (Math.random()*max);
			for(int j = 0; j < 7; j++)
			{
				input[i][j] = change.get(r + j);
			}
			//setting last 7 nodes
			for(int j = 0; j < 7; j++)
			{
				input[i][j + 7] = volume.get(r + j);
			}
			output[i][0] = change.get(r + 7);
			output[i][1] = volume.get(r + 7);
		}
		
		double[][][] temp = new double[2][][];
		temp[0] = input;
		temp[1] = output;
		return temp;
	}
	
	//test first 7 days and predict 8th in data.csv
	public static void testNetwork(BasicNetwork network)
	{
		System.out.println("Testing Network");
		double [] inputNodes = new double[14];
		double[] output = new double[2];
		System.out.println("Change");
		for(int j = 0; j < 7; j++)
		{
			inputNodes[j] = change.get(j);
			System.out.println(inputNodes[j]);
		}
		System.out.println("Volume");
		for(int j = 0; j < 7; j++)
		{
			inputNodes[j + 7] = volume.get(j);
			System.out.println(inputNodes[j + 7]);
		}
		
		network.compute(inputNodes, output);
		System.out.println("Actual: " + Arrays.toString(output));
		
		System.out.println("Expected: " + change.get(7) + " | " + volume.get(7));
	}

}
