package hello;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.BufferedWriter;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.streams.ArffFileStream;
import java.io.*;


public class HelloWorld {
    public static boolean flag = false;

    public static void main(String[] args) {
        System.out.println("hello");

        String modelPath = "./micFoalModel.ser";
        MicFoal micFoal;

        // Check if model file exists
        File modelFile = new File(modelPath);
        if (modelFile.exists()) {
            // Load the model
            micFoal = loadModel(modelPath);

        } else {
            // Train and save the model
            micFoal = new MicFoal();
            initial_training(micFoal, "D:\\intelij\\java-hello-world-with-maven\\src\\main\\java\\hello\\entry01.weka.allclass.arff", 25000);
            flag = true;
            initial_training(micFoal, "D:\\intelij\\java-hello-world-with-maven\\src\\main\\java\\hello\\entry10.weka.allclass.arff", 65037);
            saveModel(micFoal, modelPath);
        }
    }

    public static void initial_training(MicFoal micFoal, String filePath, int numInstances) {
        ArffFileStream stream = new ArffFileStream(filePath, -1);
        InstancesHeader header = stream.getHeader();
        int numAttributes = header.numAttributes();
        System.out.println("Number of attributes in header: " + numAttributes);

        Instance instance;
        micFoal.initParameters(12, 0.2, 0.1, numInstances);
        int number = 0;
        while (stream.hasMoreInstances()) {
            instance = stream.nextInstance().getData();
            int classIndex = instance.numAttributes() - 1;
            double classValue = instance.value(classIndex);
            instance.setClassValue(classValue);
            micFoal.trainOnInstanceImpl(instance);
            if (flag)
            {
                csv_write(micFoal._predictedlabel , micFoal._truelabel, micFoal._count, micFoal.multilThresholds);
            }

            number++;
            System.out.println("The number of instance is " + number + " and the last value is " + instance.value(248));
        }
        System.out.println("Finished training on " + filePath);
    }

    public static void csv_write(int predictedLabel, int realLabel, double[] predVector, double[] threVector) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter("output_s1.csv", true))) {
            File file = new File("output_s1.csv");
            if (file.length() == 0) {
                writer.write("PredictLabel y~,RealLabel y,predVector[0],predVector[1],predVector[2],predVector[3],predVector[4],predVector[5],predVector[6],predVector[7],predVector[8],predVector[9],predVector[10],threVector[0],threVector[1],threVector[2],threVector[3],threVector[4],threVector[5],threVector[6],threVector[7],threVector[8],threVector[9],threVector[10],threVector[11]");
                writer.newLine();
            }
            writer.write(predictedLabel + "," + realLabel);
            for (double v : predVector) {
                writer.write("," + v);
            }
            for (double v : threVector) {
                writer.write("," + v);
            }
            writer.newLine();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void saveModel(MicFoal micFoal, String filePath) {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filePath))) {
            oos.writeObject(micFoal);
            System.out.println("Model saved to " + filePath);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static MicFoal loadModel(String filePath) {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath))) {
            MicFoal micFoal = (MicFoal) ois.readObject();
            System.out.println("Model loaded from " + filePath);
            return micFoal;
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
            return null;
        }
    }

    public static void period_result(MicFoal micFoal) {
        System.out.println("-----------The results of periodic evaluation ------------------ ");
        double[] period_results = micFoal.PeriodicEvaluate();
        String message1 = "The precision of the algorithm is : " + period_results[0] + "\n";
        String message2 = "The recall of the algorithm is : " + period_results[1] + "\n";
        String message3 = "The F1 of the algorithm is : " + period_results[2] + "\n";
        String message4 = "The numRowsofConfusionMatrix of the algorithm is : " + period_results[3] + "\n";
        String message5 = "The newbudget of the algorithm is : " + period_results[4] + "\n";
        System.out.println(message1);
        System.out.println(message2);
        System.out.println(message3);
        System.out.println(message4);
        System.out.println(message5);
        String message = "The results of Entry10 \n" + message1 + message2 + message3 + message4 + message5 + "-------------------------------\n";
        try (FileWriter writer = new FileWriter("./output.txt", true)) {
            writer.write(message + System.lineSeparator());
        } catch (IOException e) {
            e.printStackTrace();
        }
        for (int i = 0; i < 10; i++) {
            System.out.println("finish");
        }
    }
}
