package test.ReportTimesML;

import org.deeplearning4j.datasets.iterator.DoublesDataSetIterator;
import org.deeplearning4j.datasets.iterator.IteratorDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.SgdUpdater;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import tech.tablesaw.api.*;
import tech.tablesaw.columns.Column;

import java.io.IOException;
import java.net.URL;
import java.util.List;
import java.util.function.Consumer;

/**
 * Hello world!
 *
 */
public class App {
    public static void main(String[] args) {
        System.out.println("Report exec times equivalent in Java from https://github.com/abaranovskis-redsamurai/automation-repo/blob/master/regression/report_exec_times.ipynb" );

        // read the CSV file
        try {
            URL url = App.class.getClassLoader().getResource("report_exec_times.csv");
            //File csvFile = new File(url.getFile());
            //String path = csvFile.getAbsolutePath();
            //boolean canRead = csvFile.canRead();
            Table rawDataSet = Table.read().csv(url);
            System.out.println("Table read. Rows=" + rawDataSet.rowCount());

            Table dataSet = rawDataSet.copy();

            IntColumn reportId = dataSet.intColumn("report_id");
            IntColumn intcol1 = reportId.map( x -> x == 1 ? 1 : 0).setName("report_1");
            IntColumn intcol2 = reportId.map( x -> x == 2 ? 1 : 0).setName("report_2");
            IntColumn intcol3 = reportId.map( x -> x == 3 ? 1 : 0).setName("report_3");
            IntColumn intcol4 = reportId.map( x -> x == 4 ? 1 : 0).setName("report_4");
            IntColumn intcol5 = reportId.map( x -> x == 5 ? 1 : 0).setName("report_5");
            dataSet.addColumns(intcol1, intcol2, intcol3, intcol4, intcol5);

            IntColumn dayPart = dataSet.intColumn("day_part");
            IntColumn morning = dayPart.map( x -> x == 1? 1: 0).setName("day_morning");
            IntColumn midday = dayPart.map( x -> x == 2? 1: 0).setName("day_midday");
            IntColumn afternoon = dayPart.map( x -> x == 3? 1: 0).setName("day_afternoon");
            dataSet.addColumns(morning, midday, afternoon);
            dataSet.removeColumns("report_id", "day_part");

            System.out.println(dataSet.last(5).print());

            System.out.println("Splitting training dataset into train (80%) and test data");

            Table[] samples = dataSet.sampleSplit(0.8);
            Table trainingDataSet = samples[0];
            Table testDataSet = samples[1];

            System.out.println("Training shape (" + trainingDataSet.shape() + ")");
            System.out.println("Testing shape (" + testDataSet.shape() + ")");

            //System.out.println(trainingDataSet.summary());

            System.out.println("Describe train dataset, without target feature - exec_time. Mean and std will be used to normalize training data");

            Table summaryTable = Table.create();
            trainingDataSet.columns().forEach(new Consumer<Column<?>>() {
                @Override
                public void accept(Column<?> column) {
                    Table summary = column.summary();
                    StringColumn measure = summary.stringColumn("Measure");
                    if(summaryTable.columnCount() == 0)
                    {
                        summaryTable.addColumns(measure);
                    }
                    DoubleColumn value = summary.doubleColumn("Value");
                    value.setName(column.name());
                    summaryTable.addColumns(value);
                }
            });
            System.out.println(summaryTable.print());

            //transpose
            summaryTable.removeColumns("exec_time");
            StringColumn measures = summaryTable.stringColumn("Measure");
            Table train_stats = Table.create();
            train_stats.addColumns(StringColumn.create("Column"));
            for(String measure : measures)
            {
                train_stats.addColumns(DoubleColumn.create(measure));
            }
            for(int i=1; i<summaryTable.columnCount(); i++)
            {
                DoubleColumn val = summaryTable.doubleColumn(i);
                Row row = train_stats.appendRow();
                row.setString(0, val.name());
                double[] values = val.asDoubleArray();
                for(int j=0;j<values.length;j++) {
                    row.setDouble(j+1, values[j]);
                }
            }
            System.out.println(train_stats.print());

            //# Remove exec_time feature from training data and keep it as a target for both training and testing
            Table train_labels = trainingDataSet.select("exec_time");
            trainingDataSet.removeColumns("exec_time");
            Table test_labels = testDataSet.select("exec_time");
            testDataSet.removeColumns("exec_time");

            //# Neural network learns better, when data is normalized (features look similar to each other)

            //def norm(x):
            //return (x - train_stats['mean']) / train_stats['std']
            //
            //normed_train_data = norm(train_dataset)
            //normed_test_data = norm(test_dataset)

            int i=0;
            for(NumericColumn column: trainingDataSet.numberColumns())
            {
                double mean = train_stats.row(i).getDouble("Mean");
                double std = train_stats.row(i).getDouble("Std. Dev");

                NumericColumn result = column.subtract(mean).divide(std).setName(column.name());
                trainingDataSet.replaceColumn(column.name(), result);
                i++;
            }

            System.out.println("Normalized\n" + trainingDataSet.last(10).print());

            //# Construct neural network with Keras API on top of TensorFlow. Using two layers with 50 units, non linear sigmoid activation, SGD optimizer and
            //# mean squared error loss to check training quality
            //
            //def build_model():
            //  model = keras.Sequential([
            //    layers.Dense(50, activation='sigmoid', input_shape=[len(train_dataset.keys())]),
            //    layers.Dense(50, activation='sigmoid'),
            //    layers.Dense(1)
            //  ])
            //
            //  optimizer = keras.optimizers.SGD(0.001)
            //
            //  model.compile(loss='mean_squared_error',
            //                optimizer=optimizer,
            //                metrics=['mean_absolute_error', 'mean_squared_error'])
            //  return model
            //
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    //.seed(seed)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .updater(new Sgd(0.001))
                    .list()
                    .layer(0, new DenseLayer.Builder().units(50).nIn(trainingDataSet.rowCount())
                            //.weightInit(WeightInit.SIGMOID_UNIFORM)
                            .activation(Activation.SIGMOID)
                            .build())
                    .layer(1, new DenseLayer.Builder().units(50)
                            .activation(Activation.SIGMOID).build())
                    .layer(2, new DenseLayer.Builder().units(1).build())
                    //.layer( 3, new OutputLayer.Builder(LossFunctions.LossFunction.MSE).build())
                    .build();

            MultiLayerNetwork model = new MultiLayerNetwork(conf);

            model.init();
            System.out.println(model.summary());
            //model.addListeners(new EvaluativeListener());

//            # Using 20% of data for training validation
//            history = model.fit(
//                    normed_train_data, train_labels,
//                    epochs=EPOCHS, validation_split = 0.2, batch_size=40, verbose=0,
//                    callbacks=[PrintDot()])

            int numEpochs = 1000;
            model.setEpochCount(numEpochs);


            //model.fit(new NDArray(), new NDArray(train_labels.numberColumn(0).asDoubleArray()));


            System.out.println("Evaluate model....");

            //Evaluation eval = model.evaluate(new DoublesDataSetIterator(), 40);
            //System.out.println(eval.stats());



        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }

    private void normalize(Table input)
    {
        for(NumericColumn column: input.numberColumns()) {
                DoubleColumn normalized = column.normalize();
                input.replaceColumn(normalized);
        }
    }
}
