package test.ReportTimesML;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.time.LocalDate;
import java.util.function.Consumer;
import java.util.function.Function;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import tech.tablesaw.api.*;
import tech.tablesaw.columns.Column;
import tech.tablesaw.columns.numbers.FloatParser;

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
            Table transposedSummary = Table.create();
            transposedSummary.addColumns(StringColumn.create("Column"));
            for(String measure : measures)
            {
                transposedSummary.addColumns(DoubleColumn.create(measure));
            }
            for(int i=1; i<summaryTable.columnCount(); i++)
            {
                DoubleColumn val = summaryTable.doubleColumn(i);
                Row row = transposedSummary.appendRow();
                row.setString(0, val.name());
                double[] values = val.asDoubleArray();
                for(int j=0;j<values.length;j++) {
                    row.setDouble(j+1, values[j]);
                }
            }
            System.out.println(transposedSummary.print());

            //# Remove exec_time feature from training data and keep it as a target for both training and testing
            Table train_labels = trainingDataSet.select("exec_time");
            trainingDataSet.removeColumns("exec_time");
            Table test_labels = testDataSet.select("exec_time");
            testDataSet.removeColumns("exec_time");

            //# Neural network learns better, when data is normalized (features look similar to each other)

            //def norm(x):
            //return (x - train_stats['mean']) / train_stats['std']

            //normed_train_data = norm(train_dataset)
            //normed_test_data = norm(test_dataset)

            for(Column column: trainingDataSet.columns())
            {
                if(column instanceof DoubleColumn)
                {
                    DoubleColumn normalized = ((DoubleColumn) column).normalize();
                    trainingDataSet.replaceColumn(normalized);
                }
            }

            System.out.println("Normalized\n" + trainingDataSet.print());

            //
            //MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            //        .

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
