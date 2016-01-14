package com.aluxian.mba;

import com.google.gson.Gson;
import weka.associations.FPGrowth;
import weka.associations.FPGrowth.AssociationRule;
import weka.associations.FPGrowth.AssociationRule.METRIC_TYPE;
import weka.associations.FPGrowth.BinaryItem;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

import java.io.File;
import java.lang.reflect.Field;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

public class Main {

    public static void main(String[] args) throws Exception {
        Scanner scanner = new Scanner(new File(args[0]));
        List<String> lines = new ArrayList<>();

        while (scanner.hasNextLine()) {
            lines.add(scanner.nextLine());
        }

        String delimiter = args[0].endsWith(".csv") ? "," : " ";
        String[] transactions = lines.toArray(new String[lines.size()]);
        List<Result> results = new Main().apply(transactions, delimiter);
        System.out.println(new Gson().toJson(results));
    }

    public List<Result> apply(String[] transactions, String delimiter) throws Exception {
        FPGrowth fp = new FPGrowth();
        fp.setOptions("-P 2 -I -1 -N 10 -T 0 -C 0.9 -D 0.05 -U 1.0 -M 0.001".split(" "));
        fp.buildAssociations(extractDataSet(transactions, delimiter));
        return parseRules(fp.getAssociationRules());
    }

    // Convert the transactions into an Instances object to feed into FP-Growth
    private Instances extractDataSet(String[] transactions, String delimiter) throws Exception {
        List<Attribute> attributes = extractAttributes(transactions, delimiter);
        Instances data = new Instances("affinity", convertToFastVector(attributes), transactions.length);

        for (String transaction : transactions) {
            List<String> items = Arrays.asList(transaction.split(delimiter));
            Instance instance = new Instance(attributes.size());

            for (Attribute attribute : attributes) {
                instance.setValue(attribute, items.contains(attribute.name()) ? 1 : Double.NaN);
            }

            data.add(instance);
        }

        return data;
    }

    // Extract all the attributes that appear in the transactions
    private List<Attribute> extractAttributes(String[] transactions, String delimiter) {
        return Arrays.asList(transactions).stream()
                .map(transaction -> transaction.split(delimiter))
                .map(Arrays::asList)
                .flatMap(List::stream)
                .distinct()
                .map(item -> {
                    FastVector vector = new FastVector(1);
                    vector.addElement("");
                    return new Attribute(item, vector);
                })
                .collect(Collectors.toList());
    }

    // Convert a list of objects to FastVector
    private FastVector convertToFastVector(Collection<?> items) {
        FastVector vector = new FastVector(items.size());
        items.forEach(vector::addElement);
        return vector;
    }

    // Parse the FP-Growth output
    private List<Result> parseRules(List<AssociationRule> rules) {
        return rules.stream().map(rule -> {
            Result result = new Result();

            result.premiseSupport = rule.getPremiseSupport();
            result.consequenceSupport = rule.getConsequenceSupport();

            result.confidence = getMetric(rule, METRIC_TYPE.CONFIDENCE);
            result.lift = getMetric(rule, METRIC_TYPE.LIFT);
            result.leverage = getMetric(rule, METRIC_TYPE.LEVERAGE);
            result.conviction = getMetric(rule, METRIC_TYPE.CONVICTION);

            Function<BinaryItem, String> getName = item -> item.getAttribute().name();

            result.premise = rule.getPremise().stream().map(getName).collect(Collectors.toList());
            result.consequence = rule.getConsequence().stream().map(getName).collect(Collectors.toList());

            return result;
        }).collect(Collectors.toList());
    }

    // Extract a metric value from an AssociationRule
    private double getMetric(AssociationRule rule, METRIC_TYPE metric) {
        try {
            Field metricTypeField = AssociationRule.class.getDeclaredField("m_metricType");
            metricTypeField.setAccessible(true);
            metricTypeField.set(rule, metric);
            return rule.getMetricValue();
        } catch (NoSuchFieldException | IllegalAccessException e) {
            throw new RuntimeException(e);
        }
    }

    public static class Result {

        public double confidence;
        public double lift;
        public double leverage;
        public double conviction;

        public List<String> premise;
        public List<String> consequence;

        public int premiseSupport;
        public int consequenceSupport;

    }

}
