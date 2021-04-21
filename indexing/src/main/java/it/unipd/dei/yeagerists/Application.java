package it.unipd.dei.yeagerists;

import it.unipd.dei.yeagerists.index.DirectoryIndexer;
import it.unipd.dei.yeagerists.search.Searcher;
import lombok.extern.java.Log;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.Options;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.core.LowerCaseFilterFactory;
import org.apache.lucene.analysis.core.StopFilterFactory;
import org.apache.lucene.analysis.custom.CustomAnalyzer;
import org.apache.lucene.analysis.en.EnglishPossessiveFilterFactory;
import org.apache.lucene.analysis.en.KStemFilterFactory;
import org.apache.lucene.analysis.standard.StandardTokenizerFactory;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.search.similarities.Similarity;

import java.io.IOException;
import java.util.logging.Handler;
import java.util.logging.Level;
import java.util.logging.LogManager;
import java.util.logging.Logger;

@Log
public class Application {

    public static void main(String[] args) throws IOException {

        setLogLevel(Level.INFO);

        CommandLine cli;
        try {
            cli = parseArgs(args);
        } catch (org.apache.commons.cli.ParseException e) {
            log.severe(String.format("Error parsing arguments: %s", e.getMessage()));
            return;
        }

        if (cli.hasOption("index") == cli.hasOption("search")) {
            log.severe("Choose either index or search!");
            return;
        }

        final Analyzer analyzer = CustomAnalyzer.builder()
                .withTokenizer(StandardTokenizerFactory.class)
                .addTokenFilter(LowerCaseFilterFactory.class)
                .addTokenFilter(StopFilterFactory.class)
                .addTokenFilter(EnglishPossessiveFilterFactory.class)
                .addTokenFilter(KStemFilterFactory.class)
                .build();

        final Similarity similarity = new BM25Similarity();

        if (cli.hasOption("index")) {
            final String docsPath = cli.getOptionValue("dataset");
            final String indexPath = cli.getOptionValue("output");

            final int ramBuffer = 256;
            final String extension = "json";
            final String charsetName = "ISO-8859-1";

            final DirectoryIndexer indexer = new DirectoryIndexer(analyzer, similarity, ramBuffer, indexPath, docsPath,
                    extension, charsetName);
            indexer.index();
        }

        if (cli.hasOption("search")) {
            final String indexPath = cli.getOptionValue("path");
            final String queriesPath = cli.getOptionValue("queries");
            final String resultsPath = cli.getOptionValue("results");
            final int maxDocsRetrieved = Integer.parseInt(cli.getOptionValue("max", "100"));

            final Searcher s = new Searcher(analyzer, similarity, indexPath, queriesPath, resultsPath, maxDocsRetrieved);

            try {
                s.search();
            } catch (org.apache.lucene.queryparser.classic.ParseException e) {
                log.severe(String.format("Unable to parse query: %s", e.getMessage()));
            }
        }
    }

    private static CommandLine parseArgs(String[] args) throws org.apache.commons.cli.ParseException {
        Options options = new Options();

        // Indexing options
        options.addOption("i", "index", false, "Index the given dataset");
        options.addOption("d", "dataset", true, "Path to Args.me dataset when indexing");
        options.addOption("o", "output", true, "Path to directory where the index will be stored when indexing");

        // Searching options
        options.addOption("s", "search", false, "Run a query in the given index");
        options.addOption("p", "path", true, "Path to the index");
        options.addOption("q", "queries", true, "Path to the queries to run (XML format)");
        options.addOption("r", "results", true, "Path to the file where results will be written");
        options.addOption("m", "max", true, "Maximum number of documents to retrieve");

        CommandLineParser parser = new DefaultParser();
        return parser.parse(options, args);
    }

    private static void setLogLevel(Level level) {
        Logger rootLogger = LogManager.getLogManager().getLogger("");
        rootLogger.setLevel(level);
        for (Handler h : rootLogger.getHandlers()) {
            h.setLevel(level);
        }
    }
}
