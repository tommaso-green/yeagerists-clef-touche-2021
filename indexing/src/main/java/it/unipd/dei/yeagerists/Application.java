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

        final String docsPath = cli.getOptionValue("dataset");
        final String indexPath = cli.getOptionValue("output");

        final int ramBuffer = 256;
        final String extension = "json";
        final String charsetName = "ISO-8859-1";

        final Analyzer analyzer = CustomAnalyzer.builder()
                .withTokenizer(StandardTokenizerFactory.class)
                .addTokenFilter(LowerCaseFilterFactory.class)
                .addTokenFilter(StopFilterFactory.class)
                .addTokenFilter(EnglishPossessiveFilterFactory.class)
                .addTokenFilter(KStemFilterFactory.class)
                .build();

        final Similarity similarity = new BM25Similarity();

        final DirectoryIndexer indexer = new DirectoryIndexer(analyzer, similarity, ramBuffer, indexPath, docsPath,
                extension, charsetName);
        indexer.index();

        if (cli.hasOption("topics") && cli.hasOption("run")) {
            final String topics = cli.getOptionValue("topics");
            final String runPath = cli.getOptionValue("run");

            final String runID = "yeagerists";
            final int maxDocsRetrieved = 1000;

            final Searcher s = new Searcher(analyzer, similarity, indexPath, topics, runID, runPath, maxDocsRetrieved);

            try {
                s.search();
            } catch (org.apache.lucene.queryparser.classic.ParseException e) {
                log.severe(String.format("Unable to parse query: %s", e.getMessage()));
            }
        }
    }

    private static CommandLine parseArgs(String[] args) throws org.apache.commons.cli.ParseException {
        Options options = new Options();
        options.addRequiredOption("d", "dataset", true, "Path to Args.me dataset");
        options.addRequiredOption("o", "output", true, "Path to directory where the index will be stored");
        options.addOption("t", "topics", true, "Path to the topics file");
        options.addOption("r", "run", true, "Path where the run file will be stored");

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
