package it.unipd.dei.yeagerists;

import it.unipd.dei.yeagerists.index.DirectoryIndexer;
import it.unipd.dei.yeagerists.search.Searcher;
import lombok.extern.java.Log;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.core.LowerCaseFilterFactory;
import org.apache.lucene.analysis.core.StopFilterFactory;
import org.apache.lucene.analysis.custom.CustomAnalyzer;
import org.apache.lucene.analysis.standard.StandardTokenizerFactory;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.search.similarities.Similarity;

import java.io.IOException;
import java.util.logging.Handler;
import java.util.logging.Level;
import java.util.logging.LogManager;
import java.util.logging.Logger;

@Log
public class Application {

    public static void main(String... args) throws IOException {

        setLogLevel(Level.INFO);

        final int ramBuffer = 256;
        final String docsPath = "/home/datasets/Args.me/data";
        final String indexPath = "experiment/index-stop-nostem";
        final String extension = "json";
        final String charsetName = "ISO-8859-1";

        final Analyzer analyzer = CustomAnalyzer.builder()
                .withTokenizer(StandardTokenizerFactory.class)
                .addTokenFilter(LowerCaseFilterFactory.class)
                .addTokenFilter(StopFilterFactory.class)
                .build();

        final Similarity similarity = new BM25Similarity();

        final DirectoryIndexer indexer = new DirectoryIndexer(analyzer, similarity, ramBuffer, indexPath, docsPath,
                extension, charsetName);
        indexer.index();

        final String topics = "/home/datasets/Args.me/topics-task-1-only-titles.xml";
        final String runPath = "experiment";
        final String runID = "yeagerists";
        final int maxDocsRetrieved = 1000;

        final Searcher s = new Searcher(analyzer, similarity, indexPath, topics, runID, runPath, maxDocsRetrieved);

        try {
            s.search();
        } catch (ParseException e) {
            log.severe(String.format("Unable to parse query: %s", e.getMessage()));
        }
    }

    private static void setLogLevel(Level level) {
        Logger rootLogger = LogManager.getLogManager().getLogger("");
        rootLogger.setLevel(level);
        for (Handler h : rootLogger.getHandlers()) {
            h.setLevel(level);
        }
    }
}
