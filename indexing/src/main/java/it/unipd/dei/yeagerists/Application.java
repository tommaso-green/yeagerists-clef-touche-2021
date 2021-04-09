package it.unipd.dei.yeagerists;

import it.unipd.dei.yeagerists.index.DirectoryIndexer;
import it.unipd.dei.yeagerists.parse.ArgsParser;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.core.LowerCaseFilterFactory;
import org.apache.lucene.analysis.core.StopFilterFactory;
import org.apache.lucene.analysis.custom.CustomAnalyzer;
import org.apache.lucene.analysis.standard.StandardTokenizerFactory;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.search.similarities.Similarity;

import java.io.IOException;

public class Application {

    public static void main(String... args) throws IOException {
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
                extension, charsetName, ArgsParser.class);
        indexer.index();
    }
}
