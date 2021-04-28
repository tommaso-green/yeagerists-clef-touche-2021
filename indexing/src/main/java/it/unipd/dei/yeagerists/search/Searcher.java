package it.unipd.dei.yeagerists.search;

import it.unipd.dei.yeagerists.parse.ParsedArgument;
import lombok.NonNull;
import lombok.extern.java.Log;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.benchmark.quality.QualityQuery;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.queryparser.classic.MultiFieldQueryParser;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.queryparser.classic.QueryParserBase;
import org.apache.lucene.search.*;
import org.apache.lucene.search.similarities.Similarity;
import org.apache.lucene.store.FSDirectory;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

@Log
public class Searcher {

    private final IndexReader reader;
    private final IndexSearcher searcher;

    /**
     * The topics to be searched
     */
    private final List<QualityQuery> queries;

    private final MultiFieldQueryParser queryParser;

    private final int maxDocsRetrieved;

    /**
     * Creates a new searcher.
     *
     * @param analyzer         the {@code Analyzer} to be used.
     * @param similarity       the {@code Similarity} to be used.
     * @param indexPath        the directory where containing the index to be searched.
     * @param topicsFile       the file containing the topics to search for.
     * @param titleBoost       the boost given to title hits
     * @param maxDocsRetrieved the maximum number of documents to be retrieved.
     * @throws NullPointerException     if any of the parameters is {@code null}.
     * @throws IllegalArgumentException if any of the parameters assumes invalid values.
     */
    public Searcher(
            @NonNull final Analyzer analyzer,
            @NonNull final Similarity similarity,
            @NonNull final String indexPath,
            @NonNull final String topicsFile,
            @NonNull final Float titleBoost,
            @NonNull final int maxDocsRetrieved) {

        if (indexPath.isEmpty()) {
            throw new IllegalArgumentException("Index path cannot be empty.");
        }

        if (topicsFile.isEmpty()) {
            throw new IllegalArgumentException("Topics file cannot be empty.");
        }

        if (maxDocsRetrieved <= 0) {
            throw new IllegalArgumentException(
                    "The maximum number of documents to be retrieved cannot be less than or equal to zero.");
        }

        final Path indexDir = Paths.get(indexPath);
        if (!Files.isReadable(indexDir)) {
            throw new IllegalArgumentException(
                    String.format("Index directory %s cannot be read.", indexDir.toAbsolutePath()));
        }
        if (!Files.isDirectory(indexDir)) {
            throw new IllegalArgumentException(String.format("%s expected to be a directory where to search the index.",
                    indexDir.toAbsolutePath()));
        }

        try {
            reader = DirectoryReader.open(FSDirectory.open(indexDir));
        } catch (IOException e) {
            throw new IllegalArgumentException(String.format("Unable to create the index reader for directory %s: %s.",
                    indexDir.toAbsolutePath(), e.getMessage()), e);
        }

        try {
            this.queries = getQualityQueries(topicsFile);
        } catch (IOException e) {
            throw new IllegalArgumentException(
                    String.format("Unable to process topic file %s: %s.", topicsFile, e.getMessage()), e);
        }

        searcher = new IndexSearcher(reader);
        searcher.setSimilarity(similarity);

        final Map<String, Float> boosts = new HashMap<>();
        boosts.put(ParsedArgument.FIELDS.BODY, 1.0f);
        boosts.put(ParsedArgument.FIELDS.TITLE, titleBoost);
        queryParser = new MultiFieldQueryParser(
                new String[] {ParsedArgument.FIELDS.BODY, ParsedArgument.FIELDS.TITLE},
                analyzer,
                boosts);

        this.maxDocsRetrieved = maxDocsRetrieved;
    }

    private List<QualityQuery> getQualityQueries(String queriesFile) throws IOException {
        List<ToucheQuery> toucheQueries = Utils.parseQueries(queriesFile);
        return toucheQueries.stream()
                .map(ToucheQuery::toQualityQuery)
                .collect(Collectors.toList());
    }

    /**
     * Searches for the specified topics.
     *
     * @throws IOException    if something goes wrong while searching.
     * @throws ParseException if something goes wrong while parsing topics.
     * @return list of documents read from the index.
     */
    public List<ResultArgument> search() throws IOException, ParseException {

        final long start = System.currentTimeMillis();
        List<ResultArgument> resultArguments = new ArrayList<>();

        try {
            for (QualityQuery qualityQuery : queries) {

                log.info(String.format("Searching: %s.", qualityQuery.getQueryID()));

                // Escape any operator from the original query
                String queryText = queryParser.escape(qualityQuery.getValue(ToucheQuery.TOPIC_FIELDS.TITLE));

                // Arguments should contain terms in the queryText
                Query query = new BooleanQuery.Builder()
                        .add(queryParser.parse(queryText), BooleanClause.Occur.SHOULD)
                        .build();

                log.info("Searching for query: " + query.toString());

                TopDocs docs = searcher.search(query, maxDocsRetrieved);
                // Get argument data from the index
                List<ResultArgument> queryResults = Utils.readResultsDocument(reader, qualityQuery.getQueryID(), docs);
                // Accumulate queries results in a single list
                resultArguments.addAll(queryResults);
            }

        } finally {
            reader.close();
        }

        long elapsedTime = System.currentTimeMillis() - start;
        log.info(String.format("%d queries in %d ms.", queries.size(), elapsedTime));
        return resultArguments;
    }
}
