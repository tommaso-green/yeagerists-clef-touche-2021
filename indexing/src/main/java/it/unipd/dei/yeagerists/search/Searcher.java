package it.unipd.dei.yeagerists.search;

import it.unipd.dei.yeagerists.parse.ParsedArgument;
import lombok.NonNull;
import lombok.extern.java.Log;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.benchmark.quality.QualityQuery;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.queryparser.classic.QueryParserBase;
import org.apache.lucene.search.*;
import org.apache.lucene.search.similarities.Similarity;
import org.apache.lucene.store.FSDirectory;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.*;

@Log
public class Searcher {


    /**
     * The fields of the typical TREC topics.
     */
    private static final class TOPIC_FIELDS {
        public static final String ID = "number";
        public static final String TITLE = "title";
        public static final String DESCRIPTION = "description";
        public static final String NARRATIVE = "narrative";
    }

    private final String runID;
    private final PrintWriter run;
    private final IndexReader reader;
    private final IndexSearcher searcher;

    /**
     * The topics to be searched
     */
    private final List<QualityQuery> topics;

    private final QueryParser qp;
    private final int maxDocsRetrieved;
    private long elapsedTime = Long.MIN_VALUE;


    /**
     * Creates a new searcher.
     *
     * @param analyzer         the {@code Analyzer} to be used.
     * @param similarity       the {@code Similarity} to be used.
     * @param indexPath        the directory where containing the index to be searched.
     * @param topicsFile       the file containing the topics to search for.
     * @param runID            the identifier of the run to be created.
     * @param runPath          the path where to store the run.
     * @param maxDocsRetrieved the maximum number of documents to be retrieved.
     * @throws NullPointerException     if any of the parameters is {@code null}.
     * @throws IllegalArgumentException if any of the parameters assumes invalid values.
     */
    public Searcher(
            @NonNull final Analyzer analyzer,
            @NonNull final Similarity similarity,
            @NonNull final String indexPath,
            @NonNull final String topicsFile,
            @NonNull final String runID,
            @NonNull final String runPath,
            @NonNull final int maxDocsRetrieved) {

        if (indexPath.isEmpty()) {
            throw new IllegalArgumentException("Index path cannot be empty.");
        }

        if (topicsFile.isEmpty()) {
            throw new IllegalArgumentException("Topics file cannot be empty.");
        }

        if (runID.isEmpty()) {
            throw new IllegalArgumentException("Run identifier cannot be empty.");
        }

        if (runPath.isEmpty()) {
            throw new IllegalArgumentException("Run path cannot be empty.");
        }

        final Path indexDir = Paths.get(indexPath);
        if (!Files.isReadable(indexDir)) {
            throw new IllegalArgumentException(
                    String.format("Index directory %s cannot be read.", indexDir.toAbsolutePath().toString()));
        }

        if (!Files.isDirectory(indexDir)) {
            throw new IllegalArgumentException(String.format("%s expected to be a directory where to search the index.",
                    indexDir.toAbsolutePath().toString()));
        }

        try {
            reader = DirectoryReader.open(FSDirectory.open(indexDir));
        } catch (IOException e) {
            throw new IllegalArgumentException(String.format("Unable to create the index reader for directory %s: %s.",
                    indexDir.toAbsolutePath().toString(), e.getMessage()), e);
        }

        searcher = new IndexSearcher(reader);
        searcher.setSimilarity(similarity);

        try {
            topics = parseTopics(topicsFile);
        } catch (IOException e) {
            throw new IllegalArgumentException(
                    String.format("Unable to process topic file %s: %s.", topicsFile, e.getMessage()), e);
        }

        qp = new QueryParser(ParsedArgument.FIELDS.BODY, analyzer);

        this.runID = runID;

        final Path runDir = Paths.get(runPath);
        if (!Files.isWritable(runDir)) {
            throw new IllegalArgumentException(
                    String.format("Run directory %s cannot be written.", runDir.toAbsolutePath().toString()));
        }

        if (!Files.isDirectory(runDir)) {
            throw new IllegalArgumentException(String.format("%s expected to be a directory where to write the run.",
                    runDir.toAbsolutePath().toString()));
        }

        Path runFile = runDir.resolve(runID + ".txt");
        try {
            run = new PrintWriter(Files.newBufferedWriter(runFile, StandardCharsets.UTF_8, StandardOpenOption.CREATE,
                    StandardOpenOption.TRUNCATE_EXISTING,
                    StandardOpenOption.WRITE));
        } catch (IOException e) {
            throw new IllegalArgumentException(
                    String.format("Unable to open run file %s: %s.", runFile.toAbsolutePath(), e.getMessage()), e);
        }

        if (maxDocsRetrieved <= 0) {
            throw new IllegalArgumentException(
                    "The maximum number of documents to be retrieved cannot be less than or equal to zero.");
        }

        this.maxDocsRetrieved = maxDocsRetrieved;
    }

    public long getElapsedTime() {
        return elapsedTime;
    }

    /**
     * Searches for the specified topics.
     *
     * @throws IOException    if something goes wrong while searching.
     * @throws ParseException if something goes wrong while parsing topics.
     */
    public void search() throws IOException, ParseException {

        log.info("%n#### Start searching ####%n");
        log.info("Number of topics: " + topics.size());
        // the start time of the searching
        final long start = System.currentTimeMillis();

        final Set<String> idField = new HashSet<>();
        idField.add(ParsedArgument.FIELDS.ID);

        BooleanQuery.Builder bq = null;
        Query q = null;
        TopDocs docs = null;
        ScoreDoc[] sd = null;
        String docID = null;

        try {
            for (QualityQuery t : topics) {

                log.info(String.format("Searching for topic %s.", t.getQueryID()));

                bq = new BooleanQuery.Builder();

                bq.add(qp.parse(QueryParserBase.escape(t.getValue(TOPIC_FIELDS.TITLE))), BooleanClause.Occur.SHOULD);
                //bq.add(qp.parse(QueryParserBase.escape(t.getValue(TOPIC_FIELDS.DESCRIPTION))), BooleanClause.Occur.SHOULD);

                q = bq.build();

                log.info("Searching for query: " + q.toString());

                docs = searcher.search(q, maxDocsRetrieved);

                sd = docs.scoreDocs;

                for (int i = 0, n = sd.length; i < n; i++) {
                    docID = reader.document(sd[i].doc, idField).get(ParsedArgument.FIELDS.ID);

                    run.printf(Locale.ENGLISH, "%s\tQ0\t%s\t%d\t%.6f\t%s%n", t.getQueryID(), docID, i, sd[i].score,
                            runID);
                }

                run.flush();

            }
        } finally {
            run.close();

            reader.close();
        }

        elapsedTime = System.currentTimeMillis() - start;

        log.info(String.format("%d topic(s) searched in %d seconds.", topics.size(), elapsedTime / 1000));
        log.info("#### Searching complete ####%n");
    }

    /**
     * Parse CLEF topics file (XML format)
     * @param file path to the topics file
     * @return list of QualityQueries with ID equal to the topic number, and title equal to the query body
     * @throws IOException if a reading error occurs
     */
    private List<QualityQuery> parseTopics(String file) throws IOException {
        ArrayList<QualityQuery> res = new ArrayList<>();
        DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();

        try {

            DocumentBuilder db = dbf.newDocumentBuilder();

            Document doc = db.parse(file);

            // optional, but recommended
            doc.getDocumentElement().normalize();
            NodeList list = doc.getElementsByTagName("topic");

            // iterate over topics
            for (int i = 0; i < list.getLength(); i++) {

                HashMap<String,String> fields = new HashMap<>();

                Node node = list.item(i);

                if (node.getNodeType() == Node.ELEMENT_NODE) {

                    Element element = (Element) node;

                    String id = element.getElementsByTagName(TOPIC_FIELDS.ID).item(0).getTextContent();
                    String title = element.getElementsByTagName(TOPIC_FIELDS.TITLE).item(0).getTextContent();

                    fields.put(TOPIC_FIELDS.TITLE, title);

                    QualityQuery topic = new QualityQuery(id, fields);
                    res.add(topic);
                }
            }

        } catch (ParserConfigurationException | IOException | SAXException e) {
            throw new IOException(e);
        }

        return res;
    }
}
