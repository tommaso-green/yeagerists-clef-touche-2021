package it.unipd.dei.yeagerists.search;

import it.unipd.dei.yeagerists.parse.ParsedArgument;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class Utils {
    /**
     * Parse a query file with the following XML format:
     *
     * <topics>
     * <topic>
     * <number>1</number>
     * <title>Query title</title>
     * </topic>
     * ...more topics
     * </topics>
     *
     * @param file path to the topics file
     * @return list of ToucheQueries
     * @throws IOException if a reading error occurs
     */
    public static List<ToucheQuery> parseQueries(String file) throws IOException {
        List<ToucheQuery> res = new ArrayList<>();
        DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();

        try {

            DocumentBuilder db = dbf.newDocumentBuilder();

            Document doc = db.parse(file);

            // optional, but recommended
            doc.getDocumentElement().normalize();
            NodeList list = doc.getElementsByTagName("topic");

            // iterate over topics
            for (int i = 0; i < list.getLength(); i++) {

                Node node = list.item(i);

                if (node.getNodeType() == Node.ELEMENT_NODE) {

                    Element element = (Element) node;

                    String id = element.getElementsByTagName(ToucheQuery.TOPIC_FIELDS.ID).item(0).getTextContent();
                    String title = element.getElementsByTagName(ToucheQuery.TOPIC_FIELDS.TITLE).item(0).getTextContent();

                    ToucheQuery topic = new ToucheQuery(id, title);
                    res.add(topic);
                }
            }

        } catch (ParserConfigurationException | IOException | SAXException e) {
            throw new IOException(e);
        }

        return res;
    }

    /**
     * Read and create ResultDocuments from the given index.
     * @param queryId query identifier
     * @param docs Score documents used to retrieve original documents
     * @return  Result documents loaded from index
     * @throws IOException if an error occurs when reading the index
     */
    public static List<ResultArgument> readResultsDocument(
            IndexReader reader,
            String queryId,
            TopDocs docs
    ) throws IOException {
        List<ResultArgument> resultArguments = new ArrayList<>();
        final Set<String> fields = new HashSet<>();
        fields.add(ParsedArgument.FIELDS.ID);
        fields.add(ParsedArgument.FIELDS.BODY);
        fields.add(ParsedArgument.FIELDS.STANCE);

        ScoreDoc[] scoreDocs = docs.scoreDocs;

        for (ScoreDoc sc : scoreDocs) {
            org.apache.lucene.document.Document indexedDoc = reader.document(sc.doc, fields);
            ResultArgument toAdd = ResultArgument.builder()
                    .score(sc.score)
                    .queryId(queryId)
                    .id(indexedDoc.get(ParsedArgument.FIELDS.ID))
                    .body(indexedDoc.get(ParsedArgument.FIELDS.BODY))
                    .stance(indexedDoc.get(ParsedArgument.FIELDS.STANCE))
                    .build();

            resultArguments.add(toAdd);
        }

        return resultArguments;
    }
}
