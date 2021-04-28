package it.unipd.dei.yeagerists.search;

import lombok.AllArgsConstructor;
import lombok.Value;
import org.apache.lucene.benchmark.quality.QualityQuery;

import java.util.HashMap;
import java.util.Map;

@Value
@AllArgsConstructor
public class ToucheQuery {

    /**
     * The fields of the TREC-format queries.
     */
    public static final class TOPIC_FIELDS {
        public static final String ID = "number";
        public static final String TITLE = "title";
    }

    String id;
    String title;

    public QualityQuery toQualityQuery() {
        Map<String,String> fields = new HashMap<>();

        fields.put(TOPIC_FIELDS.ID, id);
        fields.put(TOPIC_FIELDS.TITLE, title);

        return new QualityQuery(id, fields);
    }
}
