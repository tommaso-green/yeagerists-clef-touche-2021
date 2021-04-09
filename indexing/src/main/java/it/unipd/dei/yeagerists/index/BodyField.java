package it.unipd.dei.yeagerists.index;

import it.unipd.dei.yeagerists.parse.ParsedDocument;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.index.IndexOptions;


public class BodyField extends Field {

    private static final FieldType BODY_TYPE = new FieldType();

    static {
        BODY_TYPE.setIndexOptions(IndexOptions.DOCS_AND_FREQS);
        BODY_TYPE.setTokenized(true);
        BODY_TYPE.setStored(false);
    }

    /**
     * Create a new field for the body of a document.
     *
     * @param value the contents of the body of a document.
     */
    public BodyField(final String value) {
        super(ParsedDocument.FIELDS.BODY, value, BODY_TYPE);
    }
}
