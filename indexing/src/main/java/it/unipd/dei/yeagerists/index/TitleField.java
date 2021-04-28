package it.unipd.dei.yeagerists.index;

import it.unipd.dei.yeagerists.parse.ParsedArgument;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.index.IndexOptions;

public class TitleField extends Field {

    private static final FieldType TYPE = new FieldType();

    static {
        TYPE.setIndexOptions(IndexOptions.DOCS_AND_FREQS);
        TYPE.setTokenized(true);
        TYPE.setOmitNorms(false);
        TYPE.setStored(true);
        TYPE.freeze();
    }

    /**
     * Create a new field for the title of an argument.
     *
     * @param value the contents of the title of an argument.
     */
    public TitleField(final String value) {
        super(ParsedArgument.FIELDS.TITLE, value, TYPE);
    }
}
