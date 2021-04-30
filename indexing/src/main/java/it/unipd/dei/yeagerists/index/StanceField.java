package it.unipd.dei.yeagerists.index;

import it.unipd.dei.yeagerists.parse.ParsedArgument;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.index.IndexOptions;

public class StanceField extends Field {
    private static final FieldType TYPE = new FieldType();

    static {
        TYPE.setIndexOptions(IndexOptions.DOCS);
        TYPE.setOmitNorms(true);
        TYPE.setTokenized(false);
        TYPE.setStored(true);
        TYPE.freeze();
    }

    /**
     * Create a new field for the stance of an argument.
     *
     * @param stance the stance, can be PRO or CON
     * @throws IllegalArgumentException if stance is not PRO or CON
     */
    public StanceField(final String stance) throws IllegalArgumentException{
        super(ParsedArgument.FIELDS.STANCE, stance.toLowerCase(), TYPE);
        if (!stance.equalsIgnoreCase("pro") && !stance.equalsIgnoreCase("con")) {
            throw new IllegalArgumentException(String.format("%s is not a valid stance", stance));
        }
    }
}
