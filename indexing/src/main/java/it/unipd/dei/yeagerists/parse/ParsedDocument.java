package it.unipd.dei.yeagerists.parse;

import lombok.NonNull;
import org.apache.commons.lang3.builder.ToStringBuilder;
import org.apache.commons.lang3.builder.ToStringStyle;

public class ParsedDocument {

    public final static class FIELDS {
        public static final String ID = "id";
        public static final String BODY = "body";
    }

    private final String id;
    private final String body;

    /**
     * Creates a new parsed document
     *
     * @param id   the unique document identifier.
     * @param body the body of the document.
     * @throws NullPointerException  if {@code id} and/or {@code body} are {@code null}.
     * @throws IllegalStateException if {@code id} and/or {@code body} are empty.
     */
    public ParsedDocument(@NonNull final String id, @NonNull final String body) {

        if (id.isEmpty()) {
            throw new IllegalStateException("Document identifier cannot be empty.");
        }

        if (body.isEmpty()) {
            throw new IllegalStateException("Document body cannot be empty.");
        }

        this.id = id;
        this.body = body;
    }

    public String getIdentifier() {
        return id;
    }

    public String getBody() {
        return body;
    }


    @Override
    public final String toString() {
        ToStringBuilder tsb = new ToStringBuilder(this, ToStringStyle.MULTI_LINE_STYLE).append("identifier", id).append(
                "body", body);

        return tsb.toString();
    }

    @Override
    public final boolean equals(Object o) {
        return (this == o) || ((o instanceof ParsedDocument) && id.equals(((ParsedDocument) o).id));
    }

    @Override
    public final int hashCode() {
        return 37 * id.hashCode();
    }


}
