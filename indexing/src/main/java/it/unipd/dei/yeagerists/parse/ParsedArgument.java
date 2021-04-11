package it.unipd.dei.yeagerists.parse;

import lombok.*;

@Data // Generates getters/setters/tostring/hashcode/allargsconstructor
public class ParsedArgument {

    public final static class FIELDS {
        public static final String ID = "id";
        public static final String BODY = "body";
    }

    @Setter(value = AccessLevel.NONE) // disable setter for this field
    @NonNull
    private final String id;

    @Setter(value = AccessLevel.NONE) // disable setter for this field
    @NonNull
    private final String body;

    public boolean isValid() {
        return !body.isEmpty() && !id.isEmpty();
    }

}
