package it.unipd.dei.yeagerists.parse;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import com.fasterxml.jackson.core.io.SerializedString;
import lombok.extern.java.Log;

import java.io.IOException;
import java.io.Reader;
import java.util.Iterator;
import java.util.NoSuchElementException;

@Log
public class ArgsParser implements Iterator<ParsedArgument>, Iterable<ParsedArgument> {

    private ParsedArgument document;
    JsonParser parser;

    public ArgsParser(Reader in) throws IllegalArgumentException {
        JsonFactory jsonFactory = new JsonFactory();
        try {
            this.parser = jsonFactory.createParser(in);

            if (parser.nextToken() != JsonToken.START_OBJECT) {
                log.severe("input doesn't start with '{'");
                throw new IllegalStateException("Input file doesn't start with '{'");
            }

            if (!parser.nextFieldName(new SerializedString("arguments"))) {
                log.severe("input doesn't contain arguments");
                throw new IllegalStateException("input doesn't contain arguments");
            }

            if (parser.nextToken() != JsonToken.START_ARRAY) {
                log.severe("input doesn't contain an array of arguments");
                throw new IllegalStateException("input doesn't contain an array of arguments");
            }

            // Move to first object of array (i.e. first argument) - if no arguments next token will be END_ARRAY
            parser.nextToken();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public final Iterator<ParsedArgument> iterator() {
        return this;
    }

    @Override
    public boolean hasNext() {
        return parser.getCurrentToken() != JsonToken.END_ARRAY;
    }

    @Override
    public final ParsedArgument next() {

        if (!hasNext()) {
            throw new NoSuchElementException("No more documents to parse.");
        }

        try {
            return parse();
        } catch (IOException e) {
            throw new IllegalStateException("Unable to parse token.", e);
        } finally {
            try {
                // end of file
                if (!hasNext()) {
                    parser.close();
                }
            } catch (IOException e) {
                log.warning(String.format("Unable to close the reader: %s", e.getMessage()));
            }
        }

    }

    private ParsedArgument parse() throws IOException {
        String id = null, body = null, stance = null, title = null;

        if (parser.getCurrentToken() != JsonToken.START_OBJECT)
            throw new IllegalStateException("Expecting start of object");

        JsonToken currToken;
        int nestedObjCount = 0;
        while (true) {
            currToken = parser.nextToken();

            if (currToken == JsonToken.START_OBJECT)  {
                nestedObjCount++;
                continue;
            }

            if (currToken == JsonToken.END_OBJECT) {
                if (nestedObjCount > 0) {
                    nestedObjCount--;
                    continue;
                } else {
                    break; // end of document
                }
            }

            // Check if upcoming field is "id" - which is a root field for the document (hence nestedObjCount == 0)
            if (nestedObjCount == 0 && currToken == JsonToken.FIELD_NAME && parser.getText().equals("id")) {
                parser.nextToken();
                log.fine("Found id: " + parser.getText());
                id = parser.getText();
            }

            if (nestedObjCount == 1 && currToken == JsonToken.FIELD_NAME) {

                switch (parser.getText()) {
                    // Text inside nested object "premises"
                    case "text":
                        parser.nextToken();
                        log.fine("Found body: " + parser.getText());
                        if (body != null) {
                            log.warning(String.format("found multiple texts for argument %s", id));
                        }
                        body = parser.getText();
                        break;
                    // Stance inside nested object "premises"
                    case "stance":
                        // Stance inside nested object "premises"
                        parser.nextToken();
                        log.fine("Found stance: " + parser.getText());
                        if (stance != null) {
                            log.warning(String.format("found multiple stances for argument %s", id));
                        }
                        stance = parser.getText();
                        break;
                    // Discussion title inside nested object "context"
                    case "discussionTitle":
                        parser.nextToken();
                        log.fine("Found title: " + parser.getText());
                        if (title != null) {
                            log.warning(String.format("found multiple titles for argument %s", id));
                        }
                        title = parser.getText();
                        break;
                }
            }

        }

        if (title == null) {
            title = "";
        }

        if (id == null || body == null || stance == null) {
            throw new IllegalStateException("Finished parsing document but id or body are null");
        }

        // Move to next argument - if no argument next token will be END_ARRAY
        parser.nextToken();
        return new ParsedArgument(id, body, stance, title);
    }
}
