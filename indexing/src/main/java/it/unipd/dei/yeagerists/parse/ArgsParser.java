package it.unipd.dei.yeagerists.parse;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import com.fasterxml.jackson.core.io.SerializedString;
import lombok.extern.java.Log;
import lombok.extern.slf4j.Slf4j;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Reader;
import java.util.Iterator;
import java.util.NoSuchElementException;

@Log
public class ArgsParser implements Iterator<ParsedDocument>, Iterable<ParsedDocument> {

    private ParsedDocument document;
    JsonParser parser;

    public ArgsParser(Reader in) {
        JsonFactory jsonFactory = new JsonFactory();
        try {
            this.parser = jsonFactory.createParser(in);

            if (parser.nextToken() != JsonToken.START_OBJECT) {
                log.severe("input doesn't start with '{'");
                return;
            }

            if (!parser.nextFieldName(new SerializedString("arguments"))) {
                log.severe("input doesn't contain arguments");
                return;
            }

            if (parser.nextToken() != JsonToken.START_ARRAY) {
                log.severe("input doesn't contain an array of arguments");
                return;
            }

            // Move to first object of array (i.e. first argument) - if no arguments next token will be END_ARRAY
            parser.nextToken();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public final Iterator<ParsedDocument> iterator() {
        return this;
    }

    @Override
    public boolean hasNext() {
        return parser.getCurrentToken() != JsonToken.END_ARRAY;
    }

    @Override
    public final ParsedDocument next() {

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
                throw new IllegalStateException("Unable to close the reader.", e);
            }
        }

    }

    private ParsedDocument parse() throws IOException {
        String id = null, body = null;

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

            if (nestedObjCount == 1 && currToken == JsonToken.FIELD_NAME && parser.getText().equals("text")) {
                parser.nextToken();
                log.fine("Found body: " + parser.getText());
                if (body != null) {
                    log.warning(String.format("found multiple texts for argument %s", id));
                }
                body = parser.getText();
            }

        }

        if (id == null || body == null) {
            throw new IllegalStateException("Finished parsing document but id or body are null");
        }

        // Move to next argument - if no argument next token will be END_ARRAY
        parser.nextToken();
        return new ParsedDocument(id, body);
    }
}
