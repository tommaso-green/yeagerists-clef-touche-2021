package it.unipd.dei.yeagerists.parse;


import lombok.NonNull;

import java.io.IOException;
import java.io.Reader;
import java.util.Iterator;
import java.util.NoSuchElementException;

public abstract class DocumentParser implements Iterator<ParsedDocument>, Iterable<ParsedDocument> {

    protected boolean next = true;

    protected final Reader in;


    /**
     * Creates a new document parser.
     *
     * @param in the reader to the document(s) to be parsed.
     * @throws NullPointerException if {@code in} is {@code null}.
     */
    protected DocumentParser(@NonNull final Reader in) {
        this.in = in;
    }


    @Override
    public final Iterator<ParsedDocument> iterator() {
        return this;
    }

    @Override
    public boolean hasNext() {
        return next;
    }

    @Override
    public final ParsedDocument next() {

        if (!next) {
            throw new NoSuchElementException("No more documents to parse.");
        }

        try {
            return parse();
        } finally {
            try {
                // end of file
                if (!next) {
                    in.close();
                }
            } catch (IOException e) {
                throw new IllegalStateException("Unable to close the reader.", e);
            }
        }
    }

    /**
     * Creates a new {@code DocumentParser}.
     * <p>
     * It assumes the {@code DocumentParser} has a single-parameter constructor which takes a {@code Reader} as input.
     *
     * @param cls the class of the document parser to be instantiated.
     * @param in  the reader to the document(s) to be parsed.
     * @return a new instance of {@code DocumentParser} for the given class.
     * @throws NullPointerException  if {@code cls} and/or {@code in} are {@code null}.
     * @throws IllegalStateException if something goes wrong in instantiating the class.
     */
    public static DocumentParser create(
            @NonNull Class<? extends DocumentParser> cls,
            @NonNull Reader in) {

        try {
            return cls.getConstructor(Reader.class).newInstance(in);
        } catch (Exception e) {
            throw new IllegalStateException(String.format("Unable to instantiate doc parser %s.", cls.getName()), e);
        }

    }

    protected abstract ParsedDocument parse();
}
