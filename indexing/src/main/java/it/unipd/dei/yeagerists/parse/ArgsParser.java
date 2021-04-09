package it.unipd.dei.yeagerists.parse;

import java.io.BufferedReader;
import java.io.Reader;

public class ArgsParser extends DocumentParser {
    protected ArgsParser(Reader in) {
        super(new BufferedReader(in));
    }

    @Override
    public boolean hasNext() {
        return super.hasNext();
    }

    @Override
    protected ParsedDocument parse() {
        return null;
    }
}
