package it.unipd.dei.yeagerists.index;


import it.unipd.dei.yeagerists.parse.ArgsParser;
import it.unipd.dei.yeagerists.parse.ParsedArgument;
import lombok.NonNull;
import lombok.extern.java.Log;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.search.similarities.Similarity;
import org.apache.lucene.store.FSDirectory;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.HashSet;
import java.util.Set;

@Log
public class DirectoryIndexer {

    private static final int MBYTE = 1024 * 1024;

    private final IndexWriter writer;

    /**
     * The directory (and sub-directories) where documents are stored.
     */
    private final Path docsDir;

    /**
     * The extension of the files to be indexed.
     */
    private final String extension;

    /**
     * The charset used for encoding documents.
     */
    private final Charset charset;

    /**
     * The start instant of the indexing.
     */
    private final long start;

    private long filesCount;
    private long docsCount;
    private long bytesCount;

    /**
     * Creates a new indexer.
     *
     * @param analyzer        the {@code Analyzer} to be used.
     * @param similarity      the {@code Similarity} to be used.
     * @param ramBufferSizeMB the size in megabytes of the RAM buffer for indexing documents.
     * @param indexPath       the directory where to store the index.
     * @param docsPath        the directory from which documents have to be read.
     * @param extension       the extension of the files to be indexed.
     * @param charsetName     the name of the charset used for encoding documents.
     * @throws NullPointerException     if any of the parameters is {@code null}.
     * @throws IllegalArgumentException if any of the parameters assumes invalid values.
     */
    public DirectoryIndexer(
            @NonNull final Analyzer analyzer,
            @NonNull final Similarity similarity,
            final int ramBufferSizeMB,
            @NonNull final String indexPath,
            @NonNull final String docsPath,
            @NonNull final String extension,
            @NonNull final String charsetName) {

        if (ramBufferSizeMB <= 0) {
            throw new IllegalArgumentException("RAM buffer size cannot be less than or equal to zero.");
        }

        if (indexPath.isEmpty()) {
            throw new IllegalArgumentException("Index path cannot be empty.");
        }

        if (docsPath.isEmpty()) {
            throw new IllegalArgumentException("Documents path cannot be empty.");
        }

        if (extension.isEmpty()) {
            throw new IllegalArgumentException("File extension cannot be empty.");
        }

        if (charsetName.isEmpty()) {
            throw new IllegalArgumentException("Charset name cannot be empty.");
        }

        final IndexWriterConfig iwc = new IndexWriterConfig(analyzer);
        iwc.setSimilarity(similarity);
        iwc.setRAMBufferSizeMB(ramBufferSizeMB);
        iwc.setOpenMode(IndexWriterConfig.OpenMode.CREATE);
        iwc.setCommitOnClose(true);
        iwc.setUseCompoundFile(true);

        final Path indexDir = Paths.get(indexPath);
        // if the directory does not already exist, create it
        if (Files.notExists(indexDir)) {
            try {
                Files.createDirectory(indexDir);
            } catch (Exception e) {
                throw new IllegalArgumentException(
                        String.format("Unable to create directory %s: %s.", indexDir.toAbsolutePath().toString(),
                                e.getMessage()), e);
            }
        }

        if (!Files.isWritable(indexDir)) {
            throw new IllegalArgumentException(
                    String.format("Index directory %s cannot be written.", indexDir.toAbsolutePath().toString()));
        }

        if (!Files.isDirectory(indexDir)) {
            throw new IllegalArgumentException(String.format("%s expected to be a directory where to write the index.",
                    indexDir.toAbsolutePath().toString()));
        }

        final Path docsDir = Paths.get(docsPath);
        if (!Files.isReadable(docsDir)) {
            throw new IllegalArgumentException(
                    String.format("Documents directory %s cannot be read.", docsDir.toAbsolutePath().toString()));
        }

        if (!Files.isDirectory(docsDir)) {
            throw new IllegalArgumentException(
                    String.format("%s expected to be a directory of documents.", docsDir.toAbsolutePath().toString()));
        }

        this.docsDir = docsDir;

        this.extension = extension;

        try {
            charset = Charset.forName(charsetName);
        } catch (Exception e) {
            throw new IllegalArgumentException(String.format("Unable to create charset %s: %s.", charsetName, e.getMessage()), e);
        }

        this.docsCount = 0;
        this.bytesCount = 0;
        this.filesCount = 0;

        try {
            writer = new IndexWriter(FSDirectory.open(indexDir), iwc);
        } catch (IOException e) {
            throw new IllegalArgumentException(String.format("Unable to create the index writer in directory %s: %s.",
                    indexDir.toAbsolutePath().toString(), e.getMessage()), e);
        }

        this.start = System.currentTimeMillis();
    }

    public void index() throws IOException {

        log.info("%n#### Start indexing ####%n");

        // Stores the IDs of parsed documents
        Set<String> argsIdSet = new HashSet<>();

        Files.walkFileTree(docsDir, new SimpleFileVisitor<Path>() {
            @Override
            public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
                if (!file.getFileName().toString().endsWith(extension)) {
                    log.warning(String.format("Ignoring file: %s", file.getFileName().toString()));
                    return FileVisitResult.CONTINUE;
                }

                ArgsParser parser = null;
                try {
                     parser = new ArgsParser(Files.newBufferedReader(file, charset));
                } catch (IllegalArgumentException e) {
                    log.severe(String.format("Couldn't parse file %s: %s", file.getFileName().toString(), e.getMessage()));
                    return FileVisitResult.CONTINUE;
                }

                bytesCount += attrs.size();
                filesCount += 1;

                Document doc = null;
                argsIdSet.clear();

                for (ParsedArgument arg : parser) {

                    if (!arg.isValid()) {
                        log.warning(String.format("Skipping invalid doc in file %s: %s",
                                file.getFileName().toString(), arg.toString()));
                        continue;
                    }

                    if (argsIdSet.contains(arg.getId())) {
                        log.warning(String.format("Found multiple arguments with the same id %s. The arg will be ignored ", arg.getId()));
                        continue;
                    }
                    argsIdSet.add(arg.getId());

                    doc = new Document();

                    // add the document identifier
                    doc.add(new StringField(ParsedArgument.FIELDS.ID, arg.getId(), Field.Store.YES));

                    // add stance (pro/con)
                    doc.add(new StanceField(arg.getStance()));

                    // add the document text
                    doc.add(new BodyField(arg.getBody()));

                    writer.addDocument(doc);

                    docsCount++;

                    // print progress every 10000 indexed documents
                    if (docsCount % 10000 == 0) {
                        log.info(String.format("%d document(s) (%d files, %d Mbytes) indexed in %d seconds.\n\n",
                                docsCount, filesCount, bytesCount / MBYTE,
                                (System.currentTimeMillis() - start) / 1000));
                    }
                }

                return FileVisitResult.CONTINUE;
            }
        });

        writer.commit();

        writer.close();

        log.info(String.format("%d document(s) (%d files, %d Mbytes) indexed in %d seconds.%n", docsCount, filesCount,
                bytesCount / MBYTE, (System.currentTimeMillis() - start) / 1000));

        log.info("#### Indexing complete ####%n");
    }
}
