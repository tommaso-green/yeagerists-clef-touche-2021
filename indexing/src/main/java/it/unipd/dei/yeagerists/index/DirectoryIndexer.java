package it.unipd.dei.yeagerists.index;


import it.unipd.dei.yeagerists.parse.DocumentParser;
import it.unipd.dei.yeagerists.parse.ParsedDocument;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.search.similarities.Similarity;
import org.apache.lucene.store.FSDirectory;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;

@Slf4j
public class DirectoryIndexer {

    private static final int MBYTE = 1024 * 1024;

    private final IndexWriter writer;

    /**
     * The class of the {@code DocumentParser} to be used.
     */
    private final Class<? extends DocumentParser> parserClass;

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
    private final Charset cs;

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
     * @param parserClass           the class of the {@code DocumentParser} to be used.
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
            @NonNull final String charsetName,
            @NonNull final Class<? extends DocumentParser> parserClass) {

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

        this.parserClass = parserClass;

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
            cs = Charset.forName(charsetName);
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

        System.out.printf("%n#### Start indexing ####%n");

        Files.walkFileTree(docsDir, new SimpleFileVisitor<Path>() {
            @Override
            public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
                if (file.getFileName().toString().endsWith(extension)) {
                    System.out.printf("Ignoring file: %s", file.getFileName().toString());
                    return FileVisitResult.CONTINUE;
                }

                DocumentParser dp = DocumentParser.create(parserClass, Files.newBufferedReader(file, cs));
                bytesCount += attrs.size();
                filesCount += 1;

                Document doc = null;

                for (ParsedDocument pd : dp) {

                    doc = new Document();

                    // add the document identifier
                    doc.add(new StringField(ParsedDocument.FIELDS.ID, pd.getIdentifier(), Field.Store.YES));

                    // add the document body
                    doc.add(new BodyField(pd.getBody()));

                    writer.addDocument(doc);

                    docsCount++;

                    // print progress every 10000 indexed documents
                    if (docsCount % 10000 == 0) {
                        System.out.printf("%d document(s) (%d files, %d Mbytes) indexed in %d seconds.%n",
                                docsCount, filesCount, bytesCount / MBYTE,
                                (System.currentTimeMillis() - start) / 1000);
                    }
                }

                return FileVisitResult.CONTINUE;
            }
        });

        writer.commit();

        writer.close();

        System.out.printf("%d document(s) (%d files, %d Mbytes) indexed in %d seconds.%n", docsCount, filesCount,
                bytesCount / MBYTE, (System.currentTimeMillis() - start) / 1000);

        System.out.printf("#### Indexing complete ####%n");
    }
}
